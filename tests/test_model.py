import pytest
import torch
import torch.nn as nn

from algebra import DualTensor, dual_matmul
from model import DualLinear


# ── 2.1 Parameters ───────────────────────────────────────────────────────────

class TestDualLinearParameters:
    def test_w_real_shape(self):
        assert DualLinear(4, 6).W_real.shape == (4, 6)

    def test_w_dual_shape(self):
        assert DualLinear(4, 6).W_dual.shape == (4, 6)

    def test_bias_shape(self):
        assert DualLinear(4, 6).bias.shape == (6,)

    def test_no_bias_is_none(self):
        assert DualLinear(4, 6, bias=False).bias is None

    def test_w_dual_initialised_to_zero(self):
        """W_dual starts at zero so the model is a real network at init."""
        assert torch.all(DualLinear(8, 16).W_dual == 0)

    def test_w_real_not_zero(self):
        """W_real uses Kaiming uniform — should not be all zeros."""
        assert not torch.all(DualLinear(4, 6).W_real == 0)

    def test_registered_as_parameters(self):
        layer = DualLinear(4, 6)
        names = {n for n, _ in layer.named_parameters()}
        assert "W_real" in names
        assert "W_dual" in names
        assert "bias" in names

    def test_no_bias_not_in_parameters(self):
        layer = DualLinear(4, 6, bias=False)
        names = {n for n, _ in layer.named_parameters()}
        assert "bias" not in names

    @pytest.mark.parametrize("in_f,out_f", [(1, 1), (4, 4), (8, 32), (32, 8)])
    def test_various_sizes(self, in_f, out_f):
        layer = DualLinear(in_f, out_f)
        assert layer.W_real.shape == (in_f, out_f)
        assert layer.W_dual.shape == (in_f, out_f)


# ── 2.2 Forward pass ─────────────────────────────────────────────────────────

class TestDualLinearForward:
    def test_output_shape_2d(self):
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        out = layer(x)
        assert out.real.shape == (3, 6)
        assert out.dual.shape == (3, 6)

    def test_output_shape_3d_batched(self):
        """Attention projection shape: [B, L, d_in] → [B, L, d_out]."""
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(2, 5, 4), torch.randn(2, 5, 4))
        out = layer(x)
        assert out.real.shape == (2, 5, 6)
        assert out.dual.shape == (2, 5, 6)

    def test_real_part_formula(self):
        """real = x.real @ W_real + bias."""
        layer = DualLinear(4, 6, bias=True)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        out = layer(x)
        expected = x.real @ layer.W_real + layer.bias
        assert torch.allclose(out.real, expected)

    def test_dual_part_formula(self):
        """dual = x.real @ W_dual + x.dual @ W_real."""
        layer = DualLinear(4, 6, bias=False)
        with torch.no_grad():
            layer.W_dual.uniform_(-0.5, 0.5)  # make W_dual non-zero
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        out = layer(x)
        expected_dual = x.real @ layer.W_dual + x.dual @ layer.W_real
        assert torch.allclose(out.dual, expected_dual)

    def test_bias_added_to_real_only(self):
        """Bias shifts real part; dual part is unaffected."""
        layer = DualLinear(4, 6, bias=True)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        out = layer(x)
        out_no_bias = dual_matmul(x, DualTensor(layer.W_real, layer.W_dual))
        assert torch.allclose(out.real, out_no_bias.real + layer.bias)
        assert torch.allclose(out.dual, out_no_bias.dual)

    def test_no_bias_forward(self):
        layer = DualLinear(4, 6, bias=False)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        out = layer(x)
        assert torch.allclose(out.real, x.real @ layer.W_real)

    def test_matches_nn_linear_when_w_dual_zero(self):
        """With W_dual=0 (default), real output matches nn.Linear with same weights."""
        torch.manual_seed(0)
        layer = DualLinear(4, 6, bias=True)
        # nn.Linear stores weight as [out, in]; our layer stores [in, out]
        ref = nn.Linear(4, 6, bias=True)
        ref.weight.data = layer.W_real.T.detach().clone()
        ref.bias.data = layer.bias.detach().clone()

        x_r = torch.randn(3, 4)
        out = layer(DualTensor(x_r, torch.zeros_like(x_r)))
        assert torch.allclose(out.real, ref(x_r), atol=1e-6)

    def test_zero_input_gives_bias(self):
        """f(0) = bias (real part) and 0 (dual part)."""
        layer = DualLinear(4, 6, bias=True)
        x = DualTensor(torch.zeros(3, 4), torch.zeros(3, 4))
        out = layer(x)
        assert torch.allclose(out.real, layer.bias.expand(3, -1))
        assert torch.allclose(out.dual, torch.zeros(3, 6))

    def test_zero_dual_input_gives_zero_dual_output_when_w_dual_zero(self):
        """If x.dual=0 and W_dual=0, dual output is zero."""
        layer = DualLinear(4, 6)  # W_dual=0 by default
        x = DualTensor(torch.randn(3, 4), torch.zeros(3, 4))
        out = layer(x)
        assert torch.allclose(out.dual, torch.zeros(3, 6))


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestDualLinearGradients:
    def test_w_real_receives_gradient(self):
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        layer(x).real.sum().backward()
        assert layer.W_real.grad is not None
        assert not torch.allclose(layer.W_real.grad, torch.zeros(4, 6))

    def test_w_dual_grad_is_zero_when_loss_on_real(self):
        """W_dual is absent from out.real's computation graph entirely.

        PyTorch leaves .grad=None rather than allocating a zero tensor when a
        parameter is never visited by autograd. None is strictly stronger than a
        zero gradient — it confirms W_dual is algebraically dead.
        """
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        layer(x).real.sum().backward()
        grad = layer.W_dual.grad
        assert grad is None or torch.allclose(grad, torch.zeros(4, 6))

    def test_w_dual_not_in_computation_graph_of_real_output(self):
        """Confirm W_dual.grad is None — it was never reached by autograd."""
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        layer(x).real.sum().backward()
        assert layer.W_dual.grad is None

    def test_w_dual_grad_zero_regardless_of_nonzero_x_dual(self):
        """Large x.dual does not pull W_dual into the computation graph."""
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.ones(3, 4) * 100.0)
        layer(x).real.sum().backward()
        assert layer.W_dual.grad is None

    def test_bias_receives_gradient(self):
        layer = DualLinear(4, 6)
        x = DualTensor(torch.randn(3, 4), torch.randn(3, 4))
        layer(x).real.sum().backward()
        assert layer.bias.grad is not None

    def test_gradient_matches_manual_computation(self):
        """∂(sum of real output)/∂W_real == x.real.T @ ones."""
        layer = DualLinear(4, 6, bias=False)
        x_r = torch.randn(3, 4)
        x = DualTensor(x_r, torch.zeros(3, 4))
        layer(x).real.sum().backward()
        expected_grad = x_r.T @ torch.ones(3, 6)
        assert torch.allclose(layer.W_real.grad, expected_grad)
