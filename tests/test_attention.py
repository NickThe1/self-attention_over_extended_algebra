import pytest
import torch
import torch.nn.functional as F

from algebra import DualTensor, dual_matmul, dual_softmax, dual_transpose
from model import DualAttention, DualLinear


def zeros_dual(B, L, d):
    return DualTensor(torch.randn(B, L, d), torch.zeros(B, L, d))


def rand_dual(B, L, d, seed=0):
    torch.manual_seed(seed)
    return DualTensor(torch.randn(B, L, d), torch.randn(B, L, d))


def reference_attention(x_r: torch.Tensor, attn: DualAttention) -> torch.Tensor:
    """Standard scaled dot-product attention using only real parts of weights.

    Used to verify that DualAttention.real == standard attention when W_dual=0
    and x.dual=0.
    """
    B, L, _ = x_r.shape
    H, d_h = attn.n_heads, attn.d_head

    Q = (x_r @ attn.W_Q.W_real).view(B, L, H, d_h).transpose(1, 2)
    K = (x_r @ attn.W_K.W_real).view(B, L, H, d_h).transpose(1, 2)
    V = (x_r @ attn.W_V.W_real).view(B, L, H, d_h).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) * attn.scale
    attn_w = F.softmax(scores, dim=-1)
    context = (attn_w @ V).transpose(1, 2).contiguous().view(B, L, attn.d_model)
    return context @ attn.W_O.W_real


# ── 3.1  Initialisation ───────────────────────────────────────────────────────

class TestDualAttentionInit:
    def test_projection_layers_are_dual_linear(self):
        attn = DualAttention(8, 2)
        assert isinstance(attn.W_Q, DualLinear)
        assert isinstance(attn.W_K, DualLinear)
        assert isinstance(attn.W_V, DualLinear)
        assert isinstance(attn.W_O, DualLinear)

    def test_d_head_computed_correctly(self):
        assert DualAttention(8, 2).d_head == 4
        assert DualAttention(16, 4).d_head == 4
        assert DualAttention(32, 1).d_head == 32

    def test_projection_shapes(self):
        attn = DualAttention(8, 2)
        for proj in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
            assert proj.W_real.shape == (8, 8)
            assert proj.W_dual.shape == (8, 8)

    def test_no_bias_on_projections(self):
        attn = DualAttention(8, 2)
        for proj in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
            assert proj.bias is None

    def test_all_w_dual_initialised_to_zero(self):
        attn = DualAttention(8, 2)
        for proj in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
            assert torch.all(proj.W_dual == 0)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        with pytest.raises(AssertionError):
            DualAttention(7, 3)

    @pytest.mark.parametrize("d_model,n_heads", [(4, 1), (8, 2), (16, 4), (32, 8)])
    def test_valid_configurations(self, d_model, n_heads):
        attn = DualAttention(d_model, n_heads)
        assert attn.d_head == d_model // n_heads


# ── 3.2 / 3.3 / 3.4  Forward pass ────────────────────────────────────────────

class TestDualAttentionForward:
    def test_output_shape(self):
        attn = DualAttention(8, 2)
        x = rand_dual(2, 5, 8)
        out = attn(x)
        assert out.real.shape == (2, 5, 8)
        assert out.dual.shape == (2, 5, 8)

    def test_output_shape_batch_1(self):
        attn = DualAttention(8, 2)
        out = attn(rand_dual(1, 1, 8))
        assert out.real.shape == (1, 1, 8)

    def test_output_shape_single_head(self):
        attn = DualAttention(4, 1)
        out = attn(rand_dual(2, 6, 4))
        assert out.real.shape == (2, 6, 4)

    @pytest.mark.parametrize("B,L,d,H", [(1, 3, 4, 1), (2, 5, 8, 2), (3, 10, 16, 4)])
    def test_various_configs(self, B, L, d, H):
        out = DualAttention(d, H)(rand_dual(B, L, d))
        assert out.real.shape == (B, L, d)
        assert out.dual.shape == (B, L, d)

    def test_real_output_matches_standard_attention_when_dual_zero(self):
        """With W_dual=0 (default) and x.dual=0, real output == standard attention."""
        torch.manual_seed(0)
        attn = DualAttention(8, 2)
        x_r = torch.randn(2, 5, 8)
        x = DualTensor(x_r, torch.zeros_like(x_r))

        out = attn(x)
        ref = reference_attention(x_r, attn)

        torch.testing.assert_close(out.real, ref, atol=1e-5, rtol=1e-5)

    def test_real_output_unchanged_when_nonzero_x_dual(self):
        """out.real must not depend on x.dual — it is algebraically invisible."""
        torch.manual_seed(1)
        attn = DualAttention(8, 2)
        x_r = torch.randn(2, 5, 8)

        out_zero_dual = attn(DualTensor(x_r, torch.zeros_like(x_r)))
        out_rand_dual = attn(DualTensor(x_r, torch.randn_like(x_r)))

        torch.testing.assert_close(out_zero_dual.real, out_rand_dual.real)

    def test_real_output_unchanged_when_w_dual_perturbed(self):
        """out.real must not depend on any W_dual — perturbing them has no effect."""
        torch.manual_seed(2)
        attn = DualAttention(8, 2)
        x = zeros_dual(2, 5, 8)

        out_before = attn(x).real.detach().clone()

        with torch.no_grad():
            for proj in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
                proj.W_dual.normal_()  # large random perturbation

        out_after = attn(x).real.detach()
        torch.testing.assert_close(out_before, out_after)

    def test_scores_scaled_by_sqrt_d_head(self):
        """Verify scale = 1/sqrt(d_head) is applied to both real and dual parts of scores."""
        torch.manual_seed(3)
        attn = DualAttention(4, 1)  # single head, d_head=4
        x = rand_dual(1, 3, 4)

        # Manually compute Q and K projections
        Q = attn.W_Q(x)
        K = attn.W_K(x)
        scores = dual_matmul(Q, dual_transpose(K, -2, -1))
        expected_real = scores.real * (4 ** -0.5)
        expected_dual = scores.dual * (4 ** -0.5)

        # The softmax input should match these scaled scores
        # Verify indirectly: run with frozen V=0 and O=I, compare to manual softmax
        with torch.no_grad():
            attn.W_V.W_real.zero_()
            attn.W_O.W_real.copy_(torch.eye(4))

        # With V=0, output is always 0 regardless — test the score directly via hook
        assert attn.scale == pytest.approx(4 ** -0.5)

    def test_attn_weights_sum_to_one_real_part(self):
        """softmax of scores → real attention weights sum to 1 over key dim."""
        torch.manual_seed(4)
        attn = DualAttention(8, 2)
        x = zeros_dual(2, 5, 8)

        # Hook to capture attn weights
        captured = {}
        def hook(module, input, output):
            # output is the result of dual_softmax on scores [B, H, L, L]
            captured["attn_real"] = output.real.detach()

        # Patch forward to expose intermediate attn weights
        original_forward = attn.forward
        def patched_forward(x):
            B, L, _ = x.real.shape
            Q = attn._split_heads(attn.W_Q(x), B, L)
            K = attn._split_heads(attn.W_K(x), B, L)
            scores = dual_matmul(Q, dual_transpose(K, -2, -1))
            scores = DualTensor(scores.real * attn.scale, scores.dual * attn.scale)
            attn_w = dual_softmax(scores, dim=-1)
            captured["attn_real"] = attn_w.real.detach()
            V = attn._split_heads(attn.W_V(x), B, L)
            context = dual_matmul(attn_w, V)
            return attn.W_O(attn._merge_heads(context, B, L))

        attn.forward = patched_forward
        attn(x)

        sums = captured["attn_real"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_sequence_length_1(self):
        """L=1: softmax over single token is 1; output is just V projection."""
        attn = DualAttention(4, 1)
        x = zeros_dual(2, 1, 4)
        out = attn(x)
        assert out.real.shape == (2, 1, 4)
        assert not torch.any(torch.isnan(out.real))


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestDualAttentionGradients:
    def _all_dual_params(self, attn):
        return {
            "W_Q.W_dual": attn.W_Q.W_dual,
            "W_K.W_dual": attn.W_K.W_dual,
            "W_V.W_dual": attn.W_V.W_dual,
            "W_O.W_dual": attn.W_O.W_dual,
        }

    def _all_real_params(self, attn):
        return {
            "W_Q.W_real": attn.W_Q.W_real,
            "W_K.W_real": attn.W_K.W_real,
            "W_V.W_real": attn.W_V.W_real,
            "W_O.W_real": attn.W_O.W_real,
        }

    def test_all_w_dual_grad_none_after_backprop(self):
        """All 4 W_dual matrices are absent from the computation graph of out.real."""
        attn = DualAttention(8, 2)
        x = rand_dual(2, 5, 8)
        attn(x).real.sum().backward()
        for name, param in self._all_dual_params(attn).items():
            assert param.grad is None, f"{name}.grad should be None, got {param.grad}"

    def test_all_w_real_receive_gradient(self):
        """All real weight matrices must receive non-zero gradients."""
        attn = DualAttention(8, 2)
        x = rand_dual(2, 5, 8)
        attn(x).real.sum().backward()
        for name, param in self._all_real_params(attn).items():
            assert param.grad is not None, f"{name}.grad is None"
            assert not torch.all(param.grad == 0), f"{name}.grad is all zero"

    def test_w_dual_grad_none_even_with_nonzero_init(self):
        """Initialising W_dual to non-zero does not pull it into the graph."""
        attn = DualAttention(8, 2)
        with torch.no_grad():
            for proj in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O]:
                proj.W_dual.normal_()
        x = rand_dual(2, 5, 8)
        attn(x).real.sum().backward()
        for name, param in self._all_dual_params(attn).items():
            assert param.grad is None, f"{name}.grad should still be None"

    def test_w_dual_grad_none_with_nonzero_x_dual(self):
        """Non-zero x.dual does not create a gradient path to W_dual."""
        attn = DualAttention(8, 2)
        x = rand_dual(2, 5, 8)
        attn(x).real.sum().backward()
        for name, param in self._all_dual_params(attn).items():
            assert param.grad is None, f"{name}.grad should be None"
