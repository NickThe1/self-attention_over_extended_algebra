import pytest
import torch
import torch.nn.functional as F

from algebra import DualTensor, dual_add, dual_mul, dual_matmul, dual_softmax, dual_transpose


def rand_dual(*shape, seed: int = 0) -> DualTensor:
    torch.manual_seed(seed)
    return DualTensor(torch.randn(*shape), torch.randn(*shape))


# ── 1.1 DualTensor ────────────────────────────────────────────────────────────

class TestDualTensor:
    def test_stores_real_and_dual(self):
        r, d = torch.ones(3), torch.zeros(3)
        dt = DualTensor(r, d)
        assert dt.real is r and dt.dual is d

    def test_shape_property(self):
        dt = DualTensor(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))
        assert dt.shape == (2, 3, 4)

    def test_scalar_tensor(self):
        dt = DualTensor(torch.tensor(1.0), torch.tensor(2.0))
        assert dt.shape == ()

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            DualTensor(torch.zeros(2, 3), torch.zeros(3, 2))

    def test_shape_mismatch_different_ndim_raises(self):
        with pytest.raises(AssertionError):
            DualTensor(torch.zeros(4), torch.zeros(2, 2))

    @pytest.mark.parametrize("shape", [(1,), (4,), (2, 3), (2, 3, 4), (1, 1, 1)])
    def test_various_shapes(self, shape):
        dt = DualTensor(torch.zeros(*shape), torch.zeros(*shape))
        assert dt.shape == shape


# ── 1.2 dual_add ─────────────────────────────────────────────────────────────

class TestDualAdd:
    def test_real_parts_added(self):
        x = DualTensor(torch.tensor([1., 2.]), torch.zeros(2))
        y = DualTensor(torch.tensor([3., 4.]), torch.zeros(2))
        assert torch.allclose(dual_add(x, y).real, torch.tensor([4., 6.]))

    def test_dual_parts_added(self):
        x = DualTensor(torch.zeros(2), torch.tensor([1., 2.]))
        y = DualTensor(torch.zeros(2), torch.tensor([3., 4.]))
        assert torch.allclose(dual_add(x, y).dual, torch.tensor([4., 6.]))

    def test_commutativity(self):
        x, y = rand_dual(3, 4, seed=0), rand_dual(3, 4, seed=1)
        r1, r2 = dual_add(x, y), dual_add(y, x)
        assert torch.allclose(r1.real, r2.real)
        assert torch.allclose(r1.dual, r2.dual)

    def test_additive_identity(self):
        x = rand_dual(3, 4)
        zero = DualTensor(torch.zeros(3, 4), torch.zeros(3, 4))
        r = dual_add(x, zero)
        assert torch.allclose(r.real, x.real)
        assert torch.allclose(r.dual, x.dual)

    def test_associativity(self):
        x, y, z = rand_dual(5, seed=0), rand_dual(5, seed=1), rand_dual(5, seed=2)
        r1 = dual_add(dual_add(x, y), z)
        r2 = dual_add(x, dual_add(y, z))
        assert torch.allclose(r1.real, r2.real)
        assert torch.allclose(r1.dual, r2.dual)


# ── 1.2 dual_mul ─────────────────────────────────────────────────────────────

class TestDualMul:
    def test_alpha_squared_is_zero(self):
        """The defining property of the algebra: α² = 0."""
        alpha = DualTensor(torch.tensor(0.), torch.tensor(1.))
        r = dual_mul(alpha, alpha)
        assert torch.allclose(r.real, torch.tensor(0.))
        assert torch.allclose(r.dual, torch.tensor(0.))

    def test_alpha_squared_is_zero_batched(self):
        """α² = 0 holds element-wise on arbitrary shapes."""
        alpha = DualTensor(torch.zeros(3, 4), torch.ones(3, 4))
        r = dual_mul(alpha, alpha)
        assert torch.allclose(r.real, torch.zeros(3, 4))
        assert torch.allclose(r.dual, torch.zeros(3, 4))

    def test_real_part_formula(self):
        a, c = torch.tensor([2., 3.]), torch.tensor([4., 5.])
        x = DualTensor(a, torch.zeros(2))
        y = DualTensor(c, torch.zeros(2))
        assert torch.allclose(dual_mul(x, y).real, a * c)

    def test_dual_part_formula(self):
        """dual = a*d + b*c."""
        a, b = torch.tensor([2.]), torch.tensor([3.])
        c, d = torch.tensor([4.]), torch.tensor([5.])
        r = dual_mul(DualTensor(a, b), DualTensor(c, d))
        assert torch.allclose(r.dual, a * d + b * c)  # 2*5 + 3*4 = 22

    def test_commutativity(self):
        x, y = rand_dual(5, seed=0), rand_dual(5, seed=1)
        r1, r2 = dual_mul(x, y), dual_mul(y, x)
        assert torch.allclose(r1.real, r2.real)
        assert torch.allclose(r1.dual, r2.dual)

    def test_multiplicative_identity(self):
        """Multiplying by 1+α·0 leaves x unchanged."""
        x = rand_dual(4)
        one = DualTensor(torch.ones(4), torch.zeros(4))
        r = dual_mul(x, one)
        assert torch.allclose(r.real, x.real)
        assert torch.allclose(r.dual, x.dual)

    def test_real_only_is_standard_product(self):
        """b=0 on both sides → standard real multiplication, zero dual."""
        a, c = torch.randn(6), torch.randn(6)
        r = dual_mul(DualTensor(a, torch.zeros(6)), DualTensor(c, torch.zeros(6)))
        assert torch.allclose(r.real, a * c)
        assert torch.allclose(r.dual, torch.zeros(6))

    def test_distributivity(self):
        """x*(y+z) == x*y + x*z."""
        x, y, z = rand_dual(4, seed=0), rand_dual(4, seed=1), rand_dual(4, seed=2)
        lhs = dual_mul(x, dual_add(y, z))
        rhs = dual_add(dual_mul(x, y), dual_mul(x, z))
        assert torch.allclose(lhs.real, rhs.real)
        assert torch.allclose(lhs.dual, rhs.dual)


# ── 1.3 dual_matmul ──────────────────────────────────────────────────────────

class TestDualMatmul:
    def test_real_only_matches_standard_matmul(self):
        """b=0 on both → standard matmul, zero dual."""
        A, B = torch.randn(3, 4), torch.randn(4, 5)
        r = dual_matmul(DualTensor(A, torch.zeros_like(A)), DualTensor(B, torch.zeros_like(B)))
        assert torch.allclose(r.real, A @ B)
        assert torch.allclose(r.dual, torch.zeros(3, 5))

    def test_dual_part_formula_brute_force(self):
        """dual = x.r @ w.d + x.d @ w.r — verified manually."""
        A, B = torch.randn(2, 3), torch.randn(3, 4)
        C, D = torch.randn(2, 3), torch.randn(3, 4)
        r = dual_matmul(DualTensor(A, C), DualTensor(B, D))
        assert torch.allclose(r.real, A @ B)
        assert torch.allclose(r.dual, A @ D + C @ B)

    def test_pure_dual_inputs_give_zero(self):
        """dual_matmul of two a=0 tensors → both parts zero: mirrors α²=0."""
        A, B = torch.randn(3, 4), torch.randn(4, 5)
        r = dual_matmul(DualTensor(torch.zeros(3, 4), A), DualTensor(torch.zeros(4, 5), B))
        assert torch.allclose(r.real, torch.zeros(3, 5))
        assert torch.allclose(r.dual, torch.zeros(3, 5))

    def test_batched_shape(self):
        """[B, L, d] @ [d, d_out] — standard attention projection shape."""
        X_r, X_d = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
        W_r, W_d = torch.randn(4, 6), torch.randn(4, 6)
        r = dual_matmul(DualTensor(X_r, X_d), DualTensor(W_r, W_d))
        assert r.real.shape == (2, 5, 6)
        assert r.dual.shape == (2, 5, 6)
        assert torch.allclose(r.real, X_r @ W_r)
        assert torch.allclose(r.dual, X_r @ W_d + X_d @ W_r)

    def test_non_square_matrices(self):
        A, B = torch.randn(7, 3), torch.randn(3, 11)
        r = dual_matmul(DualTensor(A, torch.zeros_like(A)), DualTensor(B, torch.zeros_like(B)))
        assert r.real.shape == (7, 11)

    def test_dual_only_weight_gives_zero_real(self):
        """x.r @ 0 = 0, so real part is zero when w.real=0."""
        A, D = torch.randn(3, 4), torch.randn(4, 5)
        r = dual_matmul(DualTensor(A, torch.zeros_like(A)), DualTensor(torch.zeros(4, 5), D))
        assert torch.allclose(r.real, torch.zeros(3, 5))
        assert torch.allclose(r.dual, A @ D)


# ── 1.4 dual_softmax ─────────────────────────────────────────────────────────

class TestDualSoftmax:
    def test_real_part_is_standard_softmax(self):
        x = rand_dual(6)
        r = dual_softmax(x)
        assert torch.allclose(r.real, F.softmax(x.real, dim=-1))

    def test_real_part_sums_to_one(self):
        x = rand_dual(5)
        assert torch.allclose(dual_softmax(x).real.sum(), torch.tensor(1.))

    def test_dual_part_sums_to_zero(self):
        """Softmax Jacobian maps into tangent space of the simplex: Jb sums to 0."""
        x = rand_dual(7)
        assert torch.allclose(dual_softmax(x).dual.sum(dim=-1), torch.tensor(0.), atol=1e-6)

    def test_dual_part_sums_to_zero_batched(self):
        x = rand_dual(3, 5, 8)
        ds = dual_softmax(x).dual
        assert torch.allclose(ds.sum(dim=-1), torch.zeros(3, 5), atol=1e-6)

    def test_zero_dual_gives_zero_dual(self):
        x = DualTensor(torch.randn(5), torch.zeros(5))
        assert torch.allclose(dual_softmax(x).dual, torch.zeros(5))

    def test_matches_numerical_jacobian(self):
        """dual ≈ (softmax(a + ε·b) - softmax(a)) / ε for small ε."""
        torch.manual_seed(42)
        a, b = torch.randn(8), torch.randn(8)
        eps = 1e-4
        r = dual_softmax(DualTensor(a, b))
        numerical = (F.softmax(a + eps * b, dim=-1) - F.softmax(a, dim=-1)) / eps
        torch.testing.assert_close(r.dual, numerical, atol=1e-3, rtol=1e-3)

    def test_matches_numerical_jacobian_batched(self):
        torch.manual_seed(7)
        a, b = torch.randn(3, 6), torch.randn(3, 6)
        eps = 1e-4
        r = dual_softmax(DualTensor(a, b))
        numerical = (F.softmax(a + eps * b, dim=-1) - F.softmax(a, dim=-1)) / eps
        torch.testing.assert_close(r.dual, numerical, atol=1e-3, rtol=1e-3)

    def test_large_values_no_nan(self):
        """Torch softmax uses max-subtraction internally — no overflow."""
        a = torch.tensor([1000., 1001., 1002.])
        b = torch.randn(3)
        r = dual_softmax(DualTensor(a, b))
        assert not torch.any(torch.isnan(r.real))
        assert not torch.any(torch.isnan(r.dual))

    def test_uniform_input(self):
        """softmax([v,v,v]) = [1/3,1/3,1/3]; dual = (1/3)*(b - mean(b)) when s is uniform."""
        a = torch.ones(3) * 5.0
        b = torch.tensor([1., 0., -1.])
        r = dual_softmax(DualTensor(a, b))
        assert torch.allclose(r.real, torch.ones(3) / 3, atol=1e-6)
        expected_dual = (1.0 / 3) * (b - b.mean())
        assert torch.allclose(r.dual, expected_dual, atol=1e-6)

    def test_shape_preserved(self):
        x = rand_dual(2, 5, 8)
        r = dual_softmax(x)
        assert r.real.shape == x.shape
        assert r.dual.shape == x.shape


# ── 1.5 dual_transpose ───────────────────────────────────────────────────────

class TestDualTranspose:
    def test_both_parts_transposed(self):
        x = rand_dual(3, 4)
        r = dual_transpose(x, 0, 1)
        assert torch.allclose(r.real, x.real.T)
        assert torch.allclose(r.dual, x.dual.T)

    def test_double_transpose_is_identity(self):
        x = rand_dual(3, 4)
        r = dual_transpose(dual_transpose(x, 0, 1), 0, 1)
        assert torch.allclose(r.real, x.real)
        assert torch.allclose(r.dual, x.dual)

    def test_batch_transpose(self):
        """transpose(-2, -1) is used for attention score computation."""
        x = rand_dual(2, 5, 8)
        r = dual_transpose(x, -2, -1)
        assert r.real.shape == (2, 8, 5)
        assert r.dual.shape == (2, 8, 5)
        assert torch.allclose(r.real, x.real.transpose(-2, -1))
        assert torch.allclose(r.dual, x.dual.transpose(-2, -1))

    def test_shape_changes_correctly(self):
        x = rand_dual(7, 3)
        r = dual_transpose(x, 0, 1)
        assert r.real.shape == (3, 7)
        assert r.dual.shape == (3, 7)


# ── algebraic identity cross-checks ──────────────────────────────────────────

class TestAlgebraicIdentities:
    def test_function_lift_principle_via_softmax(self):
        """f(a+αb) = f(a) + α·f'(a)·b — verified via dual_softmax matching autograd."""
        a = torch.randn(5, requires_grad=True)
        b = torch.randn(5)
        # autograd path
        s = torch.softmax(a, dim=-1)
        s.backward(b)
        autograd_jvp = a.grad.clone()
        # dual path
        dual_jvp = dual_softmax(DualTensor(a.detach(), b)).dual
        torch.testing.assert_close(dual_jvp, autograd_jvp, atol=1e-6, rtol=1e-6)

    def test_matmul_consistent_with_mul_for_scalars(self):
        """For 1x1 matrices, dual_matmul and dual_mul must agree."""
        x = DualTensor(torch.tensor([[2.]]), torch.tensor([[3.]]))
        y = DualTensor(torch.tensor([[4.]]), torch.tensor([[5.]]))
        mat = dual_matmul(x, y)
        mul = dual_mul(DualTensor(torch.tensor(2.), torch.tensor(3.)),
                       DualTensor(torch.tensor(4.), torch.tensor(5.)))
        assert torch.allclose(mat.real.squeeze(), mul.real)
        assert torch.allclose(mat.dual.squeeze(), mul.dual)

    def test_dual_mul_alpha_squared_via_matmul(self):
        """α²=0 should also hold when expressed as 1x1 matmul."""
        alpha = DualTensor(torch.zeros(1, 1), torch.ones(1, 1))
        r = dual_matmul(alpha, alpha)
        assert torch.allclose(r.real, torch.zeros(1, 1))
        assert torch.allclose(r.dual, torch.zeros(1, 1))
