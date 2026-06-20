import pytest
import torch

from model import DualTransformerClassifier
from data.synthetic import FirstLastMatchDataset, make_dataloaders
from training.loop import evaluate, run_training
from training.diagnostics import (
    experiment_a_gradient_norms,
    experiment_b_ablation,
    experiment_c_perturbation,
    DualOutputClassifier,
)

VOCAB = 32
D = 16
H = 2
DEVICE = torch.device("cpu")


def make_model(seed=0):
    torch.manual_seed(seed)
    return DualTransformerClassifier(vocab_size=VOCAB, d_model=D, n_heads=H)


def tiny_loaders(n_train=128, n_test=64, batch_size=32, seed=0):
    return make_dataloaders(n_train, n_test, batch_size=batch_size, seed=seed)


def trained_model(n_epochs=5, seed=0):
    torch.manual_seed(seed)
    train_loader, test_loader = tiny_loaders()
    model = make_model(seed)
    run_training(model, train_loader, test_loader, n_epochs=n_epochs, device=DEVICE)
    return model, test_loader


# ── Exp A: gradient norms ─────────────────────────────────────────────────────

class TestExpA:
    def _run(self, seed=0):
        model = make_model(seed)
        train_loader, _ = tiny_loaders()
        return experiment_a_gradient_norms(model, train_loader, DEVICE)

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_keys_match_named_parameters(self):
        model = make_model()
        norms = experiment_a_gradient_norms(model, tiny_loaders()[0], DEVICE)
        expected = {name for name, _ in model.named_parameters()}
        assert set(norms.keys()) == expected

    def test_values_are_lists_of_floats(self):
        norms = self._run()
        for name, lst in norms.items():
            assert isinstance(lst, list), f"{name}: expected list"
            assert all(isinstance(v, float) for v in lst), f"{name}: expected floats"

    def test_n_entries_equals_n_batches(self):
        model = make_model()
        train_loader, _ = tiny_loaders(n_train=128, batch_size=32)
        norms = experiment_a_gradient_norms(model, train_loader, DEVICE)
        n_batches = len(train_loader)
        for name, lst in norms.items():
            assert len(lst) == n_batches, f"{name}: expected {n_batches} entries, got {len(lst)}"

    def test_dual_param_norms_all_zero(self):
        """W_dual parameters never enter the computation graph → norm 0.0 every batch."""
        norms = self._run()
        dual_keys = [
            "embedding.emb_dual.weight",
            "attention.W_Q.W_dual",
            "attention.W_K.W_dual",
            "attention.W_V.W_dual",
            "attention.W_O.W_dual",
        ]
        for key in dual_keys:
            assert all(v == 0.0 for v in norms[key]), (
                f"{key} has non-zero norm in some batch: {norms[key]}"
            )

    def test_real_param_norms_positive_on_average(self):
        """Real parameters receive non-zero gradients on average."""
        norms = self._run()
        real_keys = [
            "embedding.emb_real.weight",
            "pos_emb.weight",
            "attention.W_Q.W_real",
            "attention.W_K.W_real",
            "attention.W_V.W_real",
            "attention.W_O.W_real",
            "classifier.weight",
            "classifier.bias",
        ]
        for key in real_keys:
            mean_norm = sum(norms[key]) / len(norms[key])
            assert mean_norm > 0.0, f"{key} has zero mean gradient norm"

    def test_all_norms_non_negative(self):
        norms = self._run()
        for name, lst in norms.items():
            assert all(v >= 0.0 for v in lst), f"{name} has negative norm"

    def test_dual_norms_zero_from_first_batch(self):
        """Dead-weight property holds from the very first gradient step."""
        norms = self._run()
        dual_keys = [k for k in norms if "dual" in k]
        for key in dual_keys:
            assert norms[key][0] == 0.0, f"{key}: first-batch norm should be 0.0"


# ── Exp B: ablation ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def exp_b_result():
    torch.manual_seed(0)
    train_loader, test_loader = tiny_loaders(n_train=256, n_test=128, batch_size=32)
    return experiment_b_ablation(
        train_loader, test_loader, DEVICE,
        vocab_size=VOCAB, d_model=D, n_heads=H, n_epochs=5,
    )


class TestExpB:
    def test_returns_correct_keys(self, exp_b_result):
        assert set(exp_b_result.keys()) == {"dual", "real_only"}

    def test_accuracies_in_unit_interval(self, exp_b_result):
        for key, acc in exp_b_result.items():
            assert 0.0 <= acc <= 1.0, f"{key}: acc={acc} out of [0,1]"

    def test_accuracies_are_floats(self, exp_b_result):
        assert isinstance(exp_b_result["dual"], float)
        assert isinstance(exp_b_result["real_only"], float)

    def test_dual_and_real_only_within_tolerance(self, exp_b_result):
        """Freezing W_dual should not materially change accuracy."""
        diff = abs(exp_b_result["dual"] - exp_b_result["real_only"])
        assert diff < 0.15, (
            f"Ablation difference too large: dual={exp_b_result['dual']:.3f}, "
            f"real_only={exp_b_result['real_only']:.3f}, diff={diff:.3f}"
        )

    def test_both_beat_random(self, exp_b_result):
        """Both variants should learn something after 5 epochs."""
        assert exp_b_result["dual"] > 0.45
        assert exp_b_result["real_only"] > 0.45


# ── Exp C: perturbation ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def exp_c_setup():
    model, test_loader = trained_model(n_epochs=5)
    baseline = evaluate(model, test_loader, DEVICE)["acc"]
    result = experiment_c_perturbation(model, test_loader, DEVICE, sigmas=[0.1, 1.0, 10.0])
    return result, baseline, model, test_loader


class TestExpC:
    def test_returns_dict(self, exp_c_setup):
        result, *_ = exp_c_setup
        assert isinstance(result, dict)

    def test_keys_are_sigmas(self, exp_c_setup):
        result, *_ = exp_c_setup
        assert set(result.keys()) == {0.1, 1.0, 10.0}

    def test_all_accuracies_in_unit_interval(self, exp_c_setup):
        result, *_ = exp_c_setup
        for sigma, acc in result.items():
            assert 0.0 <= acc <= 1.0, f"sigma={sigma}: acc={acc}"

    def test_accuracy_invariant_to_perturbation(self, exp_c_setup):
        """Any sigma leaves accuracy unchanged — W_dual is dead."""
        result, baseline, *_ = exp_c_setup
        for sigma, acc in result.items():
            assert acc == pytest.approx(baseline, abs=1e-6), (
                f"sigma={sigma}: acc={acc:.6f} != baseline={baseline:.6f}"
            )

    def test_w_dual_restored_after_experiment(self, exp_c_setup):
        """The function must restore W_dual to its original values."""
        result, _, model, test_loader = exp_c_setup
        result2 = experiment_c_perturbation(model, test_loader, DEVICE, sigmas=[1.0])
        assert result2[1.0] == pytest.approx(result[1.0], abs=1e-6)

    def test_large_sigma_still_no_change(self, exp_c_setup):
        """Even extreme σ=100 must leave accuracy unchanged."""
        _, baseline, model, test_loader = exp_c_setup
        result = experiment_c_perturbation(model, test_loader, DEVICE, sigmas=[100.0])
        assert result[100.0] == pytest.approx(baseline, abs=1e-6)

    def test_custom_sigmas(self, exp_c_setup):
        _, _, model, test_loader = exp_c_setup
        result = experiment_c_perturbation(model, test_loader, DEVICE, sigmas=[0.5, 5.0])
        assert set(result.keys()) == {0.5, 5.0}


# ── 7.4  Minimal fix ─────────────────────────────────────────────────────────

class TestDualOutputClassifier:
    def make_fix(self, seed=0):
        torch.manual_seed(seed)
        return DualOutputClassifier(vocab_size=VOCAB, d_model=D, n_heads=H)

    def make_ids(self, B=2, L=16):
        return torch.randint(0, VOCAB, (B, L))

    def test_output_shape(self):
        model = self.make_fix()
        out = model(self.make_ids())
        assert out.shape == (2, 2)

    def test_output_is_tensor(self):
        out = self.make_fix()(self.make_ids())
        assert isinstance(out, torch.Tensor)

    def test_w_dual_receives_gradient(self):
        """Core fix: W_dual must now have a non-None gradient after backward."""
        model = self.make_fix()
        model(self.make_ids()).sum().backward()
        dual_keys = [
            "backbone.embedding.emb_dual.weight",
            "backbone.attention.W_Q.W_dual",
            "backbone.attention.W_K.W_dual",
            "backbone.attention.W_V.W_dual",
            "backbone.attention.W_O.W_dual",
        ]
        for name, param in model.named_parameters():
            if name in dual_keys:
                assert param.grad is not None, f"{name}.grad is None — fix did not work"

    def test_output_changes_when_w_dual_perturbed(self):
        """After the fix, perturbing W_dual must change the output."""
        model = self.make_fix()
        ids = self.make_ids()
        out_before = model(ids).detach().clone()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "dual" in name:
                    p.normal_()
        out_after = model(ids).detach()
        assert not torch.allclose(out_before, out_after), (
            "Output unchanged after W_dual perturbation — fix is not working"
        )

    def test_linear_b_exists(self):
        model = self.make_fix()
        assert hasattr(model, "linear_b")
        assert isinstance(model.linear_b, torch.nn.Linear)

    def test_linear_b_output_shape(self):
        model = self.make_fix()
        assert model.linear_b.in_features == D
        assert model.linear_b.out_features == 2

    @pytest.mark.parametrize("n_classes", [2, 3, 10])
    def test_various_n_classes(self, n_classes):
        model = DualOutputClassifier(VOCAB, D, H, n_classes=n_classes)
        out = model(self.make_ids())
        assert out.shape == (2, n_classes)

    def test_no_nan_in_output(self):
        model = self.make_fix()
        out = model(self.make_ids())
        assert not torch.any(torch.isnan(out))

    def test_standard_model_w_dual_grad_still_none(self):
        """Sanity check: the original model's W_dual is still dead."""
        model = make_model()
        ids = self.make_ids()
        model(ids).sum().backward()
        for name, param in model.named_parameters():
            if "dual" in name:
                assert param.grad is None, f"Standard model: {name}.grad should be None"
