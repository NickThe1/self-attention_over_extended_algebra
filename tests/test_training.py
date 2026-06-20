import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import DualTransformerClassifier
from data.synthetic import FirstLastMatchDataset, make_dataloaders, majority_baseline
from training.loop import train_epoch, evaluate, log_gradient_norms, run_training


VOCAB = 32
SEQ = 16
D = 16
H = 2
DEVICE = torch.device("cpu")


def make_model(seed=0):
    torch.manual_seed(seed)
    return DualTransformerClassifier(vocab_size=VOCAB, d_model=D, n_heads=H)


def tiny_loader(n=64, batch_size=16, seed=0):
    """Small balanced dataset for fast unit tests."""
    ds = FirstLastMatchDataset(n, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ── 6.1  train_epoch ─────────────────────────────────────────────────────────

class TestTrainEpoch:
    def test_returns_dict_with_loss_and_acc(self):
        model = make_model()
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_epoch(model, loader, opt, DEVICE)
        assert set(result.keys()) == {"loss", "acc"}

    def test_loss_is_positive(self):
        model = make_model()
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_epoch(model, loader, opt, DEVICE)
        assert result["loss"] > 0

    def test_acc_in_unit_interval(self):
        model = make_model()
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_epoch(model, loader, opt, DEVICE)
        assert 0.0 <= result["acc"] <= 1.0

    def test_weights_change_after_epoch(self):
        model = make_model()
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        w_before = model.embedding.emb_real.weight.detach().clone()
        train_epoch(model, loader, opt, DEVICE)
        w_after = model.embedding.emb_real.weight.detach()
        assert not torch.equal(w_before, w_after)

    def test_model_in_train_mode_during_epoch(self):
        """train_epoch must set model.training=True before iterating."""
        model = make_model()
        model.eval()
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        seen_modes = []
        original_forward = model.forward

        def patched_forward(x):
            seen_modes.append(model.training)
            return original_forward(x)

        model.forward = patched_forward
        train_epoch(model, loader, opt, DEVICE)
        assert all(seen_modes), "model was not in train mode during forward passes"

    def test_loss_decreases_over_multiple_epochs(self):
        """Loss should trend downward with enough data and steps."""
        torch.manual_seed(42)
        model = make_model(seed=42)
        loader = tiny_loader(n=128, batch_size=32)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        losses = [train_epoch(model, loader, opt, DEVICE)["loss"] for _ in range(10)]
        # First loss > last loss (not strict per-step, just overall trend)
        assert losses[0] > losses[-1]


# ── 6.2  evaluate ────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_dict_with_loss_and_acc(self):
        model = make_model()
        result = evaluate(model, tiny_loader(), DEVICE)
        assert set(result.keys()) == {"loss", "acc"}

    def test_loss_positive(self):
        result = evaluate(make_model(), tiny_loader(), DEVICE)
        assert result["loss"] > 0

    def test_acc_in_unit_interval(self):
        result = evaluate(make_model(), tiny_loader(), DEVICE)
        assert 0.0 <= result["acc"] <= 1.0

    def test_no_grad_used(self):
        """evaluate must not update weights."""
        model = make_model()
        loader = tiny_loader()
        w_before = model.embedding.emb_real.weight.detach().clone()
        evaluate(model, loader, DEVICE)
        w_after = model.embedding.emb_real.weight.detach()
        assert torch.equal(w_before, w_after)

    def test_model_in_eval_mode(self):
        """evaluate must set model.training=False before iterating."""
        model = make_model()
        model.train()
        loader = tiny_loader()
        seen_modes = []
        original_forward = model.forward

        def patched_forward(x):
            seen_modes.append(model.training)
            return original_forward(x)

        model.forward = patched_forward
        evaluate(model, loader, DEVICE)
        assert not any(seen_modes), "model was in train mode during evaluate"

    def test_perfect_predictor_gives_acc_1(self):
        """A model that always outputs correct class should score 1.0."""
        class AlwaysClass0(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                out = torch.zeros(B, 2)
                out[:, 0] = 100.0
                return out

        ds = FirstLastMatchDataset(64, seed=0)
        all_zero_labels = torch.zeros(64, dtype=torch.long)
        ds.labels = all_zero_labels
        loader = DataLoader(ds, batch_size=16)

        model = AlwaysClass0()
        result = evaluate(model, loader, DEVICE)
        assert result["acc"] == pytest.approx(1.0)

    def test_deterministic_across_calls(self):
        """Same model, same loader → same metrics every time."""
        model = make_model()
        loader = tiny_loader()
        r1 = evaluate(model, loader, DEVICE)
        r2 = evaluate(model, loader, DEVICE)
        assert r1["loss"] == pytest.approx(r2["loss"])
        assert r1["acc"] == pytest.approx(r2["acc"])


# ── 6.3  log_gradient_norms ──────────────────────────────────────────────────

class TestLogGradientNorms:
    def _norms_after_backward(self, model):
        loader = tiny_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_epoch(model, loader, opt, DEVICE)
        return log_gradient_norms(model)

    def test_returns_dict(self):
        model = make_model()
        norms = self._norms_after_backward(model)
        assert isinstance(norms, dict)

    def test_keys_match_named_parameters(self):
        model = make_model()
        norms = self._norms_after_backward(model)
        expected = {name for name, _ in model.named_parameters()}
        assert set(norms.keys()) == expected

    def test_all_values_are_floats(self):
        model = make_model()
        norms = self._norms_after_backward(model)
        assert all(isinstance(v, float) for v in norms.values())

    def test_all_values_non_negative(self):
        model = make_model()
        norms = self._norms_after_backward(model)
        assert all(v >= 0.0 for v in norms.values())

    def test_dual_param_norms_are_zero(self):
        """W_dual parameters are not in the computation graph → grad is None → norm 0.0."""
        model = make_model()
        norms = self._norms_after_backward(model)
        dual_keys = [
            "embedding.emb_dual.weight",
            "attention.W_Q.W_dual",
            "attention.W_K.W_dual",
            "attention.W_V.W_dual",
            "attention.W_O.W_dual",
        ]
        for key in dual_keys:
            assert norms[key] == 0.0, f"{key} norm should be 0.0, got {norms[key]}"

    def test_real_param_norms_are_positive(self):
        """Real parameters receive meaningful gradients."""
        model = make_model()
        norms = self._norms_after_backward(model)
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
            assert norms[key] > 0.0, f"{key} norm should be > 0, got {norms[key]}"

    def test_no_grad_model_all_zero(self):
        """Before any backward pass all norms should be 0.0 (grads are None)."""
        model = make_model()
        norms = log_gradient_norms(model)
        assert all(v == 0.0 for v in norms.values())


# ── 6.4  run_training ────────────────────────────────────────────────────────

class TestRunTraining:
    def _run(self, n_train=128, n_test=64, n_epochs=2, seed=0):
        torch.manual_seed(seed)
        train_loader, test_loader = make_dataloaders(n_train, n_test, batch_size=32, seed=seed)
        model = make_model(seed=seed)
        return run_training(model, train_loader, test_loader, n_epochs=n_epochs)

    def test_returns_list(self):
        history = self._run()
        assert isinstance(history, list)

    def test_history_length_equals_n_epochs(self):
        history = self._run(n_epochs=3)
        assert len(history) == 3

    def test_epoch_dict_keys(self):
        entry = self._run()[0]
        assert {"epoch", "train_loss", "train_acc", "test_loss", "test_acc", "grad_norms"} <= entry.keys()

    def test_epoch_numbers_sequential(self):
        history = self._run(n_epochs=4)
        assert [e["epoch"] for e in history] == [1, 2, 3, 4]

    def test_grad_norms_in_every_entry(self):
        history = self._run(n_epochs=2)
        for entry in history:
            assert isinstance(entry["grad_norms"], dict)
            assert len(entry["grad_norms"]) > 0

    def test_accuracies_in_unit_interval(self):
        history = self._run(n_epochs=2)
        for entry in history:
            assert 0.0 <= entry["train_acc"] <= 1.0
            assert 0.0 <= entry["test_acc"] <= 1.0

    def test_losses_positive(self):
        history = self._run(n_epochs=2)
        for entry in history:
            assert entry["train_loss"] > 0
            assert entry["test_loss"] > 0

    def test_dual_grad_norms_zero_every_epoch(self):
        history = self._run(n_epochs=3)
        dual_keys = [
            "embedding.emb_dual.weight",
            "attention.W_Q.W_dual",
            "attention.W_K.W_dual",
            "attention.W_V.W_dual",
            "attention.W_O.W_dual",
        ]
        for entry in history:
            for key in dual_keys:
                assert entry["grad_norms"][key] == 0.0, (
                    f"epoch {entry['epoch']}: {key} grad norm should be 0.0"
                )


# ── 6.5  End-to-end accuracy (integration) ───────────────────────────────────

class TestEndToEndAccuracy:
    def test_achieves_85_percent_test_accuracy(self):
        """Full training run on the prescribed dataset must reach >85% test accuracy."""
        torch.manual_seed(0)
        train_loader, test_loader = make_dataloaders(4000, 1000, batch_size=64)
        model = DualTransformerClassifier(vocab_size=VOCAB, d_model=D, n_heads=H)
        history = run_training(model, train_loader, test_loader, n_epochs=20, lr=1e-3, weight_decay=1e-4)
        final_test_acc = history[-1]["test_acc"]
        baseline = majority_baseline(test_loader.dataset)
        assert baseline == pytest.approx(0.5), f"majority baseline should be 0.5, got {baseline}"
        assert final_test_acc > 0.85, (
            f"Expected >85% test accuracy, got {final_test_acc:.1%} "
            f"(majority baseline: {baseline:.1%})"
        )
