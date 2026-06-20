import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algebra import DualTensor
from model import DualTransformerClassifier
from training.loop import evaluate, run_training


# ── Experiment A ──────────────────────────────────────────────────────────────

def experiment_a_gradient_norms(
    model: DualTransformerClassifier,
    train_loader: DataLoader,
    device: torch.device,
) -> dict[str, list[float]]:
    """Track ‖∂L/∂param‖ per batch for one epoch.

    Returns {param_name: [norm_batch_0, norm_batch_1, ...]} for every
    named parameter. Dual params will always have norm 0.0 because they
    are absent from the computation graph (their .grad stays None).
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    norms: dict[str, list[float]] = {name: [] for name, _ in model.named_parameters()}

    for tokens, labels in train_loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(tokens), labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                norms[name].append(0.0)
            else:
                norms[name].append(param.grad.norm().item())

        optimizer.step()

    return norms


# ── Experiment B ──────────────────────────────────────────────────────────────

def experiment_b_ablation(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    vocab_size: int = 32,
    d_model: int = 16,
    n_heads: int = 2,
    n_epochs: int = 20,
) -> dict[str, float]:
    """Ablation: full-dual model vs real-only (W_dual frozen to 0).

    Both models are trained from the same random seed so the only
    difference is whether W_dual parameters participate in optimisation.
    Because W_dual never enters the computation graph of the real output,
    both runs should converge to the same accuracy.

    Returns {'dual': test_acc, 'real_only': test_acc}.
    """
    def _train(freeze_dual: bool) -> float:
        torch.manual_seed(0)
        model = DualTransformerClassifier(vocab_size, d_model, n_heads).to(device)
        if freeze_dual:
            for name, param in model.named_parameters():
                if "dual" in name:
                    param.requires_grad_(False)
        history = run_training(
            model, train_loader, test_loader,
            n_epochs=n_epochs, device=device,
        )
        return history[-1]["test_acc"]

    return {
        "dual":      _train(freeze_dual=False),
        "real_only": _train(freeze_dual=True),
    }


# ── Experiment C ──────────────────────────────────────────────────────────────

def experiment_c_perturbation(
    model: DualTransformerClassifier,
    test_loader: DataLoader,
    device: torch.device,
    sigmas: list[float] | None = None,
) -> dict[float, float]:
    """Perturb every W_dual by N(0, σ²) and measure test accuracy.

    The model output depends only on real weights, so perturbing W_dual
    by any σ must leave accuracy unchanged.

    Returns {sigma: accuracy}.
    """
    if sigmas is None:
        sigmas = [0.1, 1.0, 10.0]

    model.eval()
    model.to(device)

    dual_params = [(name, p) for name, p in model.named_parameters() if "dual" in name]

    results: dict[float, float] = {}
    for sigma in sigmas:
        originals = {name: p.data.clone() for name, p in dual_params}

        with torch.no_grad():
            for _, p in dual_params:
                p.add_(torch.randn_like(p) * sigma)

        results[sigma] = evaluate(model, test_loader, device)["acc"]

        with torch.no_grad():
            for name, p in dual_params:
                p.data.copy_(originals[name])

    return results


# ── 7.4  Minimal fix ─────────────────────────────────────────────────────────

class DualOutputClassifier(nn.Module):
    """Minimal fix: use both real and dual pooled vectors for classification.

    Standard model:  logits = linear(pool(out.real))
    Fixed model:     logits = linear_a(pool(out.real)) + linear_b(pool(out.dual))

    The second term routes a gradient back through out.dual into W_dual,
    breaking the dead-weight property. This is still an algebraically valid
    model — it is no longer a pure dual-number algebra computation, but it
    uses the dual components as a secondary feature channel.

    Note: this is NOT the same as replacing the classifier with a dual linear
    layer, because that would still zero out W_dual via α²=0 during the score
    and weighted-sum steps. Instead, we must branch at the pooling stage.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_classes: int = 2):
        super().__init__()
        self.backbone = DualTransformerClassifier(vocab_size, d_model, n_heads, n_classes)
        self.linear_b = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, L = token_ids.shape
        x = self.backbone.embedding(token_ids)
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = DualTensor(x.real + self.backbone.pos_emb(positions), x.dual)
        x = self.backbone.attention(x)
        pooled_real = x.real.mean(dim=1)
        pooled_dual = x.dual.mean(dim=1)
        return self.backbone.classifier(pooled_real) + self.linear_b(pooled_dual)
