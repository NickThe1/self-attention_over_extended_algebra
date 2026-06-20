import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DualTransformerClassifier


def train_epoch(
    model: DualTransformerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Returns dict with 'loss' and 'acc'."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += len(labels)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: DualTransformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Returns dict with 'loss' and 'acc'."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(tokens)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += len(labels)

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def log_gradient_norms(model: DualTransformerClassifier) -> dict[str, float]:
    """Returns {param_name: grad_norm} for all named parameters.

    Parameters with no gradient (e.g. all W_dual) get norm 0.0 rather than
    being omitted, so the caller always sees a fixed set of keys.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            norms[name] = 0.0
        else:
            norms[name] = param.grad.norm().item()
    return norms


def run_training(
    model: DualTransformerClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> list[dict]:
    """Trains the model and returns a list of per-epoch metric dicts.

    Each dict contains:
      epoch, train_loss, train_acc, test_loss, test_acc,
      grad_norms (dict of param_name -> norm).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(1, n_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        grad_norms = log_gradient_norms(model)
        test_metrics = evaluate(model, test_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "grad_norms": grad_norms,
        })

    return history
