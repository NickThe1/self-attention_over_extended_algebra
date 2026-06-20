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
    raise NotImplementedError


@torch.no_grad()
def evaluate(
    model: DualTransformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Returns dict with 'loss' and 'acc'."""
    raise NotImplementedError


def log_gradient_norms(model: DualTransformerClassifier) -> dict[str, float]:
    """Returns {param_name: grad_norm} for all named parameters."""
    raise NotImplementedError


def run_training(
    model: DualTransformerClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> list[dict]:
    """Trains the model and returns a list of per-epoch metric dicts."""
    raise NotImplementedError
