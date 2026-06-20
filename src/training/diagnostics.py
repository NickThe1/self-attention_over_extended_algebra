import torch
from torch.utils.data import DataLoader

from model import DualTransformerClassifier


def experiment_a_gradient_norms(
    model: DualTransformerClassifier,
    train_loader: DataLoader,
    device: torch.device,
) -> dict[str, list[float]]:
    """
    Experiment A: track ||∂L/∂W_dual|| vs ||∂L/∂W_real|| over one training epoch.
    Returns {param_name: [norm_per_batch]} for real and dual params separately.
    """
    raise NotImplementedError


def experiment_b_ablation(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    vocab_size: int = 32,
    d_model: int = 32,
    n_heads: int = 2,
    n_epochs: int = 20,
) -> dict[str, float]:
    """
    Experiment B: train full dual model vs b=0 frozen model, return test accuracies.
    Returns {'dual': acc, 'real_only': acc}.
    """
    raise NotImplementedError


def experiment_c_perturbation(
    model: DualTransformerClassifier,
    test_loader: DataLoader,
    device: torch.device,
    sigmas: list[float] | None = None,
) -> dict[float, float]:
    """
    Experiment C: perturb W_dual by N(0, sigma) and measure accuracy drop.
    Returns {sigma: accuracy}.
    """
    raise NotImplementedError
