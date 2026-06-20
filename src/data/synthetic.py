import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


VOCAB_SIZE = 32
SEQ_LEN = 16


class FirstLastMatchDataset(Dataset):
    """Binary classification: label=1 iff tokens[0] == tokens[-1]."""

    def __init__(self, n_samples: int, seed: int = 0):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def make_dataloaders(
    n_train: int = 4000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    raise NotImplementedError


def majority_baseline(dataset: FirstLastMatchDataset) -> float:
    """Accuracy of always predicting the majority class."""
    raise NotImplementedError
