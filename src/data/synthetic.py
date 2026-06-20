import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


VOCAB_SIZE = 32
SEQ_LEN = 16


class FirstLastMatchDataset(Dataset):
    """Binary classification: label=1 iff tokens[0] == tokens[-1].

    Balanced 50/50 by construction: half positives, half negatives,
    generated with a fixed seed for reproducibility.
    """

    def __init__(self, n_samples: int, seed: int = 0):
        assert n_samples % 2 == 0, "n_samples must be even for balanced split"
        rng = torch.Generator()
        rng.manual_seed(seed)

        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        tokens = torch.randint(0, VOCAB_SIZE, (n_samples, SEQ_LEN), generator=rng)

        # Positives: force tokens[:, -1] == tokens[:, 0]
        tokens[:n_pos, -1] = tokens[:n_pos, 0]

        # Negatives: force tokens[:, -1] != tokens[:, 0] via rejection
        for i in range(n_pos, n_samples):
            while tokens[i, -1] == tokens[i, 0]:
                tokens[i, -1] = torch.randint(0, VOCAB_SIZE, (1,), generator=rng)

        labels = torch.zeros(n_samples, dtype=torch.long)
        labels[:n_pos] = 1

        # Shuffle so positives aren't all at the front
        perm = torch.randperm(n_samples, generator=rng)
        self.tokens: Tensor = tokens[perm]
        self.labels: Tensor = labels[perm]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.tokens[idx], self.labels[idx]


def make_dataloaders(
    n_train: int = 4000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_ds = FirstLastMatchDataset(n_train, seed=seed)
    test_ds = FirstLastMatchDataset(n_test, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def majority_baseline(dataset: FirstLastMatchDataset) -> float:
    """Accuracy of always predicting the majority class."""
    labels = dataset.labels
    majority_count = int(labels.sum().item())
    majority_count = max(majority_count, len(labels) - majority_count)
    return majority_count / len(labels)
