import pytest
import torch
from torch.utils.data import DataLoader

from data.synthetic import (
    VOCAB_SIZE,
    SEQ_LEN,
    FirstLastMatchDataset,
    make_dataloaders,
    majority_baseline,
)


def make_ds(n=100, seed=0):
    return FirstLastMatchDataset(n, seed=seed)


# ── 5.1  Dataset structure ────────────────────────────────────────────────────

class TestDatasetStructure:
    def test_len(self):
        assert len(make_ds(200)) == 200

    def test_getitem_returns_tuple(self):
        tokens, label = make_ds()[0]
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_token_shape(self):
        tokens, _ = make_ds()[0]
        assert tokens.shape == (SEQ_LEN,)

    def test_label_shape(self):
        _, label = make_ds()[0]
        assert label.shape == ()

    def test_token_dtype(self):
        tokens, _ = make_ds()[0]
        assert tokens.dtype == torch.long

    def test_label_dtype(self):
        _, label = make_ds()[0]
        assert label.dtype == torch.long

    def test_tokens_in_vocab(self):
        ds = make_ds(200)
        tokens = ds.tokens
        assert tokens.min() >= 0
        assert tokens.max() < VOCAB_SIZE

    def test_labels_binary(self):
        ds = make_ds(200)
        unique = ds.labels.unique().tolist()
        assert set(unique) <= {0, 1}

    def test_seq_len_constant(self):
        assert SEQ_LEN == 16

    def test_vocab_size_constant(self):
        assert VOCAB_SIZE == 32


# ── 5.2  Label correctness ────────────────────────────────────────────────────

class TestLabelCorrectness:
    def test_positive_samples_first_eq_last(self):
        ds = make_ds(400)
        pos_mask = ds.labels == 1
        pos_tokens = ds.tokens[pos_mask]
        assert (pos_tokens[:, 0] == pos_tokens[:, -1]).all()

    def test_negative_samples_first_neq_last(self):
        ds = make_ds(400)
        neg_mask = ds.labels == 0
        neg_tokens = ds.tokens[neg_mask]
        assert (neg_tokens[:, 0] != neg_tokens[:, -1]).all()

    def test_all_labels_consistent(self):
        """Label matches the first==last condition for every sample."""
        ds = make_ds(500)
        for i in range(len(ds)):
            tokens, label = ds[i]
            expected = int(tokens[0] == tokens[-1])
            assert label.item() == expected, f"Sample {i}: tokens[0]={tokens[0]}, tokens[-1]={tokens[-1]}, label={label}"


# ── 5.3  Balance ──────────────────────────────────────────────────────────────

class TestBalance:
    def test_exact_half_positive(self):
        ds = make_ds(200)
        n_pos = int((ds.labels == 1).sum())
        assert n_pos == 100

    def test_exact_half_negative(self):
        ds = make_ds(200)
        n_neg = int((ds.labels == 0).sum())
        assert n_neg == 100

    def test_balanced_after_shuffle(self):
        """Balance must hold after the internal shuffle."""
        ds = make_ds(1000, seed=7)
        n_pos = int((ds.labels == 1).sum())
        assert n_pos == 500

    @pytest.mark.parametrize("n", [50, 100, 500, 1000])
    def test_balance_for_various_sizes(self, n):
        ds = FirstLastMatchDataset(n)
        n_pos = int((ds.labels == 1).sum())
        assert n_pos == n // 2

    def test_odd_n_raises(self):
        with pytest.raises(AssertionError):
            FirstLastMatchDataset(101)


# ── 5.4  Reproducibility ──────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_data(self):
        ds1 = make_ds(100, seed=42)
        ds2 = make_ds(100, seed=42)
        assert torch.equal(ds1.tokens, ds2.tokens)
        assert torch.equal(ds1.labels, ds2.labels)

    def test_different_seed_different_data(self):
        ds1 = make_ds(100, seed=0)
        ds2 = make_ds(100, seed=1)
        assert not torch.equal(ds1.tokens, ds2.tokens)


# ── 5.5  make_dataloaders ─────────────────────────────────────────────────────

class TestMakeDataloaders:
    def test_returns_two_dataloaders(self):
        train, test = make_dataloaders(200, 100, batch_size=32)
        assert isinstance(train, DataLoader)
        assert isinstance(test, DataLoader)

    def test_train_size(self):
        train, _ = make_dataloaders(200, 100, batch_size=32)
        assert len(train.dataset) == 200

    def test_test_size(self):
        _, test = make_dataloaders(200, 100, batch_size=32)
        assert len(test.dataset) == 100

    def test_batch_shape(self):
        train, _ = make_dataloaders(200, 100, batch_size=32)
        tokens, labels = next(iter(train))
        assert tokens.shape == (32, SEQ_LEN)
        assert labels.shape == (32,)

    def test_train_test_use_different_seeds(self):
        train, test = make_dataloaders(200, 200, batch_size=64)
        tr_tokens = train.dataset.tokens
        te_tokens = test.dataset.tokens
        assert not torch.equal(tr_tokens, te_tokens)

    def test_default_args(self):
        train, test = make_dataloaders()
        assert len(train.dataset) == 4000
        assert len(test.dataset) == 1000

    def test_partial_batch_at_end(self):
        """Last batch may be smaller than batch_size — no crash."""
        train, _ = make_dataloaders(70, 30, batch_size=32)
        batches = list(train)
        sizes = [b[0].shape[0] for b in batches]
        assert sum(sizes) == 70


# ── 5.6  majority_baseline ────────────────────────────────────────────────────

class TestMajorityBaseline:
    def test_balanced_dataset_is_half(self):
        ds = make_ds(200)
        assert majority_baseline(ds) == 0.5

    def test_returns_float(self):
        ds = make_ds(100)
        assert isinstance(majority_baseline(ds), float)

    def test_baseline_at_least_half(self):
        """Majority baseline is always >= 0.5 by definition."""
        for seed in range(5):
            ds = make_ds(200, seed=seed)
            assert majority_baseline(ds) >= 0.5

    def test_various_sizes(self):
        for n in [100, 500, 1000]:
            ds = FirstLastMatchDataset(n)
            assert majority_baseline(ds) == 0.5
