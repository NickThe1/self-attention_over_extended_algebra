import pytest
import torch
import torch.nn as nn

from algebra import DualTensor
from model import DualEmbedding, DualAttention, DualTransformerClassifier

VOCAB = 32
SEQ = 16
D = 16
H = 2


def make_ids(B=2, L=SEQ, vocab=VOCAB, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (B, L))


def make_model(vocab=VOCAB, d=D, h=H):
    torch.manual_seed(0)
    return DualTransformerClassifier(vocab_size=vocab, d_model=d, n_heads=h)


def all_dual_params(model: DualTransformerClassifier) -> dict[str, nn.Parameter]:
    return {
        "emb_dual":  model.embedding.emb_dual.weight,
        "W_Q.W_dual": model.attention.W_Q.W_dual,
        "W_K.W_dual": model.attention.W_K.W_dual,
        "W_V.W_dual": model.attention.W_V.W_dual,
        "W_O.W_dual": model.attention.W_O.W_dual,
    }


def all_real_params(model: DualTransformerClassifier) -> dict[str, nn.Parameter]:
    return {
        "emb_real":          model.embedding.emb_real.weight,
        "pos_emb":           model.pos_emb.weight,
        "W_Q.W_real":        model.attention.W_Q.W_real,
        "W_K.W_real":        model.attention.W_K.W_real,
        "W_V.W_real":        model.attention.W_V.W_real,
        "W_O.W_real":        model.attention.W_O.W_real,
        "classifier.weight": model.classifier.weight,
        "classifier.bias":   model.classifier.bias,
    }


# ── 4.1  DualEmbedding ───────────────────────────────────────────────────────

class TestDualEmbedding:
    def test_embedding_tables_exist(self):
        emb = DualEmbedding(32, 8)
        assert isinstance(emb.emb_real, nn.Embedding)
        assert isinstance(emb.emb_dual, nn.Embedding)

    def test_embedding_shapes(self):
        emb = DualEmbedding(32, 8)
        assert emb.emb_real.weight.shape == (32, 8)
        assert emb.emb_dual.weight.shape == (32, 8)

    def test_emb_dual_initialised_to_zero(self):
        assert torch.all(DualEmbedding(32, 8).emb_dual.weight == 0)

    def test_emb_real_not_zero(self):
        assert not torch.all(DualEmbedding(32, 8).emb_real.weight == 0)

    def test_output_is_dual_tensor(self):
        emb = DualEmbedding(32, 8)
        out = emb(torch.tensor([[0, 1, 2]]))
        assert isinstance(out, DualTensor)

    def test_output_shape(self):
        emb = DualEmbedding(32, 8)
        ids = make_ids(B=2, L=5, vocab=32)
        out = emb(ids)
        assert out.real.shape == (2, 5, 8)
        assert out.dual.shape == (2, 5, 8)

    def test_lookup_real_correctness(self):
        """Token i maps to row i of emb_real.weight."""
        emb = DualEmbedding(32, 8)
        ids = torch.tensor([[3, 7]])
        out = emb(ids)
        assert torch.allclose(out.real[0, 0], emb.emb_real.weight[3])
        assert torch.allclose(out.real[0, 1], emb.emb_real.weight[7])

    def test_lookup_dual_correctness(self):
        """Token i maps to row i of emb_dual.weight."""
        emb = DualEmbedding(32, 8)
        with torch.no_grad():
            emb.emb_dual.weight.normal_()
        ids = torch.tensor([[5, 10]])
        out = emb(ids)
        assert torch.allclose(out.dual[0, 0], emb.emb_dual.weight[5])
        assert torch.allclose(out.dual[0, 1], emb.emb_dual.weight[10])

    def test_different_tokens_give_different_embeddings(self):
        emb = DualEmbedding(32, 8)
        out0 = emb(torch.tensor([[0]]))
        out1 = emb(torch.tensor([[1]]))
        assert not torch.allclose(out0.real, out1.real)

    def test_emb_dual_weight_grad_none_after_backprop(self):
        """emb_dual never enters the computation graph of the real output."""
        emb = DualEmbedding(32, 8)
        attn = DualAttention(8, 2)
        ids = make_ids(B=2, L=5, vocab=32)
        x = emb(ids)
        attn(x).real.sum().backward()
        assert emb.emb_dual.weight.grad is None

    def test_emb_real_weight_receives_gradient(self):
        emb = DualEmbedding(32, 8)
        attn = DualAttention(8, 2)
        ids = make_ids(B=2, L=5, vocab=32)
        x = emb(ids)
        attn(x).real.sum().backward()
        assert emb.emb_real.weight.grad is not None

    @pytest.mark.parametrize("vocab,d", [(10, 4), (32, 16), (100, 32)])
    def test_various_vocab_and_d(self, vocab, d):
        emb = DualEmbedding(vocab, d)
        ids = torch.randint(0, vocab, (2, 6))
        out = emb(ids)
        assert out.real.shape == (2, 6, d)
        assert out.dual.shape == (2, 6, d)


# ── 4.2 / 4.3  Pooling and classifier ────────────────────────────────────────

class TestDualTransformerClassifierForward:
    def test_output_shape(self):
        model = make_model()
        out = model(make_ids())
        assert out.shape == (2, 2)

    def test_output_is_plain_tensor_not_dual(self):
        model = make_model()
        out = model(make_ids())
        assert isinstance(out, torch.Tensor)
        assert not isinstance(out, DualTensor)

    def test_batch_size_1(self):
        model = make_model()
        out = model(make_ids(B=1))
        assert out.shape == (1, 2)

    def test_seq_len_1(self):
        """Single-token sequence — pooling over L=1 is trivial."""
        model = make_model()
        out = model(make_ids(B=2, L=1))
        assert out.shape == (2, 2)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.parametrize("B,L", [(1, 1), (2, 5), (4, 16), (8, 10)])
    def test_various_seq_lengths(self, B, L):
        model = make_model()
        assert model(make_ids(B=B, L=L)).shape == (B, 2)

    @pytest.mark.parametrize("n_classes", [2, 3, 10])
    def test_various_n_classes(self, n_classes):
        model = DualTransformerClassifier(VOCAB, D, H, n_classes=n_classes)
        assert model(make_ids()).shape == (2, n_classes)

    def test_output_unaffected_by_emb_dual_perturbation(self):
        """Perturbing emb_dual must not change the logits."""
        model = make_model()
        ids = make_ids()
        out_before = model(ids).detach().clone()
        with torch.no_grad():
            model.embedding.emb_dual.weight.normal_()
        out_after = model(ids).detach()
        torch.testing.assert_close(out_before, out_after)

    def test_output_unaffected_by_all_w_dual_perturbation(self):
        """Perturbing every W_dual in the model must not change logits."""
        model = make_model()
        ids = make_ids()
        out_before = model(ids).detach().clone()
        with torch.no_grad():
            for param in all_dual_params(model).values():
                param.normal_()
        out_after = model(ids).detach()
        torch.testing.assert_close(out_before, out_after)

    def test_pooling_is_mean_over_sequence(self):
        """Verify pooling: mean of out.real along dim=1 feeds the classifier."""
        model = make_model()
        ids = make_ids()

        # Capture the pooled vector via a hook on the classifier
        captured = {}
        def hook(module, inp, out):
            captured["pooled"] = inp[0].detach()
        model.classifier.register_forward_hook(hook)
        model(ids)

        # Recompute pooled manually — must mirror forward: embed → +pos → attend → mean
        with torch.no_grad():
            L = ids.shape[1]
            positions = torch.arange(L).unsqueeze(0)
            x = model.embedding(ids)
            x = DualTensor(x.real + model.pos_emb(positions), x.dual)
            x = model.attention(x)
            expected_pooled = x.real.mean(dim=1)

        torch.testing.assert_close(captured["pooled"], expected_pooled)

    def test_no_nan_in_output(self):
        model = make_model()
        out = model(make_ids())
        assert not torch.any(torch.isnan(out))

    def test_classifier_is_standard_linear(self):
        model = make_model()
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.in_features == D
        assert model.classifier.out_features == 2


# ── 4.4  Gradient flow through full model ────────────────────────────────────

class TestDualTransformerClassifierGradients:
    def test_all_dual_params_have_none_grad(self):
        """None of the 5 dual parameters appear in the computation graph."""
        model = make_model()
        model(make_ids()).sum().backward()
        for name, param in all_dual_params(model).items():
            assert param.grad is None, f"{name}.grad should be None, got {param.grad}"

    def test_all_real_params_receive_gradients(self):
        """All real parameters, including classifier head, receive non-None gradients."""
        model = make_model()
        model(make_ids()).sum().backward()
        for name, param in all_real_params(model).items():
            assert param.grad is not None, f"{name}.grad is None"

    def test_dual_params_none_grad_even_with_nonzero_dual_init(self):
        """Setting all dual params to non-zero before training doesn't create grad paths."""
        model = make_model()
        with torch.no_grad():
            for param in all_dual_params(model).values():
                param.normal_()
        model(make_ids()).sum().backward()
        for name, param in all_dual_params(model).items():
            assert param.grad is None, f"{name}.grad should be None"

    def test_parameter_count(self):
        """Model has exactly 5 dual param tensors and 8 real param tensors."""
        model = make_model()
        assert len(all_dual_params(model)) == 5
        assert len(all_real_params(model)) == 8
