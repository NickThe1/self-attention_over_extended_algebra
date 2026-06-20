import math

import torch
import torch.nn as nn
from torch import Tensor

from algebra import DualTensor, dual_matmul, dual_softmax, dual_transpose


class DualLinear(nn.Module):
    """Linear projection over the dual-number algebra.

    Weights are dual tensors (W_real, W_dual), both shape [in, out].
    W_dual is initialised to zero so the model starts as a standard real network.
    Bias, if used, is a real number added only to the real part of the output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_real = nn.Parameter(torch.empty(in_features, out_features))
        self.W_dual = nn.Parameter(torch.zeros(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_real, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: DualTensor) -> DualTensor:
        out = dual_matmul(x, DualTensor(self.W_real, self.W_dual))
        if self.bias is not None:
            out = DualTensor(out.real + self.bias, out.dual)
        return out


class DualEmbedding(nn.Module):
    """Token embedding over the dual-number algebra.

    Two independent embedding tables (emb_real, emb_dual) of the same shape.
    emb_dual is initialised to zero — consistent with the other dual parameters.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb_real = nn.Embedding(vocab_size, d_model)
        self.emb_dual = nn.Embedding(vocab_size, d_model)
        nn.init.zeros_(self.emb_dual.weight)

    def forward(self, token_ids: Tensor) -> DualTensor:
        return DualTensor(self.emb_real(token_ids), self.emb_dual(token_ids))


class DualAttention(nn.Module):
    """Scaled dot-product multi-head self-attention over the dual-number algebra.

    Every operation — projection, score computation, softmax, weighted sum,
    output projection — follows dual-number arithmetic. The real output is
    algebraically independent of all W_dual parameters.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.W_Q = DualLinear(d_model, d_model, bias=False)
        self.W_K = DualLinear(d_model, d_model, bias=False)
        self.W_V = DualLinear(d_model, d_model, bias=False)
        self.W_O = DualLinear(d_model, d_model, bias=False)

    def _split_heads(self, x: DualTensor, B: int, L: int) -> DualTensor:
        """[B, L, d_model] → [B, n_heads, L, d_head]."""
        return DualTensor(
            x.real.view(B, L, self.n_heads, self.d_head).transpose(1, 2),
            x.dual.view(B, L, self.n_heads, self.d_head).transpose(1, 2),
        )

    def _merge_heads(self, x: DualTensor, B: int, L: int) -> DualTensor:
        """[B, n_heads, L, d_head] → [B, L, d_model]."""
        return DualTensor(
            x.real.transpose(1, 2).contiguous().view(B, L, self.d_model),
            x.dual.transpose(1, 2).contiguous().view(B, L, self.d_model),
        )

    def forward(self, x: DualTensor) -> DualTensor:
        B, L, _ = x.real.shape

        # Step 1: project to Q, K, V
        Q = self._split_heads(self.W_Q(x), B, L)  # [B, H, L, d_head]
        K = self._split_heads(self.W_K(x), B, L)
        V = self._split_heads(self.W_V(x), B, L)

        # Step 2: scaled dot-product scores [B, H, L, L]
        scores = dual_matmul(Q, dual_transpose(K, -2, -1))
        scores = DualTensor(scores.real * self.scale, scores.dual * self.scale)

        # Step 3: softmax over key dimension
        attn = dual_softmax(scores, dim=-1)

        # Step 4: weighted sum [B, H, L, d_head]
        context = dual_matmul(attn, V)

        # Step 5: merge heads and output projection
        return self.W_O(self._merge_heads(context, B, L))


class DualTransformerClassifier(nn.Module):
    """Full model: DualEmbedding + pos_emb → DualAttention → mean-pool (real) → Linear classifier.

    Positional embeddings are real-valued and added only to the real part of the
    dual tensor — consistent with the algebra (position is not a dual quantity).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_classes: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = DualEmbedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attention = DualAttention(d_model, n_heads)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        _, L = token_ids.shape
        x = self.embedding(token_ids)                                   # DualTensor [B, L, d]
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = DualTensor(x.real + self.pos_emb(positions), x.dual)       # positional signal on real only
        x = self.attention(x)                                           # DualTensor [B, L, d]
        pooled = x.real.mean(dim=1)                                     # [B, d]  — real part only
        return self.classifier(pooled)                                  # [B, n_classes]
