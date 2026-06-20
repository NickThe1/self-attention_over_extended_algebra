import torch
import torch.nn as nn
from torch import Tensor

from algebra import DualTensor, dual_matmul, dual_softmax, dual_transpose


class DualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: DualTensor) -> DualTensor:
        raise NotImplementedError


class DualEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids: Tensor) -> DualTensor:
        raise NotImplementedError


class DualAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: DualTensor) -> DualTensor:
        raise NotImplementedError


class DualTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_classes: int = 2):
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids: Tensor) -> Tensor:
        """Returns logits shape [B, n_classes] computed from the real part of the output."""
        raise NotImplementedError
