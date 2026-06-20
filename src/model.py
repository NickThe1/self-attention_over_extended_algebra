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
