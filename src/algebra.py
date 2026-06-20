from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class DualTensor:
    real: Tensor
    dual: Tensor

    def __post_init__(self):
        assert self.real.shape == self.dual.shape, (
            f"real and dual must have the same shape, got {self.real.shape} vs {self.dual.shape}"
        )

    @property
    def shape(self):
        return self.real.shape


def dual_add(x: DualTensor, y: DualTensor) -> DualTensor:
    raise NotImplementedError


def dual_mul(x: DualTensor, y: DualTensor) -> DualTensor:
    """Element-wise multiplication: (a+αb)(c+αd) = ac + α(ad+bc)."""
    raise NotImplementedError


def dual_matmul(x: DualTensor, w: DualTensor) -> DualTensor:
    """Matrix multiplication over dual numbers."""
    raise NotImplementedError


def dual_softmax(x: DualTensor, dim: int = -1) -> DualTensor:
    """Softmax lifted to dual numbers via f(a+αb) = f(a) + α f'(a) b."""
    raise NotImplementedError


def dual_transpose(x: DualTensor, dim0: int, dim1: int) -> DualTensor:
    raise NotImplementedError
