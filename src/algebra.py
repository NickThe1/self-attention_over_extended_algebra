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
    return DualTensor(x.real + y.real, x.dual + y.dual)


def dual_mul(x: DualTensor, y: DualTensor) -> DualTensor:
    """Element-wise: (a+αb)(c+αd) = ac + α(ad+bc)."""
    return DualTensor(
        x.real * y.real,
        x.real * y.dual + x.dual * y.real,
    )


def dual_matmul(x: DualTensor, w: DualTensor) -> DualTensor:
    """Matrix multiply: real = x.r @ w.r, dual = x.r @ w.d + x.d @ w.r."""
    return DualTensor(
        x.real @ w.real,
        x.real @ w.dual + x.dual @ w.real,
    )


def dual_softmax(x: DualTensor, dim: int = -1) -> DualTensor:
    """Softmax lifted via f(a+αb) = f(a) + α f'(a)b.

    Jacobian of softmax: J @ b = s ⊙ (b - (s·b)), where s = softmax(a).
    """
    s = torch.softmax(x.real, dim=dim)
    ds = s * (x.dual - (s * x.dual).sum(dim=dim, keepdim=True))
    return DualTensor(s, ds)


def dual_transpose(x: DualTensor, dim0: int, dim1: int) -> DualTensor:
    return DualTensor(x.real.transpose(dim0, dim1), x.dual.transpose(dim0, dim1))
