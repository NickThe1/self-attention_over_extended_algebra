# Tech Stack

## Runtime

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.13 | pinned in `.python-version` |
| Package manager | `uv` (via `pyproject.toml`) | fast, lockfile-based, already configured |

## Core Libraries

### PyTorch
The only deep learning dependency. Used for:
- `torch.Tensor` as the primitive for both `.real` and `.dual` components of every DualTensor
- Autograd tracks gradients through all dual-algebra ops (matmul, softmax) without needing custom backward passes — we write everything in terms of standard PyTorch ops
- `torch.nn.Module` for `DualLinear`, `DualAttention`, `DualEmbedding`, classifier head
- `torch.optim.AdamW` for training

Install: `uv add torch`

### NumPy
Data generation only (synthetic sequence dataset). No deep learning ops.

Install: `uv add numpy`

## Optional / Reporting

| Library | Use |
|---|---|
| `matplotlib` | loss curves, gradient norm plots for the report |
| `tabulate` or `rich` | pretty-print results tables in the terminal |

Install: `uv add matplotlib rich`

## Why No Hugging Face / Lightning / etc.

The task is deliberately low-level. Implementing dual-number matmul and softmax from scratch is the assignment — wrapping it in a high-level framework would obscure exactly the thing being evaluated.

## Key Design Constraint

All dual-algebra operations must be expressed as compositions of standard PyTorch differentiable ops so that:
1. PyTorch autograd correctly computes gradients with respect to both `W_real` and `W_dual`
2. We can log `param.grad.norm()` for both real and dual weights as a diagnostic signal

No custom `torch.autograd.Function` is needed — the algebra rules are linear maps expressible directly in terms of `@`, `*`, `+`.

## Environment Setup

```bash
uv sync
uv run main.py
```

No GPU required — the model is tiny (d_model=32, single attention layer, vocab=32, seq_len=16). CPU training completes in under a minute.
