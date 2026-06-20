# Self-Attention over the Dual-Number Algebra

Implementation and analysis of scaled dot-product multi-head self-attention operating entirely within the dual-number algebra (where α² = 0). Trained on a synthetic first-last-token matching task. Research finding: the dual (b-) components of all weight matrices are algebraically dead — they never enter the real output and receive zero gradient throughout training.

---

## Report

The full write-up — algebra, formulas, results tables, conclusions, citations — is in **[report.md](report.md)**.

---

## Quick start

Requires [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Run all tests (227 total)
uv run pytest

# Run tests for a specific phase
uv run pytest tests/test_algebra.py       # Phase 1 — dual primitives
uv run pytest tests/test_model.py         # Phase 2 — DualLinear
uv run pytest tests/test_attention.py     # Phase 3 — DualAttention
uv run pytest tests/test_full_model.py    # Phase 4 — full model
uv run pytest tests/test_dataset.py       # Phase 5 — synthetic dataset
uv run pytest tests/test_training.py      # Phase 6 — training loop
uv run pytest tests/test_diagnostics.py   # Phase 7 — diagnostics
```

---

## Train the model

```python
import sys
sys.path.insert(0, "src")

import torch
from model import DualTransformerClassifier
from data.synthetic import make_dataloaders
from training.loop import run_training

train_loader, test_loader = make_dataloaders(4000, 1000, batch_size=64)
model = DualTransformerClassifier(vocab_size=32, d_model=16, n_heads=2)
history = run_training(model, train_loader, test_loader, n_epochs=20)

print(f"Test accuracy: {history[-1]['test_acc']:.1%}")  # ~98.7%
```

---

## Run the research diagnostics

```python
import sys
sys.path.insert(0, "src")

import torch
from model import DualTransformerClassifier
from data.synthetic import make_dataloaders
from training.loop import run_training
from training.diagnostics import (
    experiment_a_gradient_norms,
    experiment_b_ablation,
    experiment_c_perturbation,
)

train_loader, test_loader = make_dataloaders(4000, 1000, batch_size=64)

# Exp A: dual grad norms are 0.0 every batch
model = DualTransformerClassifier(vocab_size=32, d_model=16, n_heads=2)
norms = experiment_a_gradient_norms(model, train_loader, torch.device("cpu"))
print(norms["attention.W_Q.W_dual"])  # [0.0, 0.0, 0.0, ...]

# Exp B: freezing W_dual changes nothing
result = experiment_b_ablation(train_loader, test_loader, torch.device("cpu"))
print(result)  # {'dual': 0.987, 'real_only': 0.987}

# Exp C: perturbing W_dual by any sigma changes nothing
model = DualTransformerClassifier(vocab_size=32, d_model=16, n_heads=2)
run_training(model, train_loader, test_loader, n_epochs=20)
result = experiment_c_perturbation(model, test_loader, torch.device("cpu"))
print(result)  # {0.1: 0.987, 1.0: 0.987, 10.0: 0.987}
```

---

## Project structure

```
├── report.md                    # Full written report
├── plan.md                      # Implementation plan
├── main.py                      # Entry point (smoke test)
├── src/
│   ├── algebra.py               # DualTensor, dual_add, dual_mul, dual_matmul, dual_softmax
│   ├── model.py                 # DualLinear, DualEmbedding, DualAttention, DualTransformerClassifier
│   ├── data/
│   │   └── synthetic.py         # FirstLastMatchDataset, make_dataloaders, majority_baseline
│   └── training/
│       ├── loop.py              # train_epoch, evaluate, log_gradient_norms, run_training
│       └── diagnostics.py       # Exp A/B/C, DualOutputClassifier (minimal fix)
├── tests/
│   ├── test_algebra.py          # 46 tests
│   ├── test_model.py            # 27 tests
│   ├── test_attention.py        # 26 tests
│   ├── test_full_model.py       # 34 tests
│   ├── test_dataset.py          # 34 tests
│   ├── test_training.py         # 29 tests
│   └── test_diagnostics.py      # 31 tests
└── docs/
    └── formulas.md              # Mathematical formulas for every operation
```

---

## Key finding

The dual weight matrices ($W_{Q_d}, W_{K_d}, W_{V_d}, W_{O_d}, E_d$) never appear in the real output of the network:

$$\text{out}_r = \text{softmax}\!\left(\frac{X_r W_{Q_r} W_{K_r}^\top X_r^\top}{\sqrt{d_h}}\right) X_r W_{V_r} W_{O_r}$$

This follows directly from $\alpha^2 = 0$: the dual-dual cross-terms vanish at every matrix multiplication, and softmax's real output depends only on its real input. As a result, all five dual parameter tensors have `.grad = None` after every backward pass — PyTorch never allocates a gradient for them at all.
