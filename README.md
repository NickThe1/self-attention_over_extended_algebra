# Self-Attention over the Dual-Number Algebra

Scaled dot-product multi-head self-attention implemented entirely in the dual-number algebra (α² = 0). Key finding: the dual (b-) weight components are algebraically dead — they never reach the real output and receive zero gradient throughout training.

Full write-up: **[report.md](report.md)**

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync          # install dependencies
uv run pytest    # run all 227 tests
uv run python main.py  # train + run all diagnostics
```

## Structure

```
src/
  algebra.py        # DualTensor, dual_matmul, dual_softmax
  model.py          # DualLinear, DualAttention, DualTransformerClassifier, RealTransformerClassifier
  data/synthetic.py # FirstLastMatchDataset, make_dataloaders
  training/
    loop.py         # train_epoch, evaluate, run_training
    diagnostics.py  # Exp A/B/C, experiment_10_comparison, DualOutputClassifier
tests/              # 227 tests across 7 modules
report.md           # formulas, results, conclusions, citations
```
