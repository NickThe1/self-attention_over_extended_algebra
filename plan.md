# Implementation Plan

## Phase 0 — Environment & Project Scaffold ✅
Goal: deps installed, `src/` structure created, smoke test passes.

- [x] 0.1 `uv add torch numpy matplotlib rich` → torch 2.12.1, numpy 2.4.6
- [x] 0.2 Create `src/algebra.py`, `src/model.py`, `src/data/synthetic.py`, `src/training/loop.py`, `src/training/diagnostics.py` — all stubs raising `NotImplementedError`
- [x] 0.3 Smoke test: `uv run python main.py` → `NotImplementedError` in `data/synthetic.py`, no `ImportError`

---

## Phase 1 — Dual-Number Primitives ✅
Goal: `DualTensor` and all algebra ops in `src/algebra.py`, covered by unit tests.

- [x] 1.1 `DualTensor(real, dual)` dataclass with shape assertion
- [x] 1.2 `dual_add`, `dual_mul` — element-wise
- [x] 1.3 `dual_matmul(x, w)` — real: `x.r @ w.r`, dual: `x.r @ w.d + x.d @ w.r`
- [x] 1.4 `dual_softmax(x)` — real: `softmax(x.r)`, dual: `s * (x.d - (s * x.d).sum(-1, keepdim=True))`
- [x] 1.5 46 tests passing: α²=0, function-lift via autograd JVP, matmul brute-force, numerical Jacobian for softmax

---

## Phase 2 — Dual Linear Layer ✅
Goal: `DualLinear` in `src/model.py` with learnable real and dual weight matrices.

- [x] 2.1 `W_real`, `W_dual` parameters (both `[in, out]`); `W_dual` init to zero; Kaiming uniform for `W_real`
- [x] 2.2 Forward: `dual_matmul(x, DualTensor(W_real, W_dual))` + bias on real part only; 27 tests passing

---

## Phase 3 — Dual Self-Attention Block ✅
Goal: `DualAttention` implementing scaled dot-product attention entirely over dual numbers.

- [x] 3.1 Four `DualLinear` projections W_Q, W_K, W_V, W_O (no bias); head split/merge helpers
- [x] 3.2 Scores: `dual_matmul(Q, dual_transpose(K, -2, -1)) * scale`
- [x] 3.3 Weights: `dual_softmax(scores, dim=-1)`
- [x] 3.4 Output: `dual_matmul(attn, V)` → merge heads → W_O; 26 tests passing

---

## Phase 4 — Full Model ✅
Goal: `DualTransformerClassifier` end-to-end in `src/model.py`, forward pass produces correct shape.

- [x] 4.1 `DualEmbedding`: `emb_real [V,d]` + `emb_dual [V,d]`; `emb_dual` init to zero; lookup by token index
- [x] 4.2 Pool: `mean(out.real, dim=1)`
- [x] 4.3 Classifier head: `nn.Linear(d, 2)` on pooled real vector; 34 tests passing

---

## Phase 5 — Synthetic Dataset ✅
Goal: balanced 50/50 binary dataset in `src/data/synthetic.py`, loaders ready for training.

- [x] 5.1 `FirstLastMatchDataset`: label = `tokens[0] == tokens[-1]`, vocab=32, seq_len=16
- [x] 5.2 Positive: `tokens[15] = tokens[0]`; negative: `tokens[15] != tokens[0]` (explicit rejection sample)
- [x] 5.3 `make_dataloaders(4000, 1000, batch_size=64)` → `(train_loader, test_loader)`
- [x] 5.4 `majority_baseline()` returns 0.5

---

## Phase 6 — Training Loop ✅
Goal: model trains to >85% test accuracy with per-epoch gradient norms logged.

- [x] 6.1 `train_epoch` / `evaluate` in `src/training/loop.py`
- [x] 6.2 AdamW, lr=1e-3, weight_decay=1e-4, 20 epochs, cross-entropy loss
- [x] 6.3 Log `‖∂L/∂W_real‖` and `‖∂L/∂W_dual‖` per layer per epoch
- [x] 6.4 Test accuracy >85%, majority baseline (50%) printed

---

## Phase 7 — Research Diagnostics ✅
Goal: two experiments confirm b-components are dead; minimal fix proposed and discussed.

**Hypothesis**: `out.real = softmax(Q_r K_r^T / √d) @ V_r` — dual weights never enter `out.real`, so `∂L/∂W_dual = 0`.

- [x] 7.1 **Exp A** (gradient norms): confirm `‖∂L/∂W_dual‖ ≈ 0` throughout training
- [x] 7.2 **Exp B** (ablation): full-dual vs b=0-frozen → identical test accuracy
- [x] 7.3 **Exp C** (perturbation): perturb `W_dual` by σ ∈ {0.1, 1.0, 10.0} post-training → no accuracy change
- [x] 7.4 Minimal fix: `logit = linear_a(out.real) + linear_b(out.dual)` — discuss whether this is still a valid algebraic continuation

---

## Phase 8 — Report ✅
Goal: `report.md` with formulas, results table, conclusions, and 2 external source citations.

- [x] 8.1 Formulas: dual matmul, dual softmax, gradient flow proof
- [x] 8.2 Results table: accuracy vs majority baseline; `‖∂L/∂W_real‖` vs `‖∂L/∂W_dual‖`
- [x] 8.3 Explanation of dead b-components via α²=0
- [x] 8.4 Citations: one AD/dual-numbers paper + one hypercomplex/quaternion attention paper

---

## Phase 10 — b-Component Contribution Audit ✅
Goal: rigorously determine whether the b-component of weights and representations makes any meaningful contribution to predictions, vs. a plain real-valued network of equivalent size.

**Hypothesis to test**: the dual model with `d` dimensions is functionally identical to a real-only model with `d` dimensions (not `2d`), because b never reaches the loss.

- [x] 10.1 **RealTransformerClassifier**: implement a plain `nn.`-based transformer with `d_model` dims and the same depth/heads as the dual model — no dual arithmetic, no W_dual.  Param count: `~d²` per projection vs. `~2d²` in dual model.
- [x] 10.2 **Matched-parameter baseline**: a second real model with `d_model_big` chosen so its param count equals the dual model's total (real + dual params). This is the honest apples-to-apples control.
- [x] 10.3 **Three-way training run**: train all three models (dual, real-small, real-big) on the same seeds and loaders; record final test accuracy + convergence curve.
- [x] 10.4 **b-magnitude tracking**: during training of the dual model, log `mean |b_repr|` (the dual part of token representations after each attention block) to confirm it stays at zero or grows — evidence that b is never written.
- [x] 10.5 **Fixed-dual variant** (from 7.4): `logit = head_real(out.real) + head_dual(out.dual)` — train this and compare; if it outperforms real-big, b carries signal; if not, dual algebra adds nothing.
- [x] 10.6 Report section update: table with param counts, test accuracies, and b-magnitude norms for all four variants; clear verdict on research question.

---

## Phase 11 — Explicit Source Integration ✅
Goal: satisfy the requirement "ознакомьтесь с двумя источниками и явно свяжите прочитанное со своими экспериментами".

- [x] 11.1 Add inline `[1]`/`[2]` citations at every point in the report body where a source is directly relevant (function-lift rule, gradient proof, hypercomplex contrast).
- [x] 11.2 Add a dedicated "Connection to Prior Work" section that explicitly names what each paper says and how each experimental finding confirms, extends, or contrasts it.

---

## File Layout

```
├── main.py
├── src/
│   ├── algebra.py
│   ├── model.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic.py
│   └── training/
│       ├── __init__.py
│       ├── loop.py
│       └── diagnostics.py
├── log/
├── docs/
└── report.md
```

---

## Milestones

| # | Task | Done? |
|---|---|---|
| 0 | Deps installed, src/ structure, smoke test | [x] |
| 1 | DualTensor primitives + unit tests | [x] |
| 2 | DualLinear layer | [x] |
| 3 | DualAttention block | [x] |
| 4 | Full model, forward pass correct shape | [x] |
| 5 | Dataset generation | [x] |
| 6 | Training >85% accuracy + grad norm logging | [x] |
| 7 | Ablation + perturbation experiments | [x] |
| 8 | Report written | [x] |
| 10 | b-component contribution audit (real vs dual, matched params) | [x] |
| 11 | Explicit source integration — inline citations + prior-work section | [x] |
