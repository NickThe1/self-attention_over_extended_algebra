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

## Phase 4 — Full Model
Goal: `DualTransformerClassifier` end-to-end in `src/model.py`, forward pass produces correct shape.

- [ ] 4.1 `DualEmbedding`: `emb_real [V,d]` + `emb_dual [V,d]`, lookup by token index
- [ ] 4.2 Pool: `mean(out.real, dim=1)`
- [ ] 4.3 Classifier head: `nn.Linear(d, 2)` on pooled real vector

---

## Phase 5 — Synthetic Dataset
Goal: balanced 50/50 binary dataset in `src/data/synthetic.py`, loaders ready for training.

- [ ] 5.1 `FirstLastMatchDataset`: label = `tokens[0] == tokens[-1]`, vocab=32, seq_len=16
- [ ] 5.2 Positive: `tokens[15] = tokens[0]`; negative: `tokens[15] != tokens[0]` (explicit rejection sample)
- [ ] 5.3 `make_dataloaders(4000, 1000, batch_size=64)` → `(train_loader, test_loader)`
- [ ] 5.4 `majority_baseline()` returns 0.5

---

## Phase 6 — Training Loop
Goal: model trains to >85% test accuracy with per-epoch gradient norms logged.

- [ ] 6.1 `train_epoch` / `evaluate` in `src/training/loop.py`
- [ ] 6.2 AdamW, lr=1e-3, weight_decay=1e-4, 20 epochs, cross-entropy loss
- [ ] 6.3 Log `‖∂L/∂W_real‖` and `‖∂L/∂W_dual‖` per layer per epoch
- [ ] 6.4 Test accuracy >85%, majority baseline (50%) printed

---

## Phase 7 — Research Diagnostics
Goal: two experiments confirm b-components are dead; minimal fix proposed and discussed.

**Hypothesis**: `out.real = softmax(Q_r K_r^T / √d) @ V_r` — dual weights never enter `out.real`, so `∂L/∂W_dual = 0`.

- [ ] 7.1 **Exp A** (gradient norms): confirm `‖∂L/∂W_dual‖ ≈ 0` throughout training
- [ ] 7.2 **Exp B** (ablation): full-dual vs b=0-frozen → identical test accuracy
- [ ] 7.3 **Exp C** (perturbation): perturb `W_dual` by σ ∈ {0.1, 1.0, 10.0} post-training → no accuracy change
- [ ] 7.4 Minimal fix: `logit = linear_a(out.real) + linear_b(out.dual)` — discuss whether this is still a valid algebraic continuation

---

## Phase 8 — Report
Goal: `report.md` with formulas, results table, conclusions, and 2 external source citations.

- [ ] 8.1 Formulas: dual matmul, dual softmax, gradient flow proof
- [ ] 8.2 Results table: accuracy vs majority baseline; `‖∂L/∂W_real‖` vs `‖∂L/∂W_dual‖`
- [ ] 8.3 Explanation of dead b-components via α²=0
- [ ] 8.4 Citations: one AD/dual-numbers paper + one hypercomplex/quaternion attention paper

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
| 4 | Full model, forward pass correct shape | [ ] |
| 5 | Dataset generation | [ ] |
| 6 | Training >85% accuracy + grad norm logging | [ ] |
| 7 | Ablation + perturbation experiments | [ ] |
| 8 | Report written | [ ] |
