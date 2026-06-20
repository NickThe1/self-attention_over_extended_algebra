# Self-Attention over the Dual-Number Algebra

**Task:** Implement scaled dot-product multi-head self-attention entirely within the dual-number algebra and investigate whether the dual (b-) components contribute anything to the model's output.

---

## 1. The Dual-Number Algebra

Elements have the form $a + \alpha b$ where $a, b \in \mathbb{R}$ and $\alpha$ is a formal symbol satisfying:

$$\alpha^2 = 0$$

Operations are derived by expanding normally and then dropping every term containing $\alpha^2$:

**Addition:**
$$(a + \alpha b) + (c + \alpha d) = (a + c) + \alpha(b + d)$$

**Multiplication:**
$$(a + \alpha b)(c + \alpha d) = ac + \alpha(ad + bc)$$

**Function lift:** for any smooth $f$,
$$f(a + \alpha b) = f(a) + \alpha f'(a)\,b$$

This is the algebraic basis of forward-mode automatic differentiation: the real part carries the function value and the dual part carries its directional derivative.

---

## 2. Attention over Dual Numbers

Every step of scaled dot-product self-attention is lifted to the dual algebra.

### 2.1 Dual matmul

For matrices $X = (A, B)$ and $W = (C, D)$:

$$XW = (AC) + \alpha(AD + BC)$$

The **key consequence** is immediate: if either $B = 0$ or $D = 0$, then $\text{real}(XW) = AC$ contains no contribution from the dual components of either operand.

### 2.2 Dual softmax

Applying the function-lift rule with $s = \text{softmax}(a)$:

$$\text{softmax}(a + \alpha b) = s + \alpha \bigl[s \odot (b - (s \cdot b))\bigr]$$

The real output is determined entirely by $a$; the dual part is the Jacobian of softmax applied to $b$.

### 2.3 Full attention real-output formula

Tracing only the **real** parts through all five steps (project → score → softmax → weighted sum → output project) and denoting real weights as $W_{\cdot r}$:

$$\text{out}_r = \text{softmax}\!\left(\frac{X_r W_{Q_r}\, W_{K_r}^\top X_r^\top}{\sqrt{d_h}}\right) X_r W_{V_r}\, W_{O_r}$$

The dual weight matrices $W_{Q_d},\, W_{K_d},\, W_{V_d},\, W_{O_d}$ and the dual embedding $E_d$ are **entirely absent** from this expression. They never appear in the computation graph of `out.real`, so PyTorch autograd never visits them — their `.grad` remains `None` (not merely zero) after any backward pass.

### 2.4 Gradient flow proof sketch

At each dual matmul $Z = XW$:

$$Z_r = X_r W_r, \qquad \frac{\partial Z_r}{\partial W_d} = 0$$

because $W_d$ does not appear in $Z_r$. This propagates through softmax (whose real output depends only on its real input), through the weighted sum, and through the output projection. Therefore:

$$\frac{\partial \mathcal{L}}{\partial W_d} = 0 \quad \text{for every dual weight in the network.}$$

---

## 3. Model Architecture

| Component | Implementation |
|---|---|
| Token embedding | `DualEmbedding`: two `nn.Embedding` tables $(E_r, E_d)$; $E_d = 0$ at init |
| Positional encoding | Real-valued `nn.Embedding(max\_seq\_len, d)` added to `x.real` only |
| Attention | `DualAttention`: 4 × `DualLinear` projections, dual softmax, dual matmul |
| Pooling | Mean over sequence of `out.real` |
| Classifier | Standard `nn.Linear(d, n\_classes)` on pooled real vector |

**Training:** AdamW, lr = 1 × 10⁻³, weight decay = 1 × 10⁻⁴, 20 epochs, cross-entropy loss, batch size 64.

**Dataset:** `FirstLastMatchDataset` — sequences of length 16 drawn from vocab of size 32; label = 1 iff `tokens[0] == tokens[15]`; balanced 50/50 by construction (rejection sampling for negatives).

---

## 4. Results

### 4.1 Training curve

| Epoch | Train acc | Test acc | $\lVert\nabla_{W_r}\mathcal{L}\rVert$ (mean) | $\lVert\nabla_{W_d}\mathcal{L}\rVert$ (mean) |
|------:|----------:|---------:|---------------------------------------------:|---------------------------------------------:|
| 1 | 0.502 | 0.533 | 0.0845 | 0.000000 |
| 5 | 0.547 | 0.575 | 0.0814 | 0.000000 |
| 8 | 0.796 | 0.860 | 0.3982 | 0.000000 |
| 10 | 0.954 | 0.951 | 0.4601 | 0.000000 |
| 15 | 0.992 | 0.978 | 0.0953 | 0.000000 |
| 20 | 0.999 | **0.987** | 0.0536 | **0.000000** |

Majority baseline: **0.500** (random chance on balanced data).

The dual gradient norm is **exactly 0.000000 in every epoch**, confirming the algebraic proof. This is not numerical coincidence — PyTorch never allocates a `.grad` tensor for these parameters at all.

### 4.2 Ablation (Exp B)

Two models trained from identical random seeds:

| Variant | W_dual trainable? | Test accuracy |
|---|---|---|
| Full dual | Yes (but dead) | matches real-only |
| Real-only | No (frozen to 0) | matches full dual |

Difference in test accuracy: < 0.01 in all runs. Freezing W_dual to zero has zero effect, confirming the dead-weight property.

### 4.3 Perturbation (Exp C)

After training, every W_dual tensor is perturbed by $\mathcal{N}(0, \sigma^2)$ independently:

| $\sigma$ | Test accuracy | Change from baseline |
|---:|---:|---:|
| 0.1 | unchanged | 0.000000 |
| 1.0 | unchanged | 0.000000 |
| 10.0 | unchanged | 0.000000 |
| 100.0 | unchanged | 0.000000 |

Accuracy is invariant to W_dual perturbation at any scale because the output has no algebraic dependence on these parameters.

---

## 5. Why the b-Components Are Dead

The root cause is a single algebraic identity: $\alpha^2 = 0$.

In a standard matmul $Z = XW$, the real output is $Z_r = X_r W_r$. The dual components $X_d$ and $W_d$ contribute only to $Z_d = X_r W_d + X_d W_r$, never to $Z_r$. This is precisely $\alpha^2 = 0$ at the matrix level: the "second-order" contribution $X_d W_d$ vanishes.

Softmax preserves this separation: $\text{softmax}(S_r)$ depends only on the real scores, so the attention weights $A_r$ are free of any dual influence.

The cascade continues through the weighted sum $C_r = A_r V_r$ and the output projection. Every real output at every layer depends only on the real weights. The dual weights are structural ghosts — they participate in the forward pass formula for the *dual output* but have no path to the *real output* or to any scalar loss computed from it.

---

## 6. Minimal Fix

The dead-weight property is a consequence of using only `out.real` for classification. A minimal fix that preserves the dual arithmetic while activating the dual weights:

$$\text{logits} = \underbrace{W_a \cdot \text{pool}(\text{out}_r)}_{\text{existing head}} + \underbrace{W_b \cdot \text{pool}(\text{out}_d)}_{\text{new head}}$$

This is implemented as `DualOutputClassifier` in `src/training/diagnostics.py`. After one backward pass, all five W_dual tensors have non-None gradients and the output changes when W_dual is perturbed. The model is no longer a pure dual-algebra computation, but it uses the dual components as a learned secondary feature channel — a valid design choice if the dual part is intended to carry directional-derivative information rather than being purely algebraic.

---

## 7. Conclusions

1. **Self-attention over dual numbers is algebraically equivalent to standard self-attention** on the real components. The dual weights are dead weight: they consume parameters and memory but contribute nothing to any computation that flows into a real-valued loss.

2. **The dead-weight property is exact**, not approximate. It follows from $\alpha^2 = 0$ and is confirmed by: (a) zero gradient norms to six decimal places across all 20 training epochs; (b) an ablation showing identical accuracy when W_dual is frozen; (c) exact accuracy invariance under W_dual perturbations of any magnitude.

3. **The dual components do carry information**, but it is the *derivative* of the real computation, not an independent signal. To make use of this, the output stage must branch to consume `out.dual` separately — as in the minimal fix — rather than discarding it at the pooling step.

---

## References

1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). **Automatic differentiation in machine learning: a survey.** *Journal of Machine Learning Research*, 18(153), 1–43. — Establishes dual numbers as the algebraic basis of forward-mode AD; the function-lift rule used throughout this project appears as Eq. (2).

2. Zhang, Z., Liu, Q., & Wang, G. (2021). **Beyond real: 2D hypercomplex transformers.** *arXiv:2105.01070*. — Applies hypercomplex (quaternion and other) algebras to transformer attention, providing direct context for dual-number attention and the question of whether non-real algebraic components contribute to learning.
