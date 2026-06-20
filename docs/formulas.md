# Mathematical Formulas

## The Algebra

Elements have the form $a + \alpha b$ where $a, b \in \mathbb{R}$ and $\alpha$ is a formal symbol with the single additional property:

$$\alpha^2 = 0$$

In code, each element is represented as a pair of tensors `(real, dual)` of identical shape.

A smooth function $f$ extends to this algebra via the **function-lift rule**:

$$f(a + \alpha b) = f(a) + \alpha f'(a)\, b$$

This is the algebraic basis of forward-mode automatic differentiation: `real` carries the value, `dual` carries the directional derivative.

---

## `dual_add`

$$
(a + \alpha b) + (c + \alpha d) = (a + c) + \alpha(b + d)
$$

Component-wise:

$$
\text{real} = a + c, \qquad \text{dual} = b + d
$$

---

## `dual_mul`

Element-wise multiplication using $\alpha^2 = 0$:

$$
(a + \alpha b)(c + \alpha d) = ac + \alpha(ad + bc) + \underbrace{\alpha^2}_{=\,0} bd
$$

$$
\boxed{\text{real} = a \odot c, \qquad \text{dual} = a \odot d + b \odot c}
$$

where $\odot$ denotes element-wise multiplication.

---

## `dual_matmul`

Matrix multiplication of $X = (A, B)$ and $W = (C, D)$, derived from the same rule:

$$
XW = (A + \alpha B)(C + \alpha D) = AC + \alpha(AD + BC)
$$

$$
\boxed{\text{real} = A C, \qquad \text{dual} = A D + B C}
$$

**Key consequence:** if $A = 0$ (pure-dual input) and $C = 0$ (pure-dual weight), both parts vanish — mirroring $\alpha^2 = 0$ at the matrix level. This is why the $b$-components of the network weights are algebraically invisible to `out.real`.

---

## `dual_softmax`

Applying the function-lift rule to $\text{softmax}$:

$$
\text{softmax}(a + \alpha b) = \text{softmax}(a) + \alpha\, J_{\text{softmax}}(a)\, b
$$

Let $s = \text{softmax}(a)$. The Jacobian of softmax is:

$$
J_{\text{softmax}}(a) = \text{diag}(s) - s s^\top
$$

Applying it to $b$:

$$
J_{\text{softmax}}(a)\, b = s \odot b - s (s^\top b) = s \odot \bigl(b - (s \cdot b)\bigr)
$$

$$
\boxed{\text{real} = \text{softmax}(a), \qquad \text{dual} = s \odot \bigl(b - (s \cdot b)\bigr)}
$$

where $s \cdot b = \sum_i s_i b_i$ is a scalar (broadcast along the softmax dimension).

**Invariant:** the dual part always sums to zero along the softmax dimension:

$$
\sum_i \text{dual}_i = \sum_i s_i (b_i - (s \cdot b)) = (s \cdot b) - (s \cdot b) \underbrace{\sum_i s_i}_{=\,1} = 0
$$

---

## `DualLinear`

A linear projection where both weights and input are dual numbers. Parameters: $W = (W_r, W_d)$ with $W_r, W_d \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, and optional bias $b \in \mathbb{R}^{d_{\text{out}}}$.

Given input $X = (X_r, X_d)$:

$$
\text{DualLinear}(X) = X \cdot W + b = (X_r + \alpha X_d)(W_r + \alpha W_d) + b
$$

Expanding with $\alpha^2 = 0$:

$$
\boxed{\text{real} = X_r W_r + b, \qquad \text{dual} = X_r W_d + X_d W_r}
$$

**Bias placement:** $b$ is real-valued ($b + \alpha \cdot 0$), so it only shifts the real part.

**Initialisation:** $W_r \sim \text{Kaiming uniform}$, $W_d = 0$. Starting from $W_d = 0$ makes the model equivalent to a standard real network at initialisation.

**Dead-weight consequence:** `out.real` $= X_r W_r + b$ has no dependence on $W_d$. PyTorch autograd never visits $W_d$ during backprop through `out.real` — its `.grad` remains `None`, not merely zero.

---

## `DualAttention`

Multi-head scaled dot-product self-attention where every step operates over dual numbers.

**Input:** $X = (X_r, X_d) \in \mathbb{R}^{B \times L \times d}$

**Step 1 — Project**

$$Q = \text{DualLinear}_{Q}(X), \quad K = \text{DualLinear}_{K}(X), \quad V = \text{DualLinear}_{V}(X)$$

Split each into $H$ heads: reshape $[B, L, d] \to [B, H, L, d_h]$ where $d_h = d / H$.

**Step 2 — Scaled scores**

$$S = \frac{1}{\sqrt{d_h}}\, Q K^\top \in \mathbb{R}^{B \times H \times L \times L}$$

Computed as `dual_matmul(Q, dual_transpose(K, -2, -1)) * scale`, so:

$$\boxed{S_r = \frac{Q_r K_r^\top}{\sqrt{d_h}}, \qquad S_d = \frac{Q_r K_d^\top + Q_d K_r^\top}{\sqrt{d_h}}}$$

**Step 3 — Attention weights**

$$A = \text{dual\_softmax}(S, \text{dim}=-1)$$

$$\boxed{A_r = \text{softmax}(S_r), \qquad A_d = A_r \odot \bigl(S_d - (A_r \cdot S_d)\bigr)}$$

**Step 4 — Weighted sum**

$$C = A V$$

$$\boxed{C_r = A_r V_r, \qquad C_d = A_r V_d + A_d V_r}$$

**Step 5 — Output projection**

$$\text{out} = \text{DualLinear}_{O}(\text{merge\_heads}(C))$$

**Key observation across all steps:** tracing only the real components:

$$\text{out}_r = \text{softmax}\!\left(\frac{X_r W_{Q_r} W_{K_r}^\top X_r^\top}{\sqrt{d_h}}\right) X_r W_{V_r} W_{O_r}$$

$W_{Q_d}, W_{K_d}, W_{V_d}, W_{O_d}$ never appear in $\text{out}_r$. All four dual weight matrices have `.grad = None` after backpropagating a loss on `out.real`.

---

## `dual_transpose`

Transposition is a linear map, so its own derivative is itself:

$$
(A + \alpha B)^\top = A^\top + \alpha B^\top
$$

$$
\boxed{\text{real} = A^{\top}, \qquad \text{dual} = B^{\top}}
$$
