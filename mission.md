# Mission

## What

Implement a scaled dot-product self-attention block over the **dual-number algebra** — the commutative ring of pairs (a, b) where the element a + αb obeys α² = 0 — and use it to train a binary sequence classifier. Then rigorously investigate whether the algebraic b-component actually contributes to the model or is silently a dead weight.

## The Algebra

An element is a pair of real tensors `(a, b)`, representing `a + αb`:

| Operation | Rule |
|---|---|
| Addition | `(a+αb) + (c+αd) = (a+c) + α(b+d)` |
| Multiplication | `(a+αb)(c+αd) = ac + α(ad+bc)` |
| Function lift | `f(a+αb) = f(a) + α f'(a) b` |

This is the algebra of **dual numbers**, isomorphic to the truncated polynomial ring ℝ[α]/(α²). It is the algebraic object underlying forward-mode automatic differentiation: the b-component tracks the first-order directional derivative of a at the point b.

## What We Build

A minimal transformer-style model where:
- Token embeddings are dual numbers (learned real part + learned dual part)
- Q, K, V projection weights are dual numbers
- Every internal operation (matmul, softmax, weighted sum) obeys dual-number rules
- The classifier reads **only the real part** `a` of the pooled output

Trained on a synthetic task: predict whether the first and last tokens in a length-16 sequence (vocabulary size 32) are equal.

## The Research Question

Does the b-component of the weights and representations carry any useful information, or does the algebra make it structurally invisible to the loss?

**Anticipated finding**: because out_a = softmax(Q_a K_a^T / √d) V_a — the real part of the output depends only on real parts of weights — the b-components receive zero gradient and are dead. The model is computationally equivalent to a standard real attention network of the same size.

The assignment asks us to verify this, explain it through the algebra, and propose a minimal fix.

## Deliverables

1. **Code** — runnable PyTorch implementation with a single entry point
2. **Report** — formulas, two diagnostic experiments, results table, conclusions, and links to two external sources (one on automatic differentiation / algebraic extensions, one on hypercomplex/quaternion attention in NLP)
