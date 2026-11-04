# Quartic Born attention: phases disappear, key norms do not

An original, dependency-free educational implementation of two kernels from
**Reading Why: Born-Measured Attention with Exact Word Evidence on Light and
Qubits**, Teerawat Tongloy, Jiraporn Pongsopa, and Rangsan Jomtarak.
[arXiv:2611.05355v1](https://arxiv.org/abs/2611.05355v1) was submitted
**2026-11-03 at 18:52:44 UTC**. This example was added on 2026-11-04.
[Paper HTML, Section 2](https://arxiv.org/html/2611.05355v1#S2).

## Implemented mechanism

For supplied complex query, key and value vectors, Eq. 2 gives

```text
s_ij = sum_l conjugate(q_il) * k_jl
w_ij = |s_ij|^4 / sum_r |s_ir|^4
y_i  = sum_j w_ij v_j
```

The paper subsequently applies an output projection and residual addition.
This implementation returns the weights and mixed values only; it does not
implement that projection, the residual, or learned Q/K/V projections.

Eq. 4 detects a supplied complex field phi, already pooled and projected:

```text
phi = R sum_t h_t                     # supplied by caller, not computed here
P_g = sum_{b in S_g} |phi_b|^2 / sum_b |phi_b|^2
b_g = |S_g| / N
```

`born_readout` returns P and the pixel-count baseline b for disjoint nonempty
regions partitioning all N pixels. Pooling is coherent: sum amplitudes before
squaring. Summing intensities instead would discard interference.
The demo explicitly sums slots `[1, 1]` and `[-1, 1]` with identity R:
`phi = [0, 2]` produces probabilities `[0, 1]`, whereas incoherent addition
of the same per-slot intensities gives `[0.5, 0.5]`. Reversing only the second
slot phase instead produces `[1, 0]`; a common phase on all slots changes
neither class probability.

## Why the distinction matters

Our fixed query is `[1, i]`, the keys are `[1, 0]` and `[0, 2i]`, and the values
are the two coordinate basis vectors. Overlap magnitudes are 1 and 2, so attention
weights are **1/17 and 16/17**. An identity readout on the mixed value produces
class probabilities **1/257 and 256/257**.

Normalizing each key first instead produces equal weights. Likewise, squared
normalized fidelity, `(|q†k|² / (||q||² ||k||²))²`, gives equal weights here.
Neither is the unnormalized-overlap mechanism implemented above.

Independent unit-modulus phases on queries or keys preserve attention weights.
They preserve mixed values only when values stay fixed. This is not a claim
that arbitrary phases on full transformer inputs preserve model predictions:
the values, residuals, and coherent readout can change.

## Run and verify

From the repository root, using **CPython 3.14.5** and no packages:

```bash
python -m examples.born_attention.demo
python -m examples.born_attention.demo --check-sample
python -m unittest examples.born_attention.test_model -v
```

The checked JSON is rounded to 12 decimal places and includes the interpreter
version. No random seed or download is needed. `--check-sample` verifies an
existing sample; it never rewrites it.

Tests include an independent exact-Fraction real/imaginary oracle, conjugation,
phase invariance, common scale invariance, key-norm sensitivity, permutation
behavior, input immutability, readout baselines, coherent cancellation, invalid
shapes and nonfinite inputs. The Fraction oracle deliberately uses direct
unscaled polynomial scores rather than production normalization helpers.

## Numerical and scope limits

Inputs must be finite nonempty rectangular numeric sequences. Strings and
booleans are rejected. Query/key widths and key/value counts must agree. Zero
attention rows or zero readout intensity raise `ValueError`: the equations are
undefined there, and no uniform fallback or epsilon is invented.

We divide magnitudes by their row maximum before taking fourth or second
powers. This avoids overflow/underflow from a common magnitude scale at that
stage, but does not guarantee arbitrary dynamic-range accuracy. Extremely small
relative weights can underflow, and complex dot products can overflow or suffer
cancellation; nonfinite intermediate magnitudes are rejected. This is ordinary
floating-point educational code, not a numerically certified kernel.

The paper describes a larger trained architecture. This example implements no
embedding, feed-forward layer, full transformer, training, masking, autoregressive
cache, hardware circuit, or integrated-gradient evidence. It makes no accuracy,
interpretability, quantum-advantage, throughput, or benchmark-reproduction claim.
Callers must select visible keys themselves if they need causal attention.

## Attribution and license

The paper is distributed under **CC BY 4.0** (see its arXiv license link).
The code here is independently written from the mathematical description, not
copied from the authors code or trained models, and uses this repository MIT
license. The paper and its authors are credited above; no endorsement is implied.
