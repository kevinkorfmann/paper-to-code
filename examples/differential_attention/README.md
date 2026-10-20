# Differential attention: subtract two attention maps

This example implements the **single-head core** of Differential Transformer
using Python's standard library. It accepts already-projected queries, keys,
and values and exposes both softmax maps, their signed difference, and the
weighted output. It is small enough to inspect without a GPU or ML framework.

## Primary source and date

Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei.
*Differential Transformer*. ICLR 2025, oral presentation.

- [Paper and submission history](https://arxiv.org/abs/2410.05258):
  first preprint **October 7, 2024**; revised **April 7, 2025**.
- [Pinned version v2, section 2.1](https://arxiv.org/html/2410.05258v2#S2.SS1):
  equations (1) and (2) define the mechanism implemented here.
- [Conference record](https://openreview.net/forum?id=OvoCm1gGhN).

This is a recent attention architecture example from ICLR 2025, not a claim
that the paper is the newest attention research.

## Equation-to-code mapping

For N tokens and head dimension d, all four Q/K matrices have shape `N × d`;
the shared V has shape `N × 2d`.

```text
A1 = row_softmax(Q1 K1ᵀ / sqrt(d))
A2 = row_softmax(Q2 K2ᵀ / sqrt(d))
W  = A1 - λ A2
Y  = W V
```

[model.py](model.py) implements this with explicit loops and standard-library
math. In causal mode it masks keys `j > i` in **both** branches before
normalization. Softmax subtracts its largest unmasked logit before
exponentiation; a completely masked row is rejected rather than returning NaN.

The paper also parameterizes its learned scalar as

```text
λ = exp(λq1 · λk1) - exp(λq2 · λk2) + λ_init
λ_init = 0.8 - 0.6 exp(-0.3 (layer - 1)).
```

`lambda_from_vectors` evaluates this formula, with a one-based layer index.
It does not train the vectors. λ is not constrained to [0,1]. The demo chooses
λ=1 to expose a cancellation identity; **this is not the paper's default
initialization**, which is 0.2 at layer 1.

## A hand-checkable experiment

For d=1, let every query be 1. Choose each key as the log of its desired
positive probability. Because the desired probabilities sum to one, softmax
recovers them. The last query in the four-token causal sequence sees all keys:

```text
A1 = [0.4, 0.2, 0.2, 0.2]
A2 = [0.1, 0.3, 0.3, 0.3]
V  = [[10, 1],
      [10, 0],
      [10, 0],
      [10, 0]]
```

At λ=1:

```text
W   = [0.3, -0.1, -0.1, -0.1]
A1V = [10, 0.4]
WV  = [ 0, 0.3].
```

The first value channel is shared by all tokens, so it cancels. The second
channel has a token-specific contrast that remains. The marker's raw magnitude
does not increase; calling this an accuracy or performance improvement would
be unjustified.

These probabilities and values are deliberately constructed. No training,
search over random seeds, or empirical dataset is involved.

## Useful identities and counterexamples

The following are mathematical consequences of the implemented operator,
illustrated and tested here:

1. **Row sums:** each softmax row sums to 1, so each W row sums to `1 - λ`.
   W can contain negative entries and is not a probability distribution.
   We do not apply another softmax, clamp negatives, or renormalize W.
2. **Constant value shift:** adding the same vector c to every value changes
   each output by `(1 - λ)c`. At λ=1, this shared component disappears,
   regardless of whether that component is useful.
3. **Identical maps:** if A1=A2 and λ=1, all output is zero. Subtraction can
   remove the desired signal too.
4. **Varying background:** replace the first value channel by `[10,9,11,12]`.
   At λ=1 the output becomes `[-0.2,0.3]`; the varying component does not
   fully cancel.
5. **First causal position:** both maps there are `[1,0,...]`; its output is
   `(1 - λ)V[0]`, hence zero when λ=1. This is expected for the isolated
   operator, not an entire residual Transformer block.

Shared noise in arbitrary pre-softmax logits does not automatically satisfy
these cancellation conditions: the maps are separately normalized.
Learning useful branch differences is outside this experiment.

## Reproduce

From the repository root, with the pinned CPython 3.14.5 environment:

```sh
python -m unittest discover -s tests -v
python -m examples.differential_attention.demo
python -m examples.differential_attention.demo --lambda-value 0.5
```

The default result is committed as [sample_output.json](sample_output.json).
JSON rounds to nine decimal places and normalizes negative zero for readability.
Changing λ affects both the main construction and its counterexamples.
`--background` changes the constant value component, not the attention logits.

## How it is tested

- An independently written scalar reference evaluates the equation using
  60-digit `decimal.Decimal` arithmetic. Tests compare both causal and
  noncausal outputs at five λ values with 13-decimal-place agreement.
- Hand-computed cases check ordinary attention at λ=0, the `sqrt(d)` scaling,
  signed sums, the constructed cancellation, and both counterexamples.
- Perturbing future queries, keys, and values leaves earlier causal outputs
  unchanged. Permuting tokens commutes with the noncausal operator.
- Linearity in V and the derivative `dY/dλ = -A2V` are checked, including
  a central finite-difference comparison.
- Tests cover stable softmax, masks, nonfinite values, overflow, ragged or
  mismatched shapes, initialization, CLI errors, input immutability, and
  the committed output.

The oracle uses small logits so it can directly exponentiate at high
precision, independently of the production stable-softmax implementation.
Assertions use numerical tolerances; they are not claims of exact floating
point arithmetic. One seeded fixture tests linearity; the demo is seed-free.

## Scope and limitations

| Included | Omitted |
| --- | --- |
| Two scaled softmax maps and their difference | Learning Q/K/V projection matrices |
| Shared values of width 2d | Multi-head assembly and output projection |
| Optional causal mask | Headwise RMSNorm and the paper's fixed post-norm scale |
| λ parameterization evaluated numerically | Optimizer, backpropagation, or training |
| Explicit tiny CPU matrices | RoPE, KV cache, FlashAttention, or GPU kernels |
| Mathematical and numerical tests | Residual block, feed-forward network, tokenizer, checkpoints, datasets |

The outputs are **before** the paper's headwise normalization. That distinction
matters: the full architecture includes additional operations and a residual
path. This code does not reproduce its training, language-model evaluations,
scaling results, or performance claims. It is neither fast enough nor designed
for production inference. Runtime is quadratic in sequence length, and dense
attention maps are materialized.

The code was written independently from the equations. No upstream
implementation was copied.
