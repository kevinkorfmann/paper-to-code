# Continuum Attention: retrieve along the path between tokens

An original educational forward implementation of Afshin Khadangi's
[*Continuum Attention*](https://arxiv.org/abs/2610.20586v2), first posted
**October 14, 2026**, revised **October 26, 2026**. We use v2's section 2:
linear interpolation, exponential segment moments, and half-token endpoint
contributions. This is one operator on constructed inputs, not a trained
language model or a reproduction of the paper's experiments.

## What is integrated

For one query, each supplied logit `z[j]` already includes the query/key dot
product, dimension scaling, and any score gain. Connect adjacent visible
logits and value vectors with straight lines. On a unit interval between
positions `j-1` and `j`, write

```text
z(u) = (1-u)*z[j-1] + u*z[j]
v(u) = (1-u)*v[j-1] + u*v[j],       0 <= u <= 1
```

The output is the integral of `exp(z(u))*v(u)` divided by the integral of
`exp(z(u))`, summed across visible intervals and endpoint contributions.
Unlike evaluating softmax at token positions, this weights the entire
interpolated path. Only a contiguous causal prefix is included.

The implementation in [model.py](model.py) evaluates the segment integrals
analytically. To avoid exponential overflow, let `d=abs(z[j]-z[j-1])` and orient
the local coordinate from the higher-logit endpoint toward the lower:

```text
A = integral_0^1 exp(-d*u) du       = (1-exp(-d))/d
B = integral_0^1 u*exp(-d*u) du     = (A-exp(-d))/d
```

The high endpoint's value coefficient is `A-B`; the low endpoint's is `B`.
Both are multiplied by `exp(max(endpoint logits)-global shift)`. For `d=0`,
`A=1` and `B=1/2`. Near zero, a short convergent series avoids cancellation.
The closed-form integral is exact mathematically; the implementation evaluates
it with ordinary binary floating-point arithmetic, not exact real arithmetic.

The mean position within an oriented left-to-right segment is its right
coefficient divided by its mass. For logit differences `-4, 0, +4`, the executed
mean coordinates are **0.231342639636, 0.5, 0.768657360364**. These are positions
within a segment, not probabilities over the whole memory.

## Why the endpoints matter

The visible path has constant extensions of width **1/2** before its first
token and after its last. Each contributes `0.5*exp(endpoint logit)` and the
same mass times the endpoint value. With one visible token, both halves refer
to that token and the output is exactly its value.

With equal logits, an interior token receives half a coefficient from each
neighboring segment. The first and last tokens instead receive one segment
half plus an endpoint half. Every token therefore has the same total weight.
Omitting the endpoint extensions would overweight interior tokens.

[demo.py](demo.py) includes this explicit ablation and records the following
actual output in [sample_output.json](sample_output.json):

| Constructed input | Continuum output | Comparison |
| --- | ---: | --- |
| `z=[0,log(4)]`, `v=[0,1]` | 0.712764367458 | Discrete softmax: 0.8 |
| `z=[0,0,0]`, `v=[0,1,0]` | 0.333333333333 | Without endpoint caps: 0.5 |

For the first case, with `d=log(4)`, the unshifted normalizer is
`2.5 + 3/d`, and the value numerator is `2 + (4*d-3)/d^2`. Their ratio gives
the printed continuum output. The single segment's mean coordinate is
**0.611985812889**, but the full output also includes both endpoint caps.

An appended future token with logit `1e6` and value `1e100` has weight zero
when `prefix_length=2`. The visible output and last endpoint cap are unchanged;
the measured output difference is zero. The visible path stops at the last
visible token, so no segment toward that future token is constructed.

The path also makes **adjacency relevant**. For `z=[0,2,-1]`, `v=[0,1,0]`, the
output is **0.607852884793**. Jointly swapping the first two logits and values
gives **0.751681323137**. Discrete softmax would be unchanged by this joint
permutation. This illustrates a different operator, not an accuracy gain.

## Reproduce

Use **CPython 3.14.5** from the repository root. The root
[.python-version](../../.python-version) pins the interpreter and
[requirements.txt](../../requirements.txt) records the empty dependency set.

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/continuum_attention -v
python -m examples.continuum_attention.demo > /tmp/continuum-output.json
diff -u examples/continuum_attention/sample_output.json /tmp/continuum-output.json
```

Change the second logit in the two-token example:

```sh
python -m examples.continuum_attention.demo --gap 0
python -m examples.continuum_attention.demo --gap 4
```

The CLI restricts this exploratory gap to `[-100,100]`. It leaves the boundary,
causality and adjacency controls fixed. Printed floats are rounded to 12 decimal
places and default output is checked byte for byte in CI. Inputs are entirely
deterministic: no random seed, dataset, weights or external service is needed.

## Verification and limits

The **26 tests** include a **70-digit Decimal reference** using direct
exponential moments, independent composite Simpson integration of the actual
interpolated functions, near-zero and general segment formulas, monotonic
retrieval positions, full-row output and weights, flat-score uniformity,
endpoint omission, the single-token limit, constant values, linearity, common
logit shifts, whole-path reversal, adjacency sensitivity, normalized positive
weights, convex output, prefix truncation, exclusion of future tokens,
query/key interpolation, extreme inputs, invalid inputs and CLI/output checks.

Most arithmetic comparisons allow `2e-12` absolute error; segment comparisons
use `2e-13` and numerical quadrature uses `2e-11`. These tolerances check
arithmetic, not predictive accuracy. For small gaps, the implementation sums
25 series terms; the zero-gap limit is handled explicitly. Tests span gaps
near zero, the series branch boundary, logits of ±10,000, a common offset of
`1e308`, and cancellation between large opposite values. Adjacent differences
that overflow a float are rejected. Very small exponential contributions may
underflow to zero. All input numbers must be finite, including future entries.

`attend()` takes a prefix length rather than a mask with holes: removing
interior positions or connecting across them would require defining a different
path and measure. It returns full-length token weights with zeros for future
positions. `boundary_caps=False` is an explicit diagnostic ablation and rejects
the zero-measure one-token case.

This example computes only the forward operator. It contains no trained
projections, gain learning, backward kernel, tokenizer, decoder, CUDA/Metal
implementation or performance benchmark. Computing every causal row remains
quadratic in sequence length for this reference approach. No language-model
quality, speed, memory or generalization gain is inferred from these toys.
Continuous attention has earlier literature; this paper's construction should
not be taken as the invention of continuous-domain attention in general.

Primary source: Khadangi, [versioned abstract and dates](https://arxiv.org/abs/2610.20586v2),
[method, equations 2–24 and numerical evaluation](https://arxiv.org/html/2610.20586v2).
The paper carries arXiv's perpetual non-exclusive distribution license, not a
license grant for this repository. No paper prose, figures or upstream code
are copied. Our implementation, tests, explanation and constructed examples
are original and covered by this repository's MIT license.
