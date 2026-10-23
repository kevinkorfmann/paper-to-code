# Multi-head attention residuals: feature groups can read different depths

Daily paper-to-code example **2026-10-23**. Original educational implementation
of a routing operation from Cheng Luo, Zefan Cai and Junjie Hu,
[Multi-Head Attention Residuals](https://arxiv.org/abs/2607.27230v2).
The preprint was first posted **July 22, 2026**; this example follows the
**July 31, 2026 v2**, equation 3 and Appendix E / Figure 9. It is newer than
the repository's 2024 differential-attention and May 2025 gated-attention papers.

## What is being implemented

Attention residuals read earlier sublayer outputs with a softmax over **depth**.
MHAR gives contiguous feature groups separate depth distributions. The inputs
here are N available raw source vectors of width D at one token position, and
one supplied pseudo-query q of length D. These sources represent embeddings
and raw sublayer outputs, not cumulative residual sums.

```text
k_i = gain * s_i / sqrt(mean_over_D(s_i²) + epsilon)
a[h,i] = softmax_over_sources(dot(q[h], k_i[h]))
output[h] = sum_i a[h,i] * s_i[h]
output = concatenate(output[0], ..., output[H-1])
```

Full-row RMS statistics are computed **before** slicing into heads, following
equation 3 and the explicit full-D statistic in Figure 9. Values retain their
raw scale. There is no `1/sqrt(head_width)` score factor in this routing
equation. H=1 is the single-head attention-residual baseline, not a standard
additive residual connection. The same D query entries are reshaped across
heads; no extra query entries are introduced.

## Constructed experiment and actual output

Two available depths expose these four-coordinate vectors:

```text
source A = [2, 0, 0, 2]
source B = [0, 2, 2, 0]
```

The supplied query is `[b, -b, b, -b]`, with
`b = log(9)*sqrt(2+epsilon)/4` and `epsilon=1e-6`. This deliberately creates
9:1 odds in each two-coordinate head. It is not a learned parameter.

| Routing | Depth weights | Output |
| --- | --- | --- |
| One full-width head | A: 0.5, B: 0.5 | `[1, 1, 1, 1]` |
| Two feature heads | First: A 0.9 / B 0.1; second: A 0.1 / B 0.9 | `[1.8, 0.2, 1.8, 0.2]` |

The first head favors A while the second favors B. Any convex mixture of the
**full** source vectors has coordinate 0 plus coordinate 2 equal to 2. The
two-head result has sum 3.6, so one shared convex mixture cannot express it.
Each head still forms a convex mixture of its own source slices, and every
output coordinate stays within its source-coordinate range.

Two counterexamples are included: a zero query gives the ordinary depth mean
for both head counts; identical sources give the same vector regardless of
routing. The result shows an available routing behavior, not universal benefit.

## Run from the repository root

Use the root's pinned **CPython 3.14.5** and empty dependency set:

```sh
python3.14 -m venv --without-pip .venv
.venv/bin/python -c "import platform; assert platform.python_version() == '3.14.5'"
.venv/bin/python -m examples.multi_head_attention_residuals.demo
.venv/bin/python -m unittest discover -s examples/multi_head_attention_residuals -v
.venv/bin/python -m examples.multi_head_attention_residuals.demo --strength 0
.venv/bin/python -m examples.multi_head_attention_residuals.demo --strength -1
```

On Windows use `.venv\Scripts\python.exe`. There is no randomness, seed,
GPU, model download, or data dependency. Strength multiplies the constructed
query; zero disables depth preference and negative values reverse it.

[sample_output.json](sample_output.json) is the actual default output, rounded
to nine decimal places, and is checked in tests and CI:

```sh
.venv/bin/python -m examples.multi_head_attention_residuals.demo > /tmp/mhar-output.json
diff -u examples/multi_head_attention_residuals/sample_output.json /tmp/mhar-output.json
```

## Verification and limits

Eighteen tests cover an independent 60-digit Decimal oracle for H=1,2,4;
single-head reduction; full-row RMS; raw values; gain; zero queries; single and
identical sources; valid head distributions; source/head permutation; isolated
head-query changes; convex bounds; a finite-difference query derivative;
numerical extremes; invalid inputs; nonmutation of inputs; CLI and sample output.
The repository's original softmax, finite-value checks and dot product are reused.
The new implementation normalizes large finite values before squaring to avoid
overflow; positive epsilon makes all-zero rows well-defined.

This is one forward routing site at one token. There is no trained stack,
optimizer, backward kernel, block routing, delta conversion, token-attention
layer, or fused GPU implementation. Callers must supply only already available
sources; this function cannot enforce temporal availability on arbitrary input.
No quality, speed, memory, or compute benefit is inferred from these toy values.
No paper code, figures, model weights or private research are redistributed.
