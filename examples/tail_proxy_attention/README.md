# OpenAttention's residual tail-proxy component

An original, dependency-free implementation of one component from Yu et al.,
[*OpenAttention: One Percent Is Enough with Tail Proxies*](https://arxiv.org/abs/2610.30951v2).
The preprint was first submitted **October 22, 2026**, and v2 was posted
**October 23, 2026**. This example implements equations 1 and 8 with supplied
exact-token indices and constructed groups. It does not implement the complete
OpenAttention system or reproduce its model evaluations.

## The calculation

For one query, inputs `z[i]` are **already scaled logits** (`q·k[i]/sqrt(d)`).
Choose exact tokens `X` and partition the remaining visible tokens into groups.
For each residual group `b`, form:

```text
n[b]       = number of residual members
mean_z[b]  = sum(z[i] for i in b) / n[b]
mean_v[b]  = sum(v[i] for i in b) / n[b]
proxy_z[b] = mean_z[b] + log(n[b])
```

Apply one softmax to the exact logits and proxy logits, then mix exact values
and proxy mean values with those weights. Equivalently, each residual member
receives unnormalized weight `exp(mean_z[b])`; group mass is
`n[b]*exp(mean_z[b])`. All exact tokens and masked positions must be removed
**before** recomputing the count and both means.

The API in [model.py](model.py) accepts groups that partition **all** input
positions, including exact and masked positions. It performs that removal
itself and omits empty residual groups. Duplicate, missing or invalid indices
are rejected. Returned diagnostics expose the residual members, means,
component weights and effective weights over the original tokens. Values may
be vectors. `count_weighting=False` is an intentionally incorrect no-count
ablation used to illustrate the count term; it is not the cited estimator.

The count accounts for multiplicity. It does **not** generally preserve dense
unnormalized mass. Convexity of the exponential gives

```text
n * exp(mean(z)) <= sum(exp(z))
```

with equality for equal logits. When a group has equal logits, its arithmetic
value mean produces the exact dense numerator as well as the exact mass,
even when values differ. Thus equal-logit groups, singleton groups, and the
all-exact limit recover dense attention. Variable logits can distort both
normalization and the weighting of values, so output accuracy has no universal
ordering relative to simply discarding the tail.

## Executed examples

These are deterministic constructed scalar inputs; there is **no random seed**,
training, dataset, model checkpoint or GPU. Both cases keep token 0 exact and
place all positions in one initial group before removing that token.

| Constructed input | Dense | Residual proxy | Discard tail | Omit count |
| --- | ---: | ---: | ---: | ---: |
| `z=[log(4),0,0,0]`, `v=[1,2,4,6]` | 2.285714285714 | 2.285714285714 | 1 | 1.6 |
| `z=[0,-2,2]`, `v=[0,1,0]` | 0.015876239976 | 0.333333333333 | 0 | 0.25 |

In the first case, the residual count is 3, mean logit is 0, and mean value is
4. The combined output is `(4*1 + 3*4)/(4+3) = 16/7`. Removing the count changes
it to `(4*1 + 1*4)/(4+1) = 8/5`.

In the second case the true output is
`exp(-2)/(1+exp(-2)+exp(2))`. The proxy instead uses residual mean logit 0 and
mean value 1/2, yielding `1/3`. Discarding the tail gives 0 and is **closer for
this constructed case**. The high-logit tail member has value 0, so replacing
both tail weights by a common weight changes the value contribution. Splitting
the tail into singletons restores dense attention.

The first example reduces softmax entries from 4 to 2, while this transparent
implementation still **reads every input** to build its means. That count is
not a cache-read fraction or a runtime measurement. No 1% budget experiment,
speedup or language-model quality claim follows from it.

A causal-prefix check appends a future token with logit `1e6` and value `1e100`.
With its mask set false, the residual membership stays `[1,2,3]`, and the output
stays `16/7` with an observed difference of zero. This checks masking before
aggregation for supplied groups. It does not validate a future clustering or
selection algorithm against information leakage.

Complete diagnostics are in [sample_output.json](sample_output.json).

## Reproduce

From the repository root, use the existing standard-library environment,
pinned to **CPython 3.14.5** by [`.python-version`](../../.python-version).
[requirements.txt](../../requirements.txt) records the empty dependency set.

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/tail_proxy_attention -v
python -m examples.tail_proxy_attention.demo > /tmp/tail-proxy-output.json
diff -u examples/tail_proxy_attention/sample_output.json /tmp/tail-proxy-output.json
```

Explore how within-group logit spread affects the constructed counterexample:

```sh
python -m examples.tail_proxy_attention.demo --spread 0
python -m examples.tail_proxy_attention.demo --spread 4
```

The CLI accepts `0 <= spread <= 100`, sets the two tail logits to `-spread` and
`+spread`, and leaves the equal-logit example unchanged. Printed floats are
rounded to 12 decimal places. CI checks default output byte for byte.

## Tests and numerical scope

The **30 tests** include an independent **70-digit Decimal oracle** evaluating
weights on the original tokens, equal-logit exactness, the no-count ablation,
the adverse counterexample, singleton/all-exact limits, residual recomputation,
empty-group removal, masking, causal-prefix invariance, query/mean-key
linearity, constant values, shifts, permutations, linearity in values,
nonnegative normalized weights, convex output, the Jensen mass gap,
partition refinement for equal logits, extreme logits/values, invalid inputs,
CLI behavior and exact checked output.

Numerical comparisons normally use absolute tolerance `2e-12`; a common shift
of 10,000 uses `1e-11`. These are arithmetic checks, not statistical bounds.
The implementation centers component mean logits **before** adding the count
term, so a huge common offset cannot swallow `log(n)`. Means and output mixes
use scaled arithmetic to avoid avoidable overflow. Tests cover logits of
±10,000, a common offset of `1e308`, and cancellation with opposite finite
extremes. Very negative relative logits may underflow to zero weight. Ordinary
binary floating-point rounding remains; this is not an arbitrary-precision
production kernel. Nonfinite input values are rejected, including masked ones.

## Scope and provenance

Groups and exact indices are explicitly supplied. This example contains no
Query-Geometry Clustering, Query-Geometry Selector, ErrorGate allocation,
online FlashAttention implementation, cached-statistics maintenance, training
or model integration. It demonstrates the residual estimator with a useful
exact case and a failure case. The paper discusses predecessors using mean
tails and centroid proxies; this elementary primitive is **not claimed to have
been invented in October 2026**.

Primary reference: Yu et al., [versioned paper](https://arxiv.org/abs/2610.30951v2),
[HTML equations 1 and 8 and Appendix A](https://arxiv.org/html/2610.30951v2).
The paper is **CC BY 4.0**; the [official implementation](https://github.com/hyunwoo137/OpenAttention)
is **Apache-2.0**. No upstream code, diagrams or prose are copied here. This
repository's implementation, constructed examples, tests and explanation are
original and use its MIT license. No private inputs or research are included.
