# MERLIN: route with alignment mean **and** variance

**Paper:** Hee Seung Hwang, Sanghyuk Chun, Xindi Wu, Olga Russakovsky and Zhiwei
Deng, [*Multi-State Linear Attention with Generalized Second-Order Routing*](https://arxiv.org/abs/2610.37645v1),
arXiv:2610.37645 **v1, October 27, 2026** (preprint).
Example published October 29, 2026.

This original, standard-library example isolates the routing mechanism in
[§3, equations 7–9](https://arxiv.org/html/2610.37645v1#S3), followed by aggregation
of **supplied segment readouts** (equation 4). It does not implement the paper's
trained models or produce the readouts from a recurrent backbone.

## Mechanism and implementation choices

For routing keys `k` in a nonempty segment, store the mean `mu` and population
covariance `C`. Given a routing query `q`, recover:

```text
mean_score     = q dot mu
variance_score = qᵀ C q
mean_weights   = softmax(tau_mean * mean_scores)       # across segments
var_weights    = softmax(tau_variance * variance_scores)
weights        = gamma * mean_weights + (1-gamma) * var_weights
output         = sum_i weights[i] * supplied_readout[i]
```

**Two normalized distributions are mixed.** Applying a single softmax to a
weighted sum of scores is a different operator. `gamma=1` selects mean-only
routing; `gamma=0` selects variance-only routing. The temperatures **multiply**
scores; zero makes that branch uniform. This toy accepts only nonnegative
finite temperatures and a scalar gate in `[0,1]`.

The paper stores raw second moments `Sigma = mean(k kᵀ)`. Our centered covariance
`C = mean((k-mu)(k-mu)ᵀ) = Sigma - mu muᵀ` gives the same population variance in
exact arithmetic, without subtracting large raw moments. A first-key coordinate
shift preserves small spreads around large common offsets. Off-diagonal terms
matter: keys `(1,1),(-1,-1)` have zero alignment variance for query `(1,-1)`;
keeping only diagonal variances incorrectly gives two.

`SegmentMoments(keys, prefix_length=n)` summarizes only the first `n` keys and
discards the token array. Omit invisible segments entirely. The caller must also
supply causal readouts: this API cannot remove future information already baked
into a readout. This is a batch summary builder, not an online update algorithm.
All inputs are validated, including keys beyond a requested prefix.

## Reproduce

From the repository root, with **CPython 3.14.5** (the root `.python-version`):

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s examples/merlin_routing -v
python -m examples.merlin_routing.demo
python -m examples.merlin_routing.demo > /tmp/merlin-output.json
diff -u examples/merlin_routing/sample_output.json /tmp/merlin-output.json
```

Windows activation: `.venv\Scripts\Activate.ps1`. No pip install, GPU, model
weights, data download, paid API or third-party Python package is needed. The
demo uses **no randomness**; randomized oracle tests use seed **20261029**.
The checked output was actually generated with CPython 3.14.5. JSON numbers are
rounded to nine decimal places; tests also check unrounded numerical tolerances.

## Actual output: a mean tie can hide a spread

For segments `[-2,2]` and `[0,0]`, query `1`, both temperatures `1`, gate `0.5`,
and readouts `[1,0]` and `[0,1]`:

| Quantity | First segment | Second segment |
| --- | ---: | ---: |
| Mean alignment | 0 | 0 |
| Population alignment variance | 4 | 0 |
| Mean-only routing | 0.5 | 0.5 |
| Variance-only routing | 0.982013790 | 0.017986210 |
| MERLIN mixture / output coordinates | **0.741006895** | **0.258993105** |
| Dense token-softmax segment mass | 0.790012829 | 0.209987171 |

The larger variance increases the first segment's routing weight on this chosen
input. It does **not** establish that variance identifies useful information.
The dense diagnostic computes token softmax directly and sums mass by segment;
it is not the router or a token-level output reproduction. See
[`sample_output.json`](sample_output.json) for the full checked report.

## Counterexample: two moments are not the whole distribution

`[-5,0,0,5]` and `[-4,-3,3,4]` both have mean `0` and variance `12.5`, but fourth
moments `312.5` and `168.5`. At query `1`, the moment router ties at `0.5 / 0.5`;
dense token-softmax gives **0.668023139 / 0.331976861**. Higher moments matter.
This also prevents mistaking the paper's cumulant motivation for exact softmax
reconstruction.

Repeating every token within just one segment leaves its mean and variance
unchanged, hence leaves this routing unchanged. Dense grouped softmax instead
includes segment token counts. Equal-sized segments are the paper's setting;
our unequal-size API demonstrates that distinction rather than adding a hidden
length correction.

## Verification and boundaries

The **36 tests** cover an independent 80-digit Decimal oracle computed from
individual scalar query-key scores (not matrix moments), seeded multidimensional
fixtures, population variance, covariance cross-terms, separate softmax mixing,
gate endpoints, zero queries/temperatures, singleton segments, normalization,
permutation behavior, token replication, causal prefixes, large common offsets,
large logits and tiny temperatures, convex readout aggregation, invalid inputs,
nonfinite/overflow rejection, both counterexamples, and exact CLI reproduction.
The repository's existing CI runs them and compares the checked output.

Python floats have finite range and precision. Nonrepresentable means,
covariances or query scores raise `ValueError`, even if another algebraic
arrangement could avoid an intermediate overflow. Softmax handles saturated
negative log weights; near-boundary convex readouts are clamped to their input
range only to correct roundoff. Tiny negative quadratic forms within an explicit
roundoff tolerance are clamped to zero. This is not arbitrary-precision code.

No learned projections, gate training, learned temperatures, segmentation
policy, Gated DeltaNet/Mamba backbone, language-model training, performance
benchmark or full-paper replication is included. No speed, memory savings or
LLM-quality gains are claimed. Each summary stores `O(d_r²)` numbers; retaining
all `L` segments costs `O(L d_r²)` state and routing a query costs `O(L d_r²)`.
**State grows as segments accumulate**, rather than staying constant in context
length. Supplied readouts have their own storage and computation costs, omitted
from those routing-only counts.

## Sources and license

The [versioned primary paper](https://arxiv.org/abs/2610.37645v1) is CC BY 4.0.
All code, tests, documentation and constructed fixtures here are original and
covered by the repository's [MIT license](../../LICENSE). No upstream
implementation, paper text/figures, weights, real data or unpublished material
is copied or redistributed. The citation attributes the mechanism, not author
endorsement of this teaching example.
