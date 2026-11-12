# Loopback: a budget for depth updates

Source: Ouyang et al., **Loopback Transformer: Towards a Unified View of Residual
Looped Transformers**, [November 4, 2026 preprint, v1](https://arxiv.org/abs/2611.07226v1).
This original educational implementation covers equations 14–16 and the
output-versus-update distinction in section 3.2. It does not reproduce the
Transformer, train queries, or substantiate language-model or efficiency gains.

## Implemented reader

For `n` completed block applications, stack size `K`, retention `0 < lambda <= 1`,
and update index `0 <= i < n`:

```text
prior_i = lambda ** (floor(n/K) - floor(i/K))
B       = sum(prior_i)
key_i   = delta_i / sqrt(mean(delta_i**2) + epsilon)
score_i = dot(query, key_i) / sqrt(d)
c_i     = B * softmax(score_i + log(prior_i))
h_next  = h_initial + sum(c_i * delta_i)
```

Decay occurs at each completed iteration, including the final one. The query
reallocates `B`; it does not change its total. The input remains separate.
The new mixture **replaces** the old mixture. Stored updates come from the
current trajectory, not a separately evaluated reference pass. `rollout` applies
supplied toy callables with shared block order, not actual Transformer blocks.

## Constructed experiments and actual output

These are our own fixtures, not results from the paper.

* With `K=2, lambda=0.5`, the prior evolves `[1]`, `[0.5,0.5]`,
  `[0.5,0.5,1]`, `[0.25,0.25,0.5,0.5]`. The last total is `1.5`.
* Supplied scores `[log(6),0,0,0]` yield coefficients
  `[9/11,3/22,3/11,3/11]`: an old update can win while total mass remains `1.5`.
  With values `[1,0,0,0]` and initial value `10`, the read is `10.818181818`.
* Starting at `10`, equal weighting of cumulative states `[11,9]` gives `10`.
  Equal weighting of their separate updates `[1,-2]`, then adding the initial
  state, gives `9.5`. The first comparison keeps the first update's coefficient
  at one; the second does not.
* Constant scalar updates of one, `K=1`, and retention one half produce
  `[10.5,10.75,10.875,10.9375]`. Incorrectly adding each entire new mixture to
  the previous state produces `[10.5,11.25,12.125,13.0625]`.
* **Bounded coefficient mass is not a stability guarantee.** For our expansive
  block `f(h)=3h`, initial `h=1`, zero queries, `K=1`, and retention one half,
  the true recurrence is `h_next=1.5*h+0.5`. After eight steps the state is
  `50.2578125`, even though total coefficient mass is `0.99609375 < 1`.
  Updates themselves depend on the growing state. This does not contradict
  a bound on their coefficients, nor establish that trained models diverge.

The supplied-score fixture isolates allocation. A separate vector fixture and
80-digit Decimal test exercise RMS-normalized scoring. Tests also cover a
nonzero-query rollout, where updates depend on earlier learned-reader-style
outputs rather than a fixed list. Queries are supplied, never optimized.

## Reproduce

From the repository root, with **CPython 3.14.5**:

```sh
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/loopback_memory -t . -v
python -m examples.loopback_memory.demo
python -m examples.loopback_memory.demo --check-sample
```

Standard library only, no install, model weights, data, GPU, API, or randomness.
The checked JSON rounds to nine decimal places; tests compare unrounded values
with explicit tolerances. Reference interpreter: repository `.python-version`.

The 24 tests include independent exact-rational repeated transport across 600
schedule settings, an 80-digit Decimal reader, a closed-form expansive trajectory,
zero-query recurrence equivalence, prefix invariance, small/large values,
log-space prior underflow recovery, boundaries, validation, and CLI/output replay.
The root CI adds this suite to its existing attention job.

## Scope and limits

The code is a transparent Python loop with full depth history, not a fast kernel.
It has no token-axis attention, batching, projection training, gradients, learned
normalization, prefix/coda layers, or language-model evaluation. Coefficients are
nonnegative; updates and outputs need not be. The cumulative-state comparison
assumes ordinary additive states, not arbitrary reweighted nonlinear trajectories.
Floating-point extremes outside representable mass/readout are rejected; tiny
individual weights may round to zero. `epsilon` is a positive supplied constant.
No general advantage over residual architectures is established by these toys.

Original implementation, tests, fixtures and prose are MIT licensed under the
repository license. No upstream code, figures, checkpoints, or training data
are copied. The paper is credited for the mechanism, not endorsement.
