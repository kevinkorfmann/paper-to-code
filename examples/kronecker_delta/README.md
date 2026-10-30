# Kronecker Delta Rule: revise a pair score, not a whole value

**Paper:** Anamaria-Roberta Hartl, Levente Zólyomi, Michael List, Günter
Klambauer, Sebastian Böck and Sepp Hochreiter,
[*Kronecker Delta Rule for Long Context Linear Attention*](https://arxiv.org/abs/2610.26834v1),
**v1, October 19, 2026** (preprint). Example published October 30, 2026.

This original standard-library implementation isolates **equation 3**, the
ungated scalar-association update. It is **not a KronLSTM reproduction**.
The comparison implements the ungated vector Delta Rule in equation 2,
not a trained DeltaNet model.

## What is being corrected?

Memory `S` has `d_v` rows and `d_k` columns. Keys and queries have length
`d_k`, values have length `d_v`. The vector readout is `S q`, but the
Kronecker rule corrects only the scalar score `vᵀ S k`:

```text
score = vᵀ S k
S_new = S + beta * (target - score) * v kᵀ        # Kronecker, Eq. 3
S_new = S + beta * (v - S k) kᵀ                  # vector Delta, Eq. 2
```

The objectives differ. The first constrains one scalar; the second constrains
an entire retrieved vector. In column-major vectorization, the first is a
scalar delta update with feature `a = k ⊗ v` (equation 4). The implementation
never constructs the large `a aᵀ` transition matrix; the independent oracle does.

For a nonzero pair, the next residual equals the current residual times
`1 - beta * ||k||² * ||v||²`. Thus the one-step projection uses
`beta = 1 / (||k||² * ||v||²)`, not always `1`. A different pair's score
changes by the update coefficient times **both** its key overlap and value
overlap (equation 5). See the [primary equations](https://arxiv.org/html/2610.26834v1#S2.SS1).

**API choices:** no implicit normalization, no forgetting, and no learned gates.
`target` is any finite scalar and `beta` is finite and nonnegative. Steps
outside the contraction range are intentionally allowed for counterexamples.
Zero features or zero step sizes are no-ops. Inputs are copied into immutable
tuples, dimensions are checked, and nonfinite or overflowing arithmetic raises
`ValueError`. This finite-precision teaching code may reject an intermediate
overflow even when a differently arranged expression would remain finite.

## Reproduce

From the repository root with **CPython 3.14.5**, pinned by `.python-version`:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s examples/kronecker_delta -v
python -m examples.kronecker_delta.demo
python -m examples.kronecker_delta.demo > /tmp/kronecker-output.json
diff -u examples/kronecker_delta/sample_output.json /tmp/kronecker-output.json
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. No pip installation,
GPU, model weights, data downloads or paid APIs are needed. The demo has
**no randomness**. Randomized oracle tests use seed **20261030**. Checked JSON
was generated with CPython 3.14.5 and rounds floats to nine decimal places;
numerical tests additionally check unrounded values with explicit tolerances.

## Our experiment: preservation can be a feature or a bug

Start with zero memory. Write key `[1,0]` and value `[1,0]`. Then write the
**same key** with value `[0,1]`, using `target=beta=1` for Kronecker and
`beta=1` for the vector Delta control.

| After the second write | Kronecker | Vector Delta |
| --- | --- | --- |
| Readout at `[1,0]` | **[1,1]** | **[0,1]** |
| Old pair score | 1 | 0 |
| New pair score | 1 | 1 |
| Squared error if `[0,1]` is the desired replacement | **1** | **0** |

The first result is useful if we want both pair scores. It is wrong if the
second value is meant to replace the first. No unique old/new value can be
selected from the key alone in this fixture. Explicitly erasing the old scalar
association with `target=0` changes Kronecker's readout to `[0,1]` while
preserving the new score. Erasing a scalar association is not generally the
same as zeroing the entire vector readout.

Further controls in the [actual output](sample_output.json):

- Write `[0.6,0.8]` after the first-only memory: readout becomes **[1.24,0.32]**.
  The new score is 1, but the old score rises from 1 to **1.24**. Overlapping
  values still interfere.
- With one-dimensional key `[2]` and value `[3]`, `beta=1` produces score
  **36**, not the target 1. `beta=1/36` produces **1**.
- Repeating a unit pair with `beta=0.5` halves the residual each time:
  **−0.5, −0.25, −0.125, −0.0625**.

These are exact constructed algebraic controls, not measured language-model
performance or a representative retrieval benchmark.

## Tests and limits

The **31 tests** cover an independent 80-digit Decimal affine-transition oracle
on 48 seeded rectangular fixtures, finite-difference gradients, projection
cost, interference overlaps, scalar erasure, vector overwrite, contraction,
reflection/divergence controls, nonunit features, causal prefix replay,
immutability, invalid inputs, overflow, CLI behavior and exact sample bytes.
The root CI runs these alongside every existing example.

Each update and read is `O(d_v d_k)` work, with `O(d_v d_k)` memory state
independent of sequence length. The pure-function implementation allocates a
new matrix per update. Retaining a history of those matrices would add storage.
There are no trained projections, gates, normalizations, forget gate, output
network, chunkwise algorithm, backward pass or GPU kernels. In particular,
KronLSTM's **equation 10** has forgetting and decoupled erase/write gates; this
example does not implement it. We make no speed, memory-efficiency benchmark,
training-stability or LLM-quality claim. Single-pair contraction does not prove
a universal bound for arbitrary sequences of changing pairs and targets.

## Attribution and license

The cited paper is **CC BY 4.0**. This code, documentation, tests and synthetic
fixtures are original and covered by the repository's [MIT license](../../LICENSE).
No upstream code, paper figures, weights, real data or private material is
copied or redistributed. The citation identifies the source mechanism, not an
endorsement by the authors. The scalar update also has classical delta-rule
and projection-method antecedents, as discussed in the paper's Appendix A2.
