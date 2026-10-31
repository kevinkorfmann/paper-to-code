# Dual DeltaNet: correct values, then reorganize their addresses

**Qian et al., “Dual DeltaNet: Value-Side Memory Reorganization for Linear
Attention,” first posted October 28, 2026**, introduces a value-side correction
alongside a key-side delta update. This original educational implementation
covers the recurrent equations and their rank-two algebraic form:
[primary preprint v1](https://arxiv.org/abs/2610.38578v1),
[equations 5–10 and Algorithm 1](https://arxiv.org/html/2610.38578v1#S3).
It complements the repository's Kronecker example: here the extra correction
changes a selected value feature's distribution across addresses.

## Run

From the repository root, with **CPython 3.14.5**:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/dual_deltanet -v
python -m examples.dual_deltanet.demo --check-sample
```

There are no third-party packages, model weights, downloads, random inputs,
or seeds. [The root interpreter pin](../../.python-version) and
[empty dependency list](../../requirements.txt) apply. The checked output is
[sample_output.json](sample_output.json); the CLI compares it byte for byte.
Floating-point results are rounded to 12 decimal places for display.

## Operator

The state `S` has `d_key` rows and `d_value` columns. Rows are addresses.
Inputs supply key `k`, value `v`, row decay `alpha`, channel gate `m`, and
scalar gates `beta,gamma`. The key and query are L2-normalized. Let

```text
Sbar = diag(alpha) S
vhat = v / ||v||
e    = (m * vhat) / ||m * vhat||       # * is elementwise here
 tau = v^T e
K    = Sbar + beta k (v - Sbar^T k)^T
Snew = K + gamma (tau k - K e) vhat^T
out  = Snew^T normalized(query) / sqrt(d_key)
```

The order matters: the second correction reads **K after the key update**.
Its read direction `e` and write direction `vhat` are generally different.
[model.py](model.py) implements both that sequence and Algorithm 1's rank-two
form, whose two reads use `Sbar` plus a cross-term correction.

The paper obtains gates from learned projections. Here gates and projections
are supplied, with no training. The API permits endpoints 0 and 1 as explicit
limiting controls; sigmoid gates in exact arithmetic are strictly interior.
Zero values, zero keys/queries, and zero selected directions are rejected
because their normalized equations are undefined. We do not substitute an
undocumented zero-vector convention. Finite inputs can still exceed float
range in arithmetic; overflow is reported rather than silently accepted.

## Executed toy and counterexample

Start with this state, key `[0,1,0]`, and value `[1,0]`:

```text
S = [[1,2],
     [0,0],
     [1,3]]
```

Use unit decay, channel gates `[0.5,0.5]`, and `beta=gamma=0.5`.
The key-only update and two-sided update produce:

```text
K    = [[1,2],        Snew = [[0.5,2],
        [0.5,0],             [0.75,0],
        [1,3]]               [0.5,3]]
```

The first value feature's address response moves from **[1,0.5,1]** toward
target **[0,1,0]**, reaching **[0.5,0.75,0.5]**. Its squared target error falls
from **2.25 to 0.5625**. The orthogonal second value channel stays `[2,0,3]`.
Sequential and factorized implementations produce the same checked state.

This is not automatically beneficial. Suppose we want to preserve the old
association at address 0. With normalized query `[1,0,0]` and the paper's
`1/sqrt(3)` readout scale, key-only output still equals the desired
`[0.577350269190,1.154700538379]`. The dual correction changes it to
`[0.288675134595,1.154700538379]`: squared error rises from **0 to 1/12**.
The toy improves the chosen address target while damaging another target.
It is not evidence of better learned retrieval.

A second control uses `v=[3,4]` and `m=[0.8,0.2]`. Then
`vhat=[0.6,0.8]` but `e=[0.948683298051,0.316227766017]`. Even at `gamma=1`,
the address residual does not vanish: its multiplier is
**0.177807808356**, and its remaining squared error is **0.328802413809**
for the supplied state. Gamma one is not generally a full overwrite.

To see why, define `r=tau*k-K*e`. Directly expanding the implemented update
shows that the new residual equals

```text
r_new = (1 - gamma * dot(vhat,e)) r.
```

When `e=vhat`, this becomes `(1-gamma)r`; otherwise it is an oblique
correction. Likewise, projections onto any value direction orthogonal to
`vhat` are unchanged by this second step. These identities are independently
tested; the associated toy numbers are executed results.

## Validation and limits

The **32 tests** compare both implementations against an independent
**80-digit Decimal dense-matrix oracle** using the factorization
`A Sbar B + eta k vhat^T`. Controls cover no correction, row decay, full
key correction, aligned and oblique selectors, orthogonal projection
preservation, the retention counterexample, rank-two updates, scaling and
permutation invariants, input immutability, extreme normalization, invalid
inputs, causal prefixes, sequence order, CLI behavior, and checked output.
Oracle comparisons use a 2e-12 scaled absolute tolerance.

This is a recurrence demonstration, not the full DDN block. It omits learned
projections, convolutions, activations, gating networks, chunkwise training,
backward passes, GPU kernels, and language-model experiments. Each step costs
O(d_key*d_value) arithmetic with a fixed-size recurrent matrix; `run()` also
stores an output list growing with sequence length. The Python factorization
does not demonstrate hardware speed, memory-traffic reductions, or real-model
gains. The incoming synthetic state is deliberately constructed rather than
presented as a trained or naturally occurring memory.

The preprint is distributed under the arXiv non-exclusive distribution
license, not a Creative Commons license. Our Python implementation, tests,
fixtures, and prose are original and covered by the repository's MIT license.
No upstream source, paper figures, datasets, or weights are copied. The
paper builds on earlier delta-rule and gated linear-attention work; this
example does not attribute their invention to DDN.
