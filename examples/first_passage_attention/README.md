# First Passage Attention: a dip can suppress middle keys

Daily paper-to-code example **2026-10-26**. Original educational forward-pass
implementation of Yuxuan Liu's [First Passage Attention](https://arxiv.org/abs/2610.22714v1),
first posted **October 15, 2026, 22:31:45 UTC (v1)**. The mechanism follows
[equation 1 and the stack specialization in Section 2](https://arxiv.org/html/2610.22714v1#S2).
This is a recent preprint, not a claim of independent validation of its experiments.

## Mechanism

Each key has a scalar level `psi[s]`; every position has a scalar field `phi[u]`.
For a causal query at `t`, only **subsequent** fields after key `s` enter its bias:

```text
b[t,s] = beta * min(0, min(phi[s+1:t+1]) - psi[s])   for s < t
b[t,t] = 0
logit[t,s] = dot(q[t], k[s]) / sqrt(d) + b[t,s]
weights[t] = softmax(logit[t,0:t+1])
output[t] = sum_s weights[t,s] * v[s]
```

`beta` is strictly positive. In the stack specialization, set `phi=psi=W`,
where `W` is the prefix sum of supplied increments. The deepest later dip
below a key's level determines its penalty. A negative bias cannot recover
when the field rises again. Finite strength attenuates rather than hard-masks
the affected key. A repeated dip preserves that key's bias, **not necessarily
its normalized weight**: extra keys can change the softmax denominator.

## Our constructed rollback example

Nothing is learned here. We supply this six-position walk:

| Position | Token label | Walk / field / level | Content logit at final query | Final bias, beta=4 |
| ---: | --- | ---: | ---: | ---: |
| 0 | anchor | 1 | 2 | 0 |
| 1 | draft | 2 | 2 | -4 |
| 2 | detail | 3 | 2 | -8 |
| 3 | rollback | 1 | -2 | 0 |
| 4 | replacement | 2 | 2 | 0 |
| 5 | query | 2 | -2 | 0 |

The rollback field reaches 1, below the draft and detail levels but not below
the anchor's level. Replacement arrives afterward, so the earlier dip does
not penalize it. Four payload values are one-hot vectors; control values are
zero. Their output coordinates therefore expose the corresponding weights.
The ordinary-attention control uses the same Q/K/V and a constant walk.

Actual checked default output, rounded to nine decimal places:

| Payload | Ordinary causal weight | First-passage weight |
| --- | ---: | ---: |
| anchor | 0.247731321 | 0.486551147 |
| draft | 0.247731321 | 0.008911495 |
| detail | 0.247731321 | 0.000163220 |
| replacement | 0.247731321 | 0.486551147 |

Draft plus detail mass changes from **0.495462643** to **0.009074715**.
The remaining mass belongs to the two zero-valued controls. The selected
walk creates this behavior by construction; the code has not learned what
a draft, a rollback, or an obsolete fact means. Stronger penalties do not
necessarily produce more useful outputs on arbitrary tasks.

## Run from the repository root

Use the root's pinned **CPython 3.14.5** and empty Python dependency set:

```sh
python3.14 -m venv --without-pip .venv
.venv/bin/python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
.venv/bin/python -m examples.first_passage_attention.demo
.venv/bin/python -m unittest discover -s examples/first_passage_attention -v
.venv/bin/python -m examples.first_passage_attention.demo --beta 1
.venv/bin/python -m examples.first_passage_attention.demo --beta 8
```

On Windows use `.venv\Scripts\python.exe`. No GPU, downloads, model weights,
paid service, or third-party Python package is required. The demo has no
randomness. Two randomized test cases use fixed local seeds (`261022714`
and `42`); they do not train anything.

[sample_output.json](sample_output.json) is actual captured default output.
Both tests and CI check it byte for byte:

```sh
.venv/bin/python -m examples.first_passage_attention.demo > /tmp/fpa-output.json
diff -u examples/first_passage_attention/sample_output.json /tmp/fpa-output.json
```

## Implementation and verification

[`model.py`](model.py) exposes `first_passage_bias`, `causal_attention`,
`walk_from_increments`, and `IncrementalAttention`. Full attention returns
causal prefix rows of lengths `1,2,...,T`; there are no future entries to mask
after normalization. It computes range minima with a reverse scan for each
query. The decoder instead updates a stored minimum for every existing key
when the new field arrives, then appends a new key with zero diagonal bias.
Both paths retain every key and produce identical default-demo outputs.

The **28 tests** include an independent 60-digit Decimal oracle that directly
slices each field range and computes exponentials without production helpers;
full/incremental agreement; strict range boundaries; causal prefixes;
constant-walk reduction; expiry and monotone-walk forget-gate controls;
duration invariance of bias versus changing normalized weights; permanence
of negative biases; field/level translation; content-logit translation;
`sqrt(d)` scaling; linearity in values; probability normalization; finite
attenuation; numerical extremes; input validation; nonmutation; transactional
decoder failures; copied cache inputs; CLI behavior; and the saved sample.
The repository's original finite-value, matrix, dot-product and softmax helpers
are reused. Numerical comparisons use explicit tolerances, usually 12 places.

## Limits and provenance

- Q/K/V, fields, levels, and strength are supplied; there are no learned
  projections, gradients, optimizer, language model, or evaluation dataset.
  The paper's training procedure and full architecture are not implemented.
- With sequence length T, key width d and value width d_v, this full Python
  path costs `O(T^2*(d+d_v))` time and returns quadratic bias/weight storage.
  The decoder costs `O(t*(d+d_v))` per step, retaining a growing KV cache and
  two scalars per key. It is not a linear-time model or a fused GPU kernel.
- No keys are evicted. Negative bias alone is not a safe finite-strength
  eviction guarantee: content logits and other keys still determine weights.
- Binary64 exponentials can underflow to zero at extreme penalties; that is
  numerical underflow, not an explicit deletion rule. Nonfinite inputs and
  unrepresentable intermediate arithmetic are rejected. Accumulated walks
  remain subject to ordinary floating-point roundoff.
- The toy establishes an algebraic behavior on chosen inputs, **not** learned
  selective retention, retrieval accuracy, speed, memory savings, or real-LLM
  gains. It is not a full-paper reproduction.
- Code, fixture and explanations are original and covered by the repository's
  [MIT license](../../LICENSE). No upstream implementation, paper figure,
  model weights, private research, credentials, or personal data are included.
  The cited paper supplies the mechanism, not an endorsement of this example.
