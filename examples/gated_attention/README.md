# Gated attention: a head can turn down its contribution

Daily example **2026-10-21**. An original, CPU-only implementation of the
headwise sigmoid gate at position G1 in Qiu et al.,
[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708v1),
first posted **May 10, 2025**. See equations 2 and 5 and the headwise variant
in section 2.2. This is one mechanism from the paper, not its full architecture
or training experiment. No upstream code or model weights are included.

## Mechanism

For a single head, using already projected queries, keys, and values:

```text
A   = softmax(Q Kᵀ / sqrt(d))
Y   = A V
g_i = sigmoid(x_i · w_gate)
Y'_i = g_i Y_i
```

The scalar gate for query i multiplies every channel of that head's output,
after attention and before an output projection. Its input is a supplied hidden
state x_i, not the key's hidden state. This implementation covers the paper's
headwise variant; it does not implement elementwise gates, multiple heads,
normalization, or learned projections.

Every ordinary attention row sums to one. The effective coefficients g_i A_i
sum to g_i, so the head can contribute a much smaller vector. There is no second
normalization. Applying gates to values before attention generally gives a
different result: that uses a gate for each key rather than each query.

## Run from the repository root

Use the root's pinned **CPython 3.14.5**, with no third-party packages:

```sh
python3.14 -m venv --without-pip .venv
.venv/bin/python -m examples.gated_attention.demo
.venv/bin/python -m unittest discover -s examples/gated_attention -v
.venv/bin/python -m examples.gated_attention.demo --gate-logit -4
```

On Windows use `.venv\Scripts\python.exe`. There is no RNG or seed: inputs and
gate weights are deliberately constructed. All three queries have the same
attention map `[0.8, 0.1, 0.1]` and ordinary output `[10, 0.8]`.

| Gate | Gated output | Sum of effective coefficients |
| ---: | --- | ---: |
| 0.1 | `[1, 0.08]` | 0.1 |
| 0.5 | `[5, 0.4]` | 0.5 |
| 0.9 | `[9, 0.72]` | 0.9 |

These are the actual results in [sample_output.json](sample_output.json):

```sh
.venv/bin/python -m examples.gated_attention.demo > /tmp/gated-output.json
diff -u examples/gated_attention/sample_output.json /tmp/gated-output.json
```

## Checks and limits

Twelve tests include an independent 60-digit Decimal oracle, query dependence,
causal masking, zero projection, saturation, a finite-difference gate derivative,
the distinction between output gating and value gating, invalid inputs, CLI,
and checked output. Stable sigmoid avoids exponent overflow. The existing
repository's original softmax and validation utilities are reused.

The fixed map still puts **80% on token 0** after gating. Both output channels
shrink together; this scalar gate cannot separate signal from background within
one head. Learned models could change their maps during training, which this
example does not simulate. These toy outputs establish no attention-sink
removal, language-model improvement, sparse execution, or speedup.
