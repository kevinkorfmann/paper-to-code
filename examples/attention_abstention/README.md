# Attention abstention and per-read value gates

This original, deterministic example explores the distinction between reducing
an attention head's contribution and changing which values survive its read.
The source is Richard Zhe Wang's preprint
[*Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention*](https://arxiv.org/abs/2609.22005v2),
first posted **September 18, 2026**, with **v2 dated September 29, 2026**.
We implement the scalar value gate, retained-mass decomposition, phantom
denominator, and projection gate from equations 2–4 and 6 / Lemma 3.1.
This is a mechanism demonstration, not a reproduction of the paper's training
experiments or evidence of an improvement in language modeling.

The phantom mechanism predates this paper. Evan Miller's
[*Attention Is Off By One*](https://www.evanmiller.org/attention-is-off-by-one.html)
(**July 24, 2023**) proposed adding one to the denominator. Wang studies the
relationship between abstention and value filtering; we do not attribute the
invention of the phantom or projection gate to this preprint.

## Run

From the repository root, using **CPython 3.14.5**:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s examples/attention_abstention -v
python -m examples.attention_abstention.demo
python -m examples.attention_abstention.demo > /tmp/abstention-output.json
diff -u examples/attention_abstention/sample_output.json /tmp/abstention-output.json
```

The interpreter is pinned in [the root .python-version](../../.python-version).
[requirements.txt](../../requirements.txt) records the empty third-party
dependency set. The implementation reuses original stable numerical helpers
from this repository's earlier attention examples. No model, GPU, downloads,
training, or random seed is needed. The checked JSON rounds to nine decimal
places; tests compare unrounded arithmetic with explicit tolerances.

Try these controls:

```sh
# exp(0) adds exactly one to the softmax denominator.
python -m examples.attention_abstention.demo --sink-logit 0
# Change only the real scores; hold the phantom logit fixed.
python -m examples.attention_abstention.demo --real-shift -5
```

## What is calculated

[model.py](model.py) accepts one attention row's already scaled scores
`s_j = q dot k_j / sqrt(d)` and real value vectors `v_j`. With phantom logit
`beta`, it computes:

```text
D   = exp(beta) + sum_valid_j exp(s_j)
p_j = exp(s_j) / D                     # zero for masked real keys
p_0 = exp(beta) / D                    # phantom's value is fixed at zero
g_j = sigmoid(w dot v_j + bias)        # or supplied gates in [0, 1]
c_j = p_j * g_j
r   = sum_j c_j
y   = sum_j c_j * v_j

For r > 0:
q_j = c_j / r
z   = sum_j q_j * v_j
y   = r * z
```

`sink_logit=None` omits the phantom; omitted gates are one. In the returned
`Read`, `routing_weights` are the real-key probabilities `p`, `retained_mass`
is `r`, and `renormalized_output` is the counterfactual `z`. Renormalization
conditions on the surviving real mass and removes the factor `r`.

## Checked two-value experiment

We choose values `[2, 0]` and `[0, 2]`, equal real scores of zero, and phantom
logit `log(2)`. The deliberate projection `w = [log(9)/2, -log(9)/2]`, bias zero,
produces gates `[0.9, 0.1]`. No parameters are fitted. Actual default output is
saved in [sample_output.json](sample_output.json).

| Configuration | Phantom mass | Retained mass `r` | Output `y` | Renormalized output `z` |
| --- | ---: | ---: | --- | --- |
| Ordinary attention | 0 | 1 | `[1, 1]` | `[1, 1]` |
| Phantom only | 0.5 | 0.5 | `[0.5, 0.5]` | `[1, 1]` |
| Per-read gates only | 0 | 0.5 | `[0.9, 0.1]` | `[1.8, 0.2]` |
| Phantom and per-read gates | 0.5 | 0.25 | `[0.45, 0.05]` | `[1.8, 0.2]` |
| Uniform gates of 0.1, no phantom | 0 | 0.1 | `[0.1, 0.1]` | `[1, 1]` |

Multiplying the ordinary aggregate by the same scalar retained mass of 0.5
gives `[0.5, 0.5]`, whereas applying the gates to the individual reads gives
`[0.9, 0.1]`. This distinguishes the example from the earlier
[post-aggregation gated-attention example](../gated_attention/README.md).
An elementwise output gate could happen to match one chosen output; that does
not identify the underlying operation.

Two further controls are included. With both real scores at -8 and phantom
logit zero, retained mass is `0.000670475` and output is
`[0.000670475, 0.000670475]`; ordinary equal-score attention remains `[1, 1]`.
With opposing values `[1, 0]` and `[-1, 0]`, ordinary attention outputs zero
at **full retained mass of one**. Zero output alone therefore does not establish
abstention: cancellation is another possibility.

## Numerical and interpretation boundaries

Masks exclude real keys before normalization. `True` permits a key; the caller
can provide a causal mask. The phantom remains available, including when every
real key is masked. Without a phantom, an all-masked row raises `ValueError`.

Softmax subtracts the largest logit across real and phantom positions. Adding
the same constant to **all** logits, including `beta`, preserves the output.
Shifting only the real logits with a fixed phantom changes its relative mass,
even though conditional routing among real keys stays the same.

Finite scores and value entries are required. Supplied gates may include the
limiting values zero and one. With zero retained mass, output is zero and
conditional routing is undefined, returned as `None`. Binary64 can underflow
very small positive masses to zero; for example, real scores -1000 against a
phantom at 1000 do so. Subtracting a maximum prevents exponential overflow but
does not restore such lost precision. JSON rounding can also hide tiny mass.

The input is a single head and query. There are no learned Q/K/V projections,
output projection, residual block, training, or evaluation dataset. The chosen
gate sees a value's content, not the query; it is not a learned relevance or
noise detector. A scalar gate scales all coordinates of its read together.
The map is linear in values only when scores and gates are held fixed; computing
gates from the values changes that property. We do not implement the paper's
norm gate or other experimental variants. These controls support algebraic
claims on the specified inputs, not gains in perplexity, robustness, or quality.

## Verification

The **22 tests** in [test_demo.py](test_demo.py) cover an independent 60-digit
Decimal oracle, the appended-zero-value equivalence, the denominator-plus-one
case, retained-mass decomposition, all-logit shift invariance, fixed-phantom
sensitivity, masking, extreme scores, input validation, and the zero-mass case.
They also check scalar versus per-read gating, conditional renormalization,
permutation, linearity with fixed gates, finite-difference derivatives in the
phantom logit and a gate, the cancellation counterexample, the CLI, and the
checked sample output. The repository's push/PR CI runs this suite and reproduces
the JSON byte for byte alongside the existing examples.

All code, tests, and constructed examples here are original and covered by the
repository's [MIT license](../../LICENSE). No paper text, figures, model weights,
or upstream implementation is redistributed.
