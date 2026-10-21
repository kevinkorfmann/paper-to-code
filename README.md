# Paper to Code

[![Tests](https://github.com/kevinkorfmann/paper-to-code/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinkorfmann/paper-to-code/actions/workflows/tests.yml)

Small, reproducible examples of ML mechanisms, population-genetic phenomena,
and inference tools. Each example states its assumptions, tests its behavior,
and explains its limits.

| Date | Category | Example | Reference environment |
| --- | --- | --- | --- |
| 2026-10-21 | Paper to code | [Gated attention](examples/gated_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-21 | **slimulations** | [Migration erodes divergence](examples/slimulations/migration/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-21 | Inference showcase | [Mutation-clock dating with tsdate](examples/inference_showcase/tsdate_clock/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-20 | Paper to code | [Differential attention](examples/differential_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-20 | **slimulations** | [Neutral drift and loss of diversity](examples/slimulations/neutral_drift/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-20 | Inference showcase | [Genealogy inference with tsinfer](examples/inference_showcase/tsinfer_genealogy/README.md) | CPython 3.12.11; separate hash-locked environment |

The examples use only constructed inputs or synthetic data. Their documentation
includes commands, seeds where applicable, checked output, and tests. The two
inference examples call actual tools: tsinfer estimates topology; tsdate estimates
node ages conditional on a supplied topology and mutation clock.

## First example: differential attention

[Differential Transformer](https://arxiv.org/abs/2410.05258v2), by Ye et al.,
was first posted on **October 7, 2024**, revised on **April 7, 2025**, and
accepted as an **ICLR 2025 oral**. This example implements the paper's
single-head attention equation:

```text
A1 = softmax(Q1 K1ᵀ / sqrt(d))
A2 = softmax(Q2 K2ᵀ / sqrt(d))
Y  = (A1 - λ A2) V
```

Subtracting two attention maps can cancel a component they have in common.
Unlike an ordinary attention map, the resulting coefficients can be negative
and sum to `1 - λ`. This is a mechanism demonstration, **not a trained LLM,
full-paper reproduction, or performance benchmark**.

Read the [walkthrough, equations, and limitations](examples/differential_attention/README.md),
inspect the [implementation](examples/differential_attention/model.py), or
run the four-token experiment below.

## Run the differential-attention example

With **CPython 3.14.5** installed:

```sh
git clone https://github.com/kevinkorfmann/paper-to-code.git
cd paper-to-code
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s tests -v
python -m examples.differential_attention.demo
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The test
and demo commands are otherwise the same.

No GPU, model weights, data downloads, paid API, or third-party Python packages
are needed for differential attention. Its interpreter is pinned in [.python-version](.python-version);
[requirements.txt](requirements.txt) records the empty dependency set.
The demo is deterministic and does not use randomness. Output is rounded to
nine decimal places; numerical tests use explicit tolerances.

Try changing the subtraction coefficient:

```sh
python -m examples.differential_attention.demo --lambda-value 0
python -m examples.differential_attention.demo --lambda-value 0.5
python -m examples.differential_attention.demo --lambda-value 1 --background 20
```

## What the checked example shows

The final query attends to four tokens. We deliberately construct
`A1 = [0.4, 0.2, 0.2, 0.2]` and `A2 = [0.1, 0.3, 0.3, 0.3]`.
Each value has two channels: a constant background of 10 and a marker that is
1 only at token 0.

| λ | Output background | Output marker |
| ---: | ---: | ---: |
| 0: ordinary first-branch attention | 10 | 0.4 |
| 0.5 | 5 | 0.35 |
| 1 | 0 | 0.3 |

At λ=1, the constant component cancels while a contrast remains. The same
demo includes two counterexamples: identical maps cancel **all** output,
including useful signal; a varying background leaves a residual. These are
algebraic results on chosen inputs, not evidence of better language modeling,
retrieval, hallucination rates, or automatic noise removal.

[sample_output.json](examples/differential_attention/sample_output.json) is
the actual default output, checked in tests. To compare it byte for byte:

```sh
python -m examples.differential_attention.demo > /tmp/attention-output.json
diff -u examples/differential_attention/sample_output.json /tmp/attention-output.json
```

## Verification

The 25 differential-attention tests include an independent 60-digit Decimal reference, the `sqrt(d)`
scale, causal masking in both branches, signed row sums, λ=0 reduction,
shared-component cancellation, counterexamples, permutation equivariance,
linearity in values, a finite-difference λ derivative, stable softmax, invalid
inputs, CLI behavior, and the checked sample output.

[CI](.github/workflows/tests.yml) runs all **82 tests**: 25 differential-attention,
12 gated-attention, 11 neutral-drift, 11 migration, 11 tsinfer, and 12 tsdate tests.
The SLiM job builds a checksum-verified official
source revision; the inference job installs the hash-locked package set.
All jobs run on pushes and pull requests using a
standard Ubuntu runner, a pinned Python version, official actions pinned by
commit hash, and read-only repository permissions. It checks the sample output
byte for byte except for tsdate's documented numerical tolerance. Standard GitHub-hosted runners are
[free for public repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
There is no scheduled workflow, artifact upload, or paid service.

The implementation and explanatory material are original and
[MIT licensed](LICENSE). Synthetic fixtures are included for reproducibility.
No upstream code, paper figures, model weights, private research, or real-world
genetic data are redistributed. Citations identify the source idea,
not an endorsement by the paper's authors.
