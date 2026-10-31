# Paper to Code

[![Tests](https://github.com/kevinkorfmann/paper-to-code/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinkorfmann/paper-to-code/actions/workflows/tests.yml)

Small, reproducible examples of ML mechanisms, population-genetic phenomena,
and inference tools. Each example states its assumptions, tests its behavior,
and explains its limits.

| Date | Category | Example | Reference environment |
| --- | --- | --- | --- |
| 2026-10-31 | Inference showcase | [Beagle imputation and hidden-reference sensitivity](examples/inference_showcase/beagle_imputation/README.md) | CPython 3.14.5 + checksum-pinned Beagle 5.5 / Temurin 17.0.21+9 |
| 2026-10-31 | Paper to code | [Dual DeltaNet value-side reorganization](examples/dual_deltanet/README.md) | CPython 3.14.5; standard library |
| 2026-10-30 | **slimulations** | [Reversible mutation restores variation after fixation](examples/slimulations/reversible_mutation/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-30 | Inference showcase | [ADMIXTURE projection and reference sensitivity](examples/inference_showcase/admixture_projection/README.md) | CPython 3.14.5 + checksum-pinned ADMIXTURE 1.4.0 |
| 2026-10-30 | Paper to code | [Kronecker Delta Rule: pair scores versus vector overwrite](examples/kronecker_delta/README.md) | CPython 3.14.5; standard library |
| 2026-10-29 | **slimulations** | [Transmission distortion and heterozygous parents](examples/slimulations/transmission_distortion/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-29 | Paper to code | [MERLIN mean-and-variance routing](examples/merlin_routing/README.md) | CPython 3.14.5; standard library |
| 2026-10-29 | Inference showcase | [KING-robust kinship and a mixed-parent counterexample](examples/inference_showcase/plink_king/README.md) | CPython 3.14.5 + checksum-pinned PLINK 2 binary |
| 2026-10-28 | **slimulations** | [Recessive deleterious alleles and masking](examples/slimulations/recessive_selection/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-28 | Inference showcase | [Haploid IBD tracts with hmmIBD](examples/inference_showcase/hmmibd_tracts/README.md) | CPython 3.14.5 + checksum-pinned C source |
| 2026-10-28 | Paper to code | [Continuum Attention over interpolated memory](examples/continuum_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-27 | Paper to code | [OpenAttention residual tail proxies](examples/tail_proxy_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-27 | Inference showcase | [Conditional migration inference with moments](examples/inference_showcase/moments_migration/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-27 | **slimulations** | [Wahlund effect: pooling samples versus parents](examples/slimulations/wahlund_effect/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-26 | Paper to code | [First Passage Attention: selective bias from path minima](examples/first_passage_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-26 | **slimulations** | [Recombination erodes linkage disequilibrium](examples/slimulations/linkage_disequilibrium/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-26 | Inference showcase | [Hudson F_ST and aggregation sensitivity with scikit-allel](examples/inference_showcase/scikit_allel_fst/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-25 | Paper to code | [Attention abstention and per-read value gates](examples/attention_abstention/README.md) | CPython 3.14.5; standard library |
| 2026-10-25 | **slimulations** | [Heterozygote advantage and finite-population loss](examples/slimulations/heterozygote_advantage/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-25 | Inference showcase | [fastDFE and target-site sensitivity](examples/inference_showcase/fastdfe_gamma/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-23 | Paper to code | [Multi-head attention residuals](examples/multi_head_attention_residuals/README.md) | CPython 3.14.5; standard library |
| 2026-10-23 | **slimulations** | [Selfing removes heterozygotes](examples/slimulations/selfing/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-23 | Inference showcase | [Conditional size-change inference with dadi](examples/inference_showcase/dadi_size_change/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-21 | Paper to code | [Gated attention](examples/gated_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-21 | **slimulations** | [Migration erodes divergence](examples/slimulations/migration/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-21 | Inference showcase | [Mutation-clock dating with tsdate](examples/inference_showcase/tsdate_clock/README.md) | CPython 3.12.11; separate hash-locked environment |
| 2026-10-20 | Paper to code | [Differential attention](examples/differential_attention/README.md) | CPython 3.14.5; standard library |
| 2026-10-20 | **slimulations** | [Neutral drift and loss of diversity](examples/slimulations/neutral_drift/README.md) | SLiM 4.0.1 + CPython 3.14.5 |
| 2026-10-20 | Inference showcase | [Genealogy inference with tsinfer](examples/inference_showcase/tsinfer_genealogy/README.md) | CPython 3.12.11; separate hash-locked environment |

The examples use only constructed inputs or synthetic data. Their documentation
includes commands, seeds where applicable, checked output, and tests. The ten
inference examples call actual tools: tsinfer estimates topology; tsdate estimates
node ages conditional on a supplied topology and mutation clock; dadi estimates
a population-size ratio conditional on a supplied change time; fastDFE estimates
a deleterious gamma DFE conditional on model assumptions and target-site counts;
scikit-allel estimates pairwise Hudson differentiation for a specified locus panel;
moments estimates symmetric migration conditional on a supplied split time;
hmmIBD infers haploid IBD states and marker-averaged posterior probabilities;
PLINK 2 estimates diploid KING-robust kinship, with an explicit ancestry counterexample;
ADMIXTURE estimates global mixture coefficients conditional on fixed source profiles,
with reference-sensitivity and nonidentifiability controls; Beagle estimates
untyped-marker dosages conditional on a supplied phased reference panel,
with hidden-allele and independent copying controls.

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

[CI](.github/workflows/tests.yml) runs all **629 tests**: 25 differential-attention,
12 gated-attention, 18 multi-head attention-residual, 22 attention-abstention, 28 first-passage-attention, 30 tail-proxy-attention, 26 continuum-attention, 36 MERLIN-routing, 31 Kronecker-delta, 32 Dual-DeltaNet, 11 neutral-drift,
11 migration, 12 selfing, 17 heterozygote-advantage, 20 linkage-disequilibrium, 20 Wahlund-effect, 22 recessive-selection, 22 transmission-distortion, 23 reversible-mutation, 11 tsinfer, 12 tsdate, 15 dadi,
18 fastDFE, 22 scikit-allel, 22 moments, 26 hmmIBD, 25 PLINK/KING, 28 ADMIXTURE, and 32 Beagle tests.
The SLiM job builds a checksum-verified official
source revision; hmmIBD similarly builds checksum-verified C source, PLINK/KING
uses a checksum-pinned official binary, ADMIXTURE does likewise, Beagle uses a
checksum-pinned official JAR and isolated Java runtime, and the other inference jobs install
their separate hash-locked package sets.
All jobs run on pushes and pull requests using a
standard Ubuntu runner, a pinned Python version, official actions pinned by
commit hash, and read-only repository permissions. It checks the sample output
byte for byte except for tsdate's, dadi's, and fastDFE's documented numerical tolerances. Standard GitHub-hosted runners are
[free for public repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
There is no scheduled workflow, artifact upload, or paid service.

The implementation and explanatory material are original and
[MIT licensed](LICENSE). Synthetic fixtures are included for reproducibility.
Separately installed dependencies retain their own licenses; fastDFE, for
example, is GPL-3.0-or-later, as documented in its example.
No upstream code, paper figures, model weights, private research, or real-world
genetic data are redistributed. Citations identify the source idea,
not an endorsement by the paper's authors.
