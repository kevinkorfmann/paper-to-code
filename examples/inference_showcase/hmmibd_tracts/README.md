# hmmIBD: shared haploid ancestry is inferred, not counted

This original example calls **hmmIBD**, an actual hidden Markov model (HMM)
inference tool, on three synthetic haploid pairs. It estimates a posterior
IBD probability at each marker and a most probable sequence of IBD states.
Matching alleles alone are not proof of identity by descent: their frequency
and the evidence at neighboring markers matter.

The primary reference is Schaffner et al., **“hmmIBD: software to infer pairwise
identity by descent between haploid genotypes,”** *Malaria Journal* 17:196,
published **May 15, 2018** ([paper](https://doi.org/10.1186/s12936-018-2349-7),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5952413/)). This is a
small tool showcase, not a reproduction of that paper's validation study.

## Reproduce

From the repository root, use **CPython 3.14.5**, a C compiler (`cc`), and an
internet connection for the first dependency download. No Python packages,
GPU, real genomic data, or paid services are needed.

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m examples.inference_showcase.hmmibd_tracts.build
python -m unittest discover -s examples/inference_showcase/hmmibd_tracts -v
python -m examples.inference_showcase.hmmibd_tracts.demo --check-sample
```

The build fetches unmodified official source and its license at
[`c553843db8a7c070c50bdd96623e1f8280a8009d`](https://github.com/glipsnort/hmmIBD/tree/c553843db8a7c070c50bdd96623e1f8280a8009d)
(commit dated December 15, 2023). This exact revision, rather than an inferred
release number, is the tool version used here. The source SHA-256 is
`03d08b190ceedefa1e834ae1525c1cd5ff778b9818777ed8e1e4e3a58defd922`.
[build.py](build.py) verifies both downloads before compiling with
`cc -O3 -Wall … -lm`. They remain in ignored `.cache/hmmibd/`; the executable
is not redistributed. Local validation used Apple clang 15.0.0
(`clang-1500.3.9.4`); CI builds the same source with Ubuntu 24.04's `cc` and
checks the same output. The upstream unused-variable compiler warning is benign.

The wrapper uses temporary input/output files and short relative filenames
because this upstream revision has fixed-size filename buffers. It validates
its restricted biallelic inputs before calling the CLI and fails explicitly
if the tool skips a pair. This demonstration is intended for macOS/Linux.

## Inputs and estimands

[synthetic_pairs.json](synthetic_pairs.json) contains two haploid genotype
strings for each of three cases, with **1,200 markers spaced 1,000 bp apart**.
The generator uses Python `random.Random(20261028)`. At shared sites, both
samples inherit one Bernoulli(0.5) allele; at nonshared sites, each draws an
independent allele. Each observed allele is then flipped independently with
probability 0.001. Latent state blocks are constructed, not sampled from a
pedigree or a coalescent. They are stored separately for evaluation and never
passed to the inference tool:

- `blocks`: sites 400–799 (zero-based, inclusive) share a latent allele;
- `identical`: all sites share a latent allele before genotyping error;
- `independent`: all latent alleles are independent between samples.

Allele frequencies are supplied as 0.5 from the generating model; they are
not estimated from these two samples. All sites are on chromosome 1. The
wrapper runs `hmmIBD -i data.tsv -f freq.tsv -o result -m 50` separately for
each pair. The original error model (0.001) and uniform recombination rate
(7.4e-7 per bp per outcrossed meiosis, a *P. falciparum* default) are unchanged.
Increasing the iteration limit to 100 leaves the reported results unchanged.

The principal estimand is **the mean marginal posterior IBD probability
across input markers**, upstream `fract_sites_IBD`, under the fitted model.
The separate Viterbi fraction counts markers in the single most probable
state path (`fract_vit_sites_IBD`). State **0 means IBD**; state 1 means
non-IBD. Neither is a directly observed pedigree fraction or a generally
length-weighted estimate. Segment endpoints are observed marker positions;
the true breakpoint can lie in the unobserved gap between them.

## Executed results

[sample_output.json](sample_output.json) is the actual checked output:

| Synthetic case | Constructed shared-marker fraction | Posterior mean IBD | Viterbi marker fraction |
| --- | ---: | ---: | ---: |
| One central shared block | 400/1200 | 0.33172 | 0.33333 |
| Fully shared before error | 1 | 1.00000 | 1.00000 |
| Independent latent alleles | 0 | 0.00000 | 0.00000 |

For this seed, the central Viterbi tract contains exactly 400 markers, from
**400,001 to 799,001 bp**. These are outcomes of this constructed fixture,
not a general accuracy guarantee. The tool prints posterior fractions to
five decimal places: 0.00000 and 1.00000 do not establish absolute certainty.

A separate frequency control uses 12 matching allele-1 markers with extremely
large spacing, making transitions effectively independent. With fixed prior
IBD probability 0.5 and one decoding pass, changing the supplied allele-1
frequency from 0.5 to 0.1 changes the posterior mean from **0.66622 to 0.90760**.
The independent Bayes calculations are **0.666222074 and 0.907598322**. This
artificial spacing is a mathematical control, not a plausible chromosome
model. It illustrates that matching a rarer allele is stronger evidence
under the model; it does not prescribe an allele-frequency estimate.

## Independent checks

[demo.py](demo.py) independently sums over unobserved alleles to obtain
biallelic emission probabilities. For example, without genotyping error,
matching allele 1 has likelihood `f` under IBD and `f²` under independence,
so its Bayes factor is `1/f`. The transition model is expressed as
`P(d) = exp(-k*r*d) I + (1-exp(-k*r*d)) 1 πᵀ`.

For an interpretable conditional check, `-m 1 -r 0.5` decodes at the
source's initial `k=1` and a fixed IBD prior of 0.5. The independent scaled
sum-product calculation gives **0.331682313**, versus the CLI's **0.33168**.
The CLI updates `k` after that single pass and prints the updated value in
`N_generation`; it was not the value used to decode that pass. We therefore
do not use that column as an oracle parameter or report it as an ancestry
estimate. The fixed-parameter check validates decoding, not the EM optimizer.

The **26 tests** include exhaustive enumeration of all 4,096 hidden paths
for a 12-marker problem using 50-digit Decimal arithmetic, checking both
posterior mean and Viterbi path against the real tool. Other checks cover
emission normalization, transition stationarity and composition, positive
and negative controls, iteration-limit sensitivity, missing markers, allele
and sample relabeling, coordinate shifts, invalid inputs, dependency checksum
rejection, exact fixture regeneration, and byte-for-byte CLI/sample agreement.
Oracle-to-CLI comparisons allow 5.1e-6 for the CLI's five-decimal printing.

## Assumptions and limits

This is haploid, biallelic, one-population inference with supplied frequencies.
It assumes independent emissions conditional on ancestry states, uniform
recombination and error, no new mutations within IBD tracts, and a common
ancestry timescale in the transition model. Real linkage disequilibrium,
frequency uncertainty, mixtures of ancestry ages, selection, and spatially
varying error or recombination can invalidate these simplifications. All
markers are equally weighted here. The tiny controlled cases do not estimate
clinical risk, establish human relatedness, quantify population history, or
supply confidence intervals. No recovery rate or runtime benchmark is claimed.

## Source and license

See the official [CLI and output documentation](https://github.com/glipsnort/hmmIBD/blob/c553843db8a7c070c50bdd96623e1f8280a8009d/README.md)
and [pinned source](https://github.com/glipsnort/hmmIBD/blob/c553843db8a7c070c50bdd96623e1f8280a8009d/hmmIBD.c).
The external dependency is **GPL-3.0** ([upstream license](https://github.com/glipsnort/hmmIBD/blob/c553843db8a7c070c50bdd96623e1f8280a8009d/LICENSE));
this repository's MIT license does not relicense it. The Python wrapper,
independent mathematical checks, synthetic fixtures, and explanatory text are
original. No upstream implementation, empirical data, paper figure, or binary
is copied into the published example. The paper is CC BY 4.0.
