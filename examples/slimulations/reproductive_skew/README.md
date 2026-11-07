# Same census size, different founder-lineage diversity

Original synthetic one-generation SLiM experiment, added 2026-11-07.
A population can retain 100 individuals while a single founder supplies half
of its expected offspring. This changes ancestry concentration even without
any change in census size. This is a clonal demonstration, not a realistic fish
life history or a reproduction of a published sweepstakes model.

## Model and actual output

There are 100 diploid individuals and 512 independent populations per treatment.
At the end of tick 1 each individual receives its own unique neutral marker,
homozygous on its two chromosomes. SLiM's Wright–Fisher reproduction is set to
100% cloning. Tick 2 contains 100 offspring, each independently sampled from
founders according to the supplied reproductive weights. Marker copies identify
the founder of each clone; the model checks that counts remain paired and sum
to the population size. No new mutation, migration, sexual reproduction or
recombination occurs during the measured generation.

- Uniform: all 100 founders have equal weight.
- Skew: founder 0 has weight 99, each other founder has weight 1. Its probability
  of supplying each independently sampled offspring is 1/2; the other probabilities are 1/198.
- Single-founder control: only founder 0 has positive weight.

Weights are assigned externally, not caused by marker mutations. The markers
are neutral, but reproductive contributions are unequal by construction.
Founder 0 is fixed for reproducibility: randomizing its identity would leave
these symmetric lineage-diversity statistics unchanged. This does not model
heritable fitness differences, repeated winner selection or long-term evolution.

| Treatment | Mean lineage diversity | Exact expectation | Exact SE of mean | Mean founder richness |
|---|---:|---:|---:|---:|
| Uniform | 0.98014648 | 0.98010000 | 0.000061875 | 63.552734 |
| Skew | 0.73982422 | 0.74000000 | 0.002171424 | 40.117188 |
| Single | 0 | 0 | 0 | 1 |

Richness expectations are 63.396766, 40.332370 and 1. In the skew treatment the
observed mean fraction from founder 0 is 0.499805. All populations are retained.
The diversity means satisfy a fixed six-exact-standard-error diagnostic; this
is an implementation check, not a significance test or biological calibration.

## What is measured

For realized offspring counts C_i summing to N, the Simpson founder-lineage
diversity is D = 1 - sum((C_i/N)^2). It is the probability that two draws **with
replacement** from the offspring population have different founder labels.
It is **not** observed diploid heterozygosity (every individual is homozygous),
per-site nucleotide diversity, or a selection statistic.

Let p_i be parental sampling probabilities, s2 = sum(p_i^2) and
s3 = sum(p_i^3). Counts are multinomial, so

```text
E[D] = (1 - 1/N)(1 - s2)
Var(D) = ((N-1)/N)^2 * [2*s2*(1-s2) + 4*(N-2)*(s3-s2^2)]/[N*(N-1)]
E[number of surviving founders] = sum(1 - (1-p_i)^N).
```

The variance follows by counting disjoint versus overlapping pairs of offspring
in the unbiased pair-match statistic. Its standard error across R independent
populations is sqrt(Var(D)/R). The reciprocal 1/s2 is the **effective number of
contributors** for this one-generation clonal parent distribution: 100, 3.96 and
1. It is not a general diploid effective population size. Notice that 3.96
contributors does not mean only four lineages survive: many tiny families still
contribute, while one family dominates squared frequencies.

## Reproduce and test

Use CPython 3.14.5 (root `.python-version`) and SLiM 4.0.1 / Eidos 3.0.1 on PATH.
No Python packages are required. The existing `slimulations` CI job builds the
checksum-verified official SLiM revision, runs tests and checks saved output.
From the repository root:

```sh
python -m unittest discover -s examples/slimulations/reproductive_skew -v
python -m examples.slimulations.reproductive_skew.demo
python -m examples.slimulations.reproductive_skew.demo --check-sample
```

Seeds are 202611070 (uniform), 202611071 (skew), and 202611072 (single).
`sample_summary.json` records the actual output and versions. The simulator is
required; there is no Python replacement for simulation. Runtime is a few
seconds on the development CPU. Public functions accept N=2..200 and
1..1024 replicates; the checked experiment uses N=100 and R=512.

Twenty tests include exact rational enumeration of all ordered offspring draws
for N=2..5, matching the mean, variance and richness, plus deterministic
single-parent, common-weight-scaling, N=2 equivalence, replay and malformed-row
checks. The formulas and tests are original, not copied from an upstream package.

## Context and limits

Reproductive skew can make census size a poor description of genealogical
concentration. The broader sweepstakes literature includes sexual populations,
multiple generations and ecological mechanisms absent here. We demonstrate
only one conditional multinomial/clonal step, not a fitted multiple-merger
coalescent, an empirical Ne estimate or evidence for selection in nature.

- [Hedgecock & Pudovkin (2011), Sweepstakes Reproductive Success in Highly
  Fecund Marine Fish and Shellfish](https://doi.org/10.5343/bms.2010.1051): broader review.
- [Árnason et al. (2023), Sweepstakes reproductive success via pervasive and
  recurrent selective sweeps](https://elifesciences.org/articles/80781): primary
  genomic study and discussion of distinct random and selective mechanisms;
  not reproduced by this toy.
- [Haller & Messer (2023), SLiM 4](https://doi.org/10.1086/723601): simulator.

Original code and text use the repository MIT license. No upstream code, paper
figures or real genetic data are included; SLiM retains its own license.
