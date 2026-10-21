# Migration erodes divergence between two demes

Daily SLiMulation **2026-10-21**. This original Eidos script runs real
**SLiM 4.0.1**. It illustrates gene flow between populations, a different
mechanism from the first example's isolated neutral drift.

## Model and expectation

Each of 256 independent replicates contains two demes of 50 diploid individuals.
A neutral marker starts fixed in deme 1 and absent in deme 2: p1=1, p2=0.
Reproduction follows SLiM's Wright–Fisher model, with equal constant deme sizes,
no new mutation, no recombination, and independent random parent choices within
the selected source (incidental selfing allowed).

In SLiM WF models, `a.setMigrationRates(b, m)` means that each offspring of a
chooses deme b as its parental source with probability m. **Both parents**
then come from that source. It does not move a fixed number of adults. See the
[official 4.0.1 manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf),
sections 5.2.1 and 22.2.1. The model therefore has correlated source choices for
the two gene copies in an offspring; we do not assume independent migration
of all 2N copies when predicting variances.

The conditional means are nevertheless exactly:

```text
E[p1(t+1) | p1,p2] = (1-m) p1 + m p2
E[p2(t+1) | p1,p2] = (1-m) p2 + m p1
E[p1(t)-p2(t)]     = (1-2m)^t
E[(p1(t)+p2(t))/2] = 0.5
```

Subtract the first two equations and iterate to obtain the third. Adding them
gives the conserved expected pooled frequency. This is a statement about
ensemble means of **signed differences**, not absolute differences or F_ST.

## Run from the repository root

Use **CPython 3.14.5** and **SLiM 4.0.1**. No Python dependencies are required.
Installation and the checksum-verified source build are documented in the
[first SLiM example](../neutral_drift/README.md) and the
[CI workflow](../../../.github/workflows/tests.yml).

```sh
slim -v
.venv/bin/python -m examples.slimulations.migration.demo
.venv/bin/python -m unittest discover -s examples/slimulations/migration -v
# Change run length, number of replicates, or the base RNG seed:
.venv/bin/python -m examples.slimulations.migration.demo --generations 40 --replicates 512 --seed 401
# Or run the Eidos model directly:
slim -s 20261022 -d MIGRATION=0.05 examples/slimulations/migration/migration.slim
```

Create `.venv` as in the root README if necessary. The wrapper also accepts
`--slim /path/to/slim` and refuses other SLiM versions. Seed **20261021** is used
for the m=0 control and **20261022** for m=0.05; these are seeds, not run dates.
There are no connections between replicate pairs. Each pair has a separate
marker. Tick 1 late initializes standing variation after reproduction, and is
recorded as t=0; tick 21 is t=20. Fixed markers remain countable.

## Checked result

| Treatment at t=20 | Mean p1−p2 | Expected p1−p2 | Empirical standard error |
| --- | ---: | ---: | ---: |
| m=0 | 1.000000 | 1.000000 | 0 |
| m=0.05 | 0.106094 | 0.121577 | 0.009723 |

The migration treatment's pooled mean is **0.498594**, against expectation 0.5.
Individual signed differences range from **−0.27 to 0.61**: drift can reverse
which deme has the higher frequency. No-migration demes remain at their
opposite fixation states because there is no new mutation.

[sample_summary.json](sample_summary.json) contains actual output:

```sh
.venv/bin/python -m examples.slimulations.migration.demo > /tmp/migration-summary.json
diff -u examples/slimulations/migration/sample_summary.json /tmp/migration-summary.json
```

Eleven tests check the analytical recursion independently, zero elapsed time,
exact m=0 and m=1 boundaries (complete migration swaps the fixed demes),
replicate statistics, seed behavior, invalid inputs, CLI, and sample output.
The joint probabilistic check uses Hoeffding's bound for all 20 reported mean
comparisons at alpha=10^-6. Signed differences lie in [−1,1]; pooled frequency
has half that range. The bound is deliberately conservative, not a fitted
confidence interval. The fixed-seed regression checks provide a more sensitive
check of unintended implementation changes.

## Limitations and sources

No selection, unequal sizes, asymmetric migration, population growth, linked
loci, or observation error is modeled. Signed differences approach zero in
expectation; individual trajectories need not converge monotonically, and
finite populations can retain divergence. This is not an estimate of real
migration rates or a claim about genome-wide differentiation.

The simulator is described by Haller and Messer (2023),
[SLiM 4: Multispecies eco-evolutionary modeling](https://doi.org/10.1086/723601).
The [official source](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216)
is pinned in CI. This repository contains only the original experiment and
its outputs, not redistributed simulator code.
