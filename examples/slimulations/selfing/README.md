# Selfing removes heterozygotes without selection

Daily SLiMulation **2026-10-23**. This original experiment counts actual
heterozygous individuals under complete self-fertilization and random mating.
It illustrates mating-system effects, distinct from the earlier isolated-drift
and migration examples. All data are synthetic.

## Model and exact expectations

Each of 256 replicate populations has 100 diploid hermaphrodites. Every
individual starts as Aa at one neutral locus. Thus p=0.5, observed
heterozygosity H_obs=1, and gene diversity 2p(1-p)=0.5. This deliberately
constructed initial state is not Hardy–Weinberg equilibrium.

Complete selfing chooses one parent and makes two independent meiotic gametes.
An Aa parent gives AA, Aa, aa with probabilities 1/4, 1/2, 1/4; homozygotes
remain homozygous. Therefore, exactly in ensemble expectation:

```text
E[H_obs(t+1) | population(t)] = H_obs(t)/2
E[H_obs(t)] = 2^(-t), starting with all Aa.
```

The control independently samples two parents, permitting incidental selfing.
This matches gene-copy Wright–Fisher sampling, rather than enforced outcrossing.
For this control, t=0 has H_obs=1, and for t>=1:

```text
E[H_obs(t)] = 0.5 * (1 - 1/(2N))^(t-1).
```

The first random-mating generation has expected H_obs=0.5. Each later
generation inherits the preceding generation's gene diversity, which decays
by the finite-population factor 1-1/(2N). Both treatments retain E[p_t]=0.5,
but realized populations drift. These are expectations, not claims about every
trajectory or exact equality of finite replicate means.

There is no migration, selection, new mutation, or recombination. Fitnesses
are equal: this does **not** simulate inbreeding depression. Self-fertilization
is sexual segregation, not cloning. Sources: Wright (1921),
[Systems of Mating II](https://doi.org/10.1093/genetics/6.2.124), for the classical
inbreeding context; the recursions above follow directly from the stated
Mendelian and random-sampling assumptions.

## Reference environment and run

**SLiM 4.0.1 / Eidos 3.0.1**, **CPython 3.14.5**, standard-library wrapper.
The [official version-matched manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf),
section 6.3.1 and the `setSelfingRate` reference, specifies the explicit selfing
fraction and incidental selfing. The simulator reference is Haller and Messer
(2023), [SLiM 4](https://doi.org/10.1086/723601).

From the repository root, using the root Python environment:

```sh
slim -v
.venv/bin/python -m examples.slimulations.selfing.demo
.venv/bin/python -m unittest discover -s examples/slimulations/selfing -v
.venv/bin/python -m examples.slimulations.selfing.demo --generations 12 --replicates 512
slim -s 20261024 -d SELFING=1 examples/slimulations/selfing/selfing.slim
```

The root README explains `.venv` creation. The
[first SLiM example](../neutral_drift/README.md) and
[CI](../../../.github/workflows/tests.yml) document the checksum-verified
official source build. `--slim /path/to/slim` selects a binary; other versions
are rejected. The wrapper accepts `--seed`, `--generations` and `--replicates`.

The default seeds are **20261023** for random mating and **20261024** for
complete selfing. These are RNG seed integers, not execution dates. Replicates
are isolated subpopulations sharing one seeded pseudorandom stream per
treatment. Standing genotypes are initialized at tick 1 late, after
reproduction; that is t=0. Tick 9 is t=8. Fixed markers remain countable.
`INITIAL_STATE=0` or `2` gives all-aa or all-AA controls in the Eidos script.

## Actual checked output at t=8

| Treatment | Mean H_obs | Expected H_obs | Empirical SE | Mean 2p(1-p) |
| --- | ---: | ---: | ---: | ---: |
| Random mating | 0.482265625 | 0.482760323 | 0.003530083 | 0.479843750 |
| Complete selfing | 0.004648437 | 0.003906250 | 0.000649853 | 0.466890820 |

The selfing treatment has very few heterozygotes while retaining substantial
allelic variation across homozygous individuals. Treating 2p(1-p) as an observed
heterozygote count would miss this distinction. Mean p is 0.508281250 in the
control and 0.507832031 under selfing; neither finite mean is forced to 0.5.

[sample_summary.json](sample_summary.json) records the actual default run:

```sh
.venv/bin/python -m examples.slimulations.selfing.demo > /tmp/selfing-summary.json
diff -u examples/slimulations/selfing/sample_summary.json /tmp/selfing-summary.json
```

Twelve tests cover independent Mendelian and exact-binomial checks, initial
genotypes, zero elapsed time, absorbing homozygous controls, first-generation
segregation, ensemble expectations, the heterozygosity/diversity distinction,
seeds, direct Eidos validation, CLI and checked output. A conservative
Hoeffding bound simultaneously checks 20 bounded means at nominal alpha=10^-6
under independent replicate sampling. The first-generation 1:2:1 fixed-seed
regression also checks each genotype frequency within 0.03 of its expectation;
this separate tolerance is not part of that union bound.

This single-locus idealization excludes partial selfing, linkage, separate
sexes, changing size, fitness differences and observation error. It does not
infer a species' selfing rate or effective population size. Exact random output
is pinned and checked on CI; arbitrary SLiM versions or compilers need not
produce identical realizations.
