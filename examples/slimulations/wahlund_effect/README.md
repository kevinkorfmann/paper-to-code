# Wahlund effect: pooling samples is different from pooling parents

**2026-10-27 · slimulations · actual SLiM 4.0.1 / Eidos 3.0.1**

Two populations can each mate randomly and nevertheless show an apparent
heterozygote deficit when their genotype counts are pooled. Pooling the counts
does not change a single genotype. Allowing the founders to mate in one common
population does change the next generation's genotypes.

This original synthetic experiment follows the mathematical idea in Sten
Wahlund's [1928 paper](https://doi.org/10.1111/j.1601-5223.1928.tb02483.x),
*Zusammensetzung von Populationen und Korrelationserscheinungen vom Standpunkt
der Vererbungslehre aus betrachtet*, **Hereditas 11:65–106, May 1928**.
The [primary PDF](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1601-5223.1928.tb02483.x),
especially pp. 83–88, relates the heterozygote deficit to variation in allele
frequencies among populations. We demonstrate that identity and add a finite
one-generation control; we do not reproduce the full paper or use human data.

## Run and environment

From the repository root, using **CPython 3.14.5**, **SLiM 4.0.1**, and its
bundled **Eidos 3.0.1**:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
slim -v
eidos -v
python -m unittest discover -s examples/slimulations/wahlund_effect -v
python -m examples.slimulations.wahlund_effect.demo
python -m examples.slimulations.wahlund_effect.demo > /tmp/wahlund-summary.json
diff -u examples/slimulations/wahlund_effect/sample_summary.json /tmp/wahlund-summary.json
```

The interpreter is pinned in the root [.python-version](../../../.python-version),
and the wrapper uses only the standard library. No new Python dependency is
required. Use `--slim /path/to/slim` to select an existing engine;
`--replicates 128` or `--seed 1234` changes the experiment. Other SLiM versions
are rejected by the wrapper, because API and random-stream compatibility are
not assumed. This is a pin to an existing tested release, not the newest SLiM.

The [existing CI recipe](../../../.github/workflows/tests.yml) builds the
[official 4.0.1 source](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1),
revision `09fcf604a06307cf75098dfa0a2171a9b7852216`, in a temporary directory.
It verifies SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`
and builds only the CLI. No system installation is changed.

## Controlled model

[pooling.slim](pooling.slim) runs one neutral, biallelic locus in diploid
hermaphrodites with nonoverlapping WF generations. Fitness is equal, mutation
and recombination are disabled, and separate demes have no migration.
`preventIncidentalSelfing=F` explicitly allows independent parental draws,
including selection of the same parent twice by chance. This is the convention
in the [version-matched manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf),
`initializeSLiMOptions` reference, used by our analytical expectations.

Each scenario has **256 replicate population systems per mating treatment**:

| Scenario | Founder group sizes | Founder A frequencies | Separate seed | Common-pool seed |
| --- | --- | --- | ---: | ---: |
| Different frequencies | 100, 100 | 0.2, 0.8 | 202610270 | 202610271 |
| Equal-frequency control | 100, 100 | 0.5, 0.5 | 202610272 | 202610273 |
| Unequal-size control | 100, 300 | 0.2, 0.8 | 202610274 | 202610275 |
| Opposite-fixation control | 100, 100 | 0, 1 | 202610276 | 202610277 |

Founder genotype counts are exactly Hardy–Weinberg within each founder group.
For example, N=100, p=0.2 gives `(AA,Aa,aa)=(4,32,64)`; p=0.8 gives
`(64,32,4)`. The common-pool treatment places those same genotypes in a single
population, without altering them. The separate treatment keeps two mating
populations and combines their counts only during analysis. The mutation
object representing A is shared across founder groups in a replicate.

Founders are initialized in `1 late()`, after tick 1 reproduction; this is
recorded as t=0. Tick 2 records t=1, after exactly one subsequent reproduction.
Fixed alleles remain countable via `convertToSubstitution=F`. Every individual
in the census is counted; there is no additional sampling or genotype error.

Run the main separate-deme treatment directly:

```sh
slim -s 202610270 -d POOLED=0 -d REPLICATES=256 \
  examples/slimulations/wahlund_effect/pooling.slim
```

`POOLED=1` changes the mating population. Direct parameters `N1`, `N2`, `AA1`,
`HET1`, `AA2`, and `HET2` specify founder counts. `GENERATIONS` is deliberately
limited to zero or one; the analytical oracle is a one-generation calculation.
`DATA` rows contain `generation,replicate,group,N,AA_count,Aa_count`.

## Quantities and exact predictions

For mating-group weights `w_i=N_i/N_total` and frequencies `p_i`, define

```text
p_bar = sum_i w_i*p_i
H_T   = 2*p_bar*(1-p_bar)
H_S   = sum_i w_i*2*p_i*(1-p_i)
H_T - H_S = 2*sum_i w_i*(p_i-p_bar)^2.
```

Observed heterozygosity `H_obs` is the fraction of **individuals** that are Aa.
The quantities `H_T` and `H_S` instead use allele frequencies. In the exact HWE
founders, `H_obs=H_S`; with equal groups at 0.2 and 0.8, `H_obs=0.32` while
`H_T=0.50`, a gap of **0.18**. Unequal groups receive individual weights, not
equal group weights: the 100:300 control has `p_bar=0.65`, `H_T=0.455`, and
gap **0.135**.

Realized offspring genotypes need not be exactly HWE. The general accounting
identity is

```text
H_T - H_obs = 2*Var_w(p_i) + (H_S - H_obs).
```

The JSON reports both terms. If the same founders are treated as one mating
group, their initial deficit is entirely in the within-group residual; the
total gap and observed heterozygosity remain unchanged. No genotypes change
merely because a group label is removed.

For t=1, condition on the fixed founders. Separate mating gives
`E[H_obs]=H_S`; common-pool mating gives `E[H_obs]=H_T`, where these right-hand
sides use the **founder** frequencies. The pooled offspring frequency is random:

```text
Var(p_hat) = E[H_obs]/(4*N_total)
E[2*p_hat*(1-p_hat)] = H_T - E[H_obs]/(2*N_total).
```

Consequently, in the equal-size differentiated scenario, the exact finite
expectations at t=1 are:

| Mating | E[H_obs] | E[realized-frequency H_T] | E[H_T-H_obs] |
| --- | ---: | ---: | ---: |
| Separate demes | 0.32 | 0.49920 | 0.17920 |
| Common pool | 0.50 | 0.49875 | -0.00125 |

The small negative common-pool gap is expected from this uncorrected
finite-census statistic. We do not clip it to zero or interpret it as evidence
of selection. A finite replicate can have a gap of either sign.

## Actual output

[sample_summary.json](sample_summary.json) is the executed default run,
rounded to nine decimals. These are t=1 observations, not invented expectations:

| Scenario | Mating | Mean H_obs | Monte Carlo SE | Expected H_obs | Mean H_T-H_obs |
| --- | --- | ---: | ---: | ---: | ---: |
| Different frequencies | Separate | 0.320917969 | 0.002026212 | 0.320 | 0.178321826 |
| Different frequencies | Common pool | 0.501035156 | 0.002315332 | 0.500 | -0.002350830 |
| Equal frequency | Separate | 0.498574219 | 0.002235570 | 0.500 | -0.000090967 |
| Equal frequency | Common pool | 0.497675781 | 0.002071013 | 0.500 | 0.001051416 |
| Unequal sizes | Separate | 0.322548828 | 0.001349619 | 0.320 | 0.132432141 |
| Unequal sizes | Common pool | 0.456767578 | 0.001616500 | 0.455 | -0.002647864 |
| Opposite fixation | Separate | 0 | 0 | 0 | 0.500000000 |
| Opposite fixation | Common pool | 0.501347656 | 0.002178676 | 0.500 | -0.002714307 |

The extreme control is useful: separate populations fixed for opposite alleles
contain no heterozygotes, whereas their common-pool offspring can be heterozygous.
Equal founder frequencies remove the structural deficit even with two mating
groups. All replicate systems contribute to each result. Standard errors are
between-replicate sample standard deviations divided by sqrt(256).

## Tests and limitations

The **20 tests** in [test_demo.py](test_demo.py) check exact founders, zero-duration
timing, the weighted-variance identity with rational arithmetic, unequal sizes,
population swapping and allele relabeling, count pooling, and actual mating.
An independent exact binomial enumeration verifies the one-generation oracle.
All eight treatments are compared with theoretical moments for allele frequency,
observed heterozygosity, and the realized-frequency `H_T`. Six theoretical
standard errors are used as diagnostics; Bernoulli cumulants supply the variance
of the quadratic `H_T` statistic. Tests also cover absorbing boundaries, seeded
repeatability, invalid inputs, malformed engine output, Eidos validation, CLI
behavior, and the saved summary. CI compares the complete JSON byte for byte.

This is a mechanism demonstration with fixed founder frequencies, not an
inference of population structure, a pedigree-inbreeding coefficient, or an
HWE significance test. The starting differentiation is imposed, not evolved.
There is no explicit selfing treatment, selection, migration process, linkage,
demography, or organism-specific biology. Common-pool mating is a deliberately
different reproductive treatment, not a claim that combining data causes
biological admixture. It also differs from the earlier migration-rate and
selfing examples.

Replicates are isolated systems sharing a seeded pseudorandom stream per
treatment. Monte Carlo SE describes variation under this model, not uncertainty
in biological parameters. Engine/compiler changes can alter random streams;
the reference output is checked locally and in the pinned Ubuntu CI build.

The script, Python wrapper, tests, and constructed data are original and
[MIT licensed](../../../LICENSE). SLiM is installed separately under its GPL
license; no upstream recipes or simulator implementation are redistributed.
Simulator citation: Haller and Messer (2023),
[*SLiM 4: Multispecies Eco-Evolutionary Modeling*](https://doi.org/10.1086/723601).
