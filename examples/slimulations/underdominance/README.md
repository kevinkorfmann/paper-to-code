# Underdominance: an average can conceal population polarization

In this original single-locus SLiM experiment, both homozygotes have reproductive
fitness 1 and the heterozygote has fitness 0.6. This **underdominance** makes the
infinite-population equilibrium at allele frequency 0.5 unstable. A finite
population does not stay balanced merely because it starts exactly there.

The useful distinction is between an ensemble average and the populations that
make up that average. After 80 generations, all 512 selected N=100 populations
starting at p=0.5 had lost one allele: 281 lost A and 231 fixed it. Their mean
p was **0.451172**, with Monte Carlo SE **0.022013**. Calculating gene diversity
from that pooled frequency gives **0.495232**, despite gene diversity being
**zero in every individual population**. This does not imply a stable polymorphism.

## Reproduce

Reference versions are **SLiM 4.0.1**, **Eidos 3.0.1**, and **CPython 3.14.5**.
The wrapper and exact reference calculations use only Python's standard library.
From the repository root with the pinned SLiM executable on PATH:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
slim -v
python -m unittest discover -s examples/slimulations/underdominance -v
python -m examples.slimulations.underdominance.demo --check-sample
```

The wrapper also accepts `--slim /path/to/slim`, `--replicates 128`, and
`--seed 123`. Omit `--check-sample` when changing parameters. For a direct
selected p=0.5 run:

```sh
slim -s 202611031 -d REPLICATES=512 \
  examples/slimulations/underdominance/underdominance.slim
```

The existing [CI recipe](../../../.github/workflows/tests.yml) builds the
[official source revision](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216)
`09fcf604a06307cf75098dfa0a2171a9b7852216`, after checking archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
It needs CMake and a C++ compiler, uses the existing `-include cstdint`
compatibility flag, and builds only the CLI in a temporary directory. See
[setup details](../neutral_drift/README.md). No system installation, paid
service, genetic-data download, or new Python dependency is required.
SLiM 4.0.1 is a deliberate reproducibility pin, not a claim to be the newest release.

## Model and timing

- Diploid, hermaphroditic Wright–Fisher populations with nonoverlapping generations.
- One synthetic biallelic locus, A represented by a retained neutral marker mutation.
  A fitness callback supplies **aa=1, Aa=0.6, AA=1**, or all 1 in neutral controls.
- No new mutation, migration, linkage, or environmental variation.
- Each parental role is drawn independently with probability proportional to
  reproductive fitness. Mendelian gametes are independent. Incidental selfing
  is explicitly allowed; forbidding it would change the reference distribution.
- Exact founder genotype counts are installed in tick 1's late event, after
  creation, and fitness is recalculated. That state is elapsed generation 0.
  The first selected offspring are recorded at tick 2, elapsed generation 1.
- Founder fitnesses are audited using SLiM's cached fitnesses. Every generation's
  genotype counts are parsed, with checks for total size, missing/duplicate
  rows, initial conditions, versions, and impossible reversals of fixation.
- Each replicate is an isolated subpopulation. There is one seeded pseudorandom
  stream per treatment, not separately reseeded populations.

All main founders have exact Hardy–Weinberg genotype proportions. Counts below
are ordered **aa, Aa, AA**:

| Treatment | N | Founders | Heterozygote fitness | Seed |
| --- | ---: | --- | ---: | ---: |
| Below threshold | 100 | 36, 48, 16 | 0.6 | 202611030 |
| At threshold | 100 | 25, 50, 25 | 0.6 | 202611031 |
| Above threshold | 100 | 16, 48, 36 | 0.6 | 202611032 |
| Neutral at 0.5 | 100 | 25, 50, 25 | 1 | 202611033 |
| Small selected | 8 | 2, 4, 2 | 0.6 | 202611034 |
| Small neutral | 8 | 2, 4, 2 | 1 | 202611035 |

Each treatment runs 512 replicates for 80 generations.

## Three different predictions

### Exact next-generation conditional distribution

For current counts `(a, h, d)` and heterozygote fitness `w`, a fitness-weighted
parental gamete carries A with probability

```text
q = (d + w*h/2) / (a + w*h + d).
```

Consequently the next genotype counts have a multinomial distribution:

```text
(aa', Aa', AA') ~ Multinomial(N; (1-q)^2, 2*q*(1-q), q^2).
```

The allele count alone is conditionally binomial, `K' ~ Binomial(2N, q)`.
But **allele count is not a sufficient state for a multigeneration selected
model**. For example, `(6,0,2)` and `(4,4,0)` both have p=0.25 at N=8. With
w=0.6 their next-generation expected frequencies are respectively 0.25 and
0.1875. The implementation retains genotype counts rather than silently
reimposing Hardy–Weinberg proportions in each finite population.

The first-generation expectations from p=0.4, 0.5, 0.6 main founders are
0.376237624, 0.5, and 0.623762376. Tests compare actual means, variances,
heterozygosities, and a complete tiny-population genotype distribution with
independent weighted-parent/Mendelian enumeration.

### Infinite random-mating benchmark

With Hardy–Weinberg proportions in an infinite population,

```text
p_next = [p^2 + w*p*(1-p)] / [1 - 2*(1-w)*p*(1-p)].
```

At w=0.6 the derivative at p=0.5 is 1.25. Small deviations increase rather
than contract. Frequencies below 0.5 move downward and those above it upward;
exactly 0.5 remains exactly 0.5 in this deterministic recurrence. These curves
are benchmarks, **not exact predictions of finite-ensemble means**.

### Exact finite-genotype reference

For N=8, there are 45 possible genotype-count states. [demo.py](demo.py)
constructs their full transition matrix from the multinomial formula and
propagates the distribution numerically. This is an exact finite-model
calculation evaluated in floating point, not a diffusion approximation.

Symmetric founders and fitnesses make the finite ensemble's expected p equal
to 0.5 at every time, even while its distribution separates toward 0 and 1.
At generation 80, the N=8 selected model predicts gene diversity
0.000000019 and loss/fixation probabilities 0.499999968 each. Its neutral
counterpart predicts diversity 0.002862016. Neutral diversity also has the
independent identity `E[H_t] = H_0*(1-1/(2N))^t`.

## Actual checked output

[sample_summary.json](sample_summary.json) is the actual pinned-engine output,
rounded to nine decimal places. It includes generations 0, 1, 10, 40, and 80,
seeds, fitnesses, founder counts, uncertainty summaries and reference values.
At generation 80:

| Treatment | Mean p | MC SE of mean p | Mean within-population H | Lost / fixed / segregating |
| --- | ---: | ---: | ---: | --- |
| Below threshold | 0.019531250 | 0.006121692 | 0 | 502 / 10 / 0 |
| At threshold | 0.451171875 | 0.022012975 | 0 | 281 / 231 / 0 |
| Above threshold | 0.980468750 | 0.006121692 | 0 | 10 / 502 / 0 |
| Neutral at 0.5 | 0.496435547 | 0.012265843 | 0.346213770 | 19 / 18 / 475 |
| Small selected | 0.517578125 | 0.022105025 | 0 | 247 / 265 / 0 |
| Small neutral | 0.521362305 | 0.022056911 | 0.001876831 | 244 / 266 / 2 |

The neutral N=100 H has Monte Carlo SE 0.007073358 and theoretical expectation
0.334823910. All averages include every replicate, including loss and fixation.
The selected p=0.5 sample's deviation from its exact ensemble mean is about
2.22 Monte Carlo SEs, not evidence of an allele-label preference.

Starting below the threshold predicts the majority outcome, not every outcome:
10/512 of the p=0.4 populations fixed A. These are outcomes at generation 80,
not general fixation probabilities, and their magnitudes depend on N, fitnesses,
and founders. No absent allele can reappear under this mutation-free model.

Gene diversity is `H=2p(1-p)`, not the realized heterozygote fraction. Moreover,
`2*mean(p)*(1-mean(p)) - mean(H) = 2*population_variance(p)` across equal-size
replicates. The example tests this identity directly. Pooling independently
fixed populations must not be mistaken for maintaining variation within them.

## Verification and limits

The **32 tests** include exact rational ordered-parent and ordered-offspring
oracles, a multigeneration rational calculation, transition conservation and
moments, neutral reductions, allele-label symmetry, absorbing boundaries,
unstable-equilibrium direction and derivative, actual SLiM distribution checks,
incidental-selfing and genotype-composition controls, all generations, seed
repeatability, invalid inputs, parser corruption, the CLI, and byte-identical
checked JSON. Reference-model moments supply Monte Carlo error for the small
selected population. Statistical envelopes and qualitative cutoffs are fixed
numerical sanity checks, not preregistered significance tests or guarantees
about arbitrary seeds. Reported SEs measure simulation sampling variation,
not uncertainty in real biological fitnesses.

This is a deliberately simple and strongly selected single-locus demonstration.
It does not model a specific organism, chromosomal arrangement, measured fitness
landscape, structured population, or genetic intervention. Heterozygote
disadvantage need not have these outcomes in models with migration, multiple
loci, sex-specific selection, or frequency-dependent fitness.

The pinned output was produced in Linux with the stated versions. CI rebuilds
SLiM from the official source and rechecks it. Other engine/compiler versions
are not guaranteed identical random streams.

## Primary sources and scope

- Altrock, Traulsen & Reed (2011), [Stability Properties of Underdominance in
  Finite Subdivided Populations](https://doi.org/10.1371/journal.pcbi.1002260).
  This primary theoretical study discusses underdominance, unstable single-
  population equilibria, and finite-population behavior. Its subdivided Moran
  model is **not reproduced here**; our synthetic model is discrete-generation,
  isolated-population, diploid Wright–Fisher, with its own explicit derivation.
- Haller & Messer (2023), [SLiM 4: Multispecies Eco-Evolutionary
  Modeling](https://doi.org/10.1086/723601), the simulator reference.
- [Official SLiM 4.0.1 release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1)
  and [pinned WF reproduction implementation](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/population.cpp),
  for the engine version and fitness-weighted parental sampling semantics.

The script, wrapper, tests and explanation are original and use the repository's
[MIT license](../../../LICENSE). Separately installed SLiM retains its own GPL
license. No source-paper figures, upstream recipes, private research or real
individuals' genetic information are included.
