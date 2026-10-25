# slimulations: heterozygote advantage does not prevent finite-population loss

An actual SLiM simulation follows a single allele **A** when heterozygotes have
higher fitness than either homozygote. The infinite-population model has a
stable interior equilibrium. Finite populations fluctuate and can still fix or
lose A. Neutral and additive-selection controls separate this behavior from
ordinary drift and directional selection.

The classical source is R. A. Fisher's
[*On the Dominance Ratio*](https://hdl.handle.net/2440/15098), commonly cited as
1922, *Proceedings of the Royal Society of Edinburgh* 42:321–341.
Its [publisher summary](https://doi.org/10.1017/S0370164600023993) discusses
selection favoring heterozygotes; Cambridge's volume metadata labels the year
1923. This example implements a specified one-locus model, not the full paper.

## Model and reference environment

| Mode | SLiM `s` | SLiM `h` | `w_aa` | `w_Aa` | `w_AA` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 0 | 0.5 | 1 | 1 | 1 |
| Additive | 0.1 | 0.5 | 1 | 1.05 | 1.1 |
| Heterozygote advantage / overdominance | 0.1 | 2 | 1 | 1.2 | 1.1 |

The [official SLiM 4.0.1 manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf),
section 1.5.2 and `initializeMutationType`, defines heterozygote fitness as
`1+h*s` and mutant homozygote fitness as `1+s`; `h>1` is allowed.
We use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**, the repository's
existing verified environment. This is a reproducibility pin, not a claim that
4.0.1 is current. See the
[SLiM 4 paper](https://doi.org/10.1086/723601) and
[official release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1).

The WF model uses diploid hermaphrodites, nonoverlapping generations, constant
census size, one locus, and independent fitness-weighted parental draws.
Incidental selfing is explicitly permitted. Mutation, migration, and recombination
are absent. There are no sex differences, density effects, or spatial structure.
The advantage is a fixed property of the genotype, not an environmental feedback.

## Run

From the repository root, with the pinned `slim` on PATH:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
slim -v
python -m unittest discover -s examples/slimulations/heterozygote_advantage -v
python -m examples.slimulations.heterozygote_advantage.demo
python -m examples.slimulations.heterozygote_advantage.demo > /tmp/selection-summary.json
diff -u examples/slimulations/heterozygote_advantage/sample_summary.json /tmp/selection-summary.json
```

The standard-library wrapper needs no third-party Python dependencies. The root
[.python-version](../../../.python-version) pins Python. If SLiM is unavailable,
the repository's [CI build recipe](../../../.github/workflows/tests.yml) builds
the checksum-verified official 4.0.1 CLI in a temporary directory without replacing
system tools. Its source revision is `09fcf604a06307cf75098dfa0a2171a9b7852216`.
`--slim /path/to/slim` selects an existing executable; other versions are rejected.

The original [Eidos script](selection.slim) can also run directly:

```sh
slim -s 20261029 -d MODE=2 -d N=200 -d REPLICATES=128 \
  -d GENERATIONS=80 -d INITIAL_AA=8 -d INITIAL_HET=64 \
  examples/slimulations/heterozygote_advantage/selection.slim
```

`MODE=0,1,2` means neutral, additive, overdominant. `DATA` rows contain
`elapsed_generation,replicate_id,number_AA,number_Aa`; `aa` is the remaining
population. `FIT` rows audit SLiM's cached fitness for each genotype present at
initialization. The Python wrapper checks this audit against the intended
fitnesses. For CLI exploration, `--replicates`, `--generations`, and `--seed`
change the defaults. `--generations` changes the N=200 comparison only: the
explicit N=25 control always runs 200 generations.

## Initialization and predictions

Each of 128 isolated populations starts at p(A)=0.2 or 0.8 with **exact
Hardy–Weinberg genotype counts**. For N=200 these are `(AA,Aa,aa)=(8,64,128)`
or `(128,64,8)`; for N=25 they are `(1,8,16)` or `(16,8,1)`.
The script adds one shared mutation object per population to the relevant
genomes after tick 1 reproduction, records t=0, and keeps fixed mutations
countable with `convertToSubstitution=F`. Tick 81 is therefore t=80.

Let p be A frequency and q=1-p. In an infinite randomly mating population:

```text
mean_fitness = q^2*w_aa + 2*p*q*w_Aa + p^2*w_AA
p_next = (p^2*w_AA + p*q*w_Aa) / mean_fitness
p_star = (w_Aa-w_aa)/(2*w_Aa-w_aa-w_AA) = 2/3
```

For the overdominant fitnesses above, the recursion increases p below 2/3 and
decreases it above 2/3. Additive selection instead increases A at every interior
frequency; neutrality leaves the deterministic frequency unchanged.

In a finite population, actual genotype counts also matter. Conditional on
parental counts `(n_aa,n_Aa,n_AA)`, the probability that a gamete is A is

```text
P_A = (n_AA*w_AA + n_Aa*w_Aa/2) /
      (n_aa*w_aa + n_Aa*w_Aa + n_AA*w_AA)
X_next | parental genotypes ~ Binomial(2*N, P_A).
```

For example, all-Aa parents and a 50:50 mixture of AA and aa both have p=0.5,
yet produce different fitness-weighted gamete frequencies. Thus the nonlinear
infinite-population recursion is **not the exact ensemble expectation** of the
finite model. Both the genotype detail and averaging a nonlinear function matter.

## Actual checked results

The complete [sample_summary.json](sample_summary.json) records checkpoints,
mean frequencies, ordinary between-replicate Monte Carlo SEs, heterozygosity,
and fixed/lost/segregating counts. Every mean includes **all 128 populations**,
including those absorbed at a boundary. No survivor conditioning is used.

| Mode | Initial p | Seed | Mean p at t=80 | Monte Carlo SE | Infinite-population p at t=80 | Lost / fixed / segregating |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Neutral | 0.2 | 20261025 | 0.185390625 | 0.014372765 | 0.200000000 | 16 / 0 / 112 |
| Neutral | 0.8 | 20261026 | 0.807578125 | 0.015580842 | 0.800000000 | 0 / 15 / 113 |
| Additive | 0.2 | 20261027 | 0.889101563 | 0.009605585 | 0.917357405 | 0 / 13 / 115 |
| Additive | 0.8 | 20261028 | 0.988769531 | 0.002522505 | 0.994048914 | 0 / 93 / 35 |
| Overdominant | 0.2 | 20261029 | 0.679863281 | 0.006796557 | 0.663692198 | 0 / 0 / 128 |
| Overdominant | 0.8 | 20261030 | 0.680312500 | 0.006325577 | 0.668066386 | 0 / 0 / 128 |

At **N=25, t=200**, overdominance does not preserve all replicates:

| Initial p | Seed | Mean p | Monte Carlo SE | Lost / fixed / segregating |
| ---: | ---: | ---: | ---: | --- |
| 0.2 | 20261031 | 0.840937500 | 0.026032667 | 10 / 86 / 32 |
| 0.8 | 20261032 | 0.906562500 | 0.020178960 | 5 / 101 / 22 |

These are outcomes at the stated observation times, not estimates of eventual
fixation probabilities. Once all copies are A or a, selection cannot recreate
the absent allele because mutation and migration are disabled. The small-population
means should not be interpreted as a new deterministic equilibrium.

## Checks and limits

The **17 tests** in [test_demo.py](test_demo.py) include exact rational
enumeration of Mendelian parental crosses, the equilibrium and attraction
direction, initial timing/genotypes, the engine's fitness audit, absorbing
boundaries, seed repeatability, direct Eidos validation, the CLI, and the checked
JSON. A one-generation experiment compares all three modes at both starting
frequencies with binomial mean, variance, and offspring-genotype predictions.
Those checks use six theoretical standard errors as diagnostics. Neutral means
also have a conservative simultaneous Hoeffding bound. Selection comparisons
check the qualitative behavior in the fixed example without asserting equality
between the finite mean and deterministic curve.

Replicates are isolated subpopulations using one seeded pseudorandom stream per
treatment. SE is sample standard deviation divided by sqrt(replicates); it
describes simulation variation under this model, not uncertainty about real
biological fitnesses. Exact random streams are only checked for the pinned
engine and reference platforms; other simulator/compiler versions can differ.
CI builds SLiM, runs all tests, and compares the JSON byte for byte.

This is one synthetic locus with chosen, relatively strong selection. It does
not model a named disease, organism, genomic background, or measured fitness
landscape. Census size should not automatically be equated to effective size
in other models. The original script, wrapper, tests, and documentation use the
repository's [MIT license](../../../LICENSE); separately installed SLiM retains
its GPL license. No upstream recipes or simulator source are copied into this
example.
