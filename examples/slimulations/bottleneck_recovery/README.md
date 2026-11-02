# Census recovery does not restore alleles lost in a bottleneck

This original SLiM experiment compares the same final census size after different
population histories. One neutral allele begins at frequency **0.1**: exactly
20 copies in 100 diploid founders. Three treatments each run **1,024 replicates**
for 40 generations:

| Treatment | Offspring population sizes | Seed |
| --- | --- | ---: |
| Constant | N=100 for generations 1–40 | 202611020 |
| Early bottleneck | N=5 for 1–5, then N=100 for 6–40 | 202611021 |
| Late bottleneck | N=100 for 1–34, N=5 for 35–39, N=100 at 40 | 202611022 |

The early and late histories have the same multiset of sizes and final census,
so their expected final gene diversity is identical. Their allele-loss
probabilities need not be identical: diversity is only one summary of the full
frequency distribution. Census recovery alone introduces no new allele copies
into a population where the allele has disappeared.

## Reproduce

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**, deliberately pinned
for this reference run. The Python wrapper uses only the standard library.
From the repository root, with the pinned SLiM CLI on PATH:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
slim -v
python -m unittest discover -s examples/slimulations/bottleneck_recovery -v
python -m examples.slimulations.bottleneck_recovery.demo --check-sample
```

The wrapper accepts `--slim /path/to/slim`, `--replicates 128` and `--seed 123`.
Changed parameters should be run without `--check-sample`. To run the default
early-bottleneck experiment directly:

```sh
slim -s 202611021 -d REPLICATES=1024 \
  examples/slimulations/bottleneck_recovery/bottleneck.slim
```

The existing [CI build recipe](../../../.github/workflows/tests.yml) downloads
[official SLiM source](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216)
commit `09fcf604a06307cf75098dfa0a2171a9b7852216`, verifies archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`,
and builds only the CLI in a temporary directory. It requires CMake and a C++
compiler; no system installation is performed. Its `-include cstdint` flag is
an existing compatibility fix for modern GCC, not a model change. See the
[simulator setup notes](../neutral_drift/README.md). The checked output is
verified on the local macOS engine and Ubuntu CI; arbitrary builds or versions
are not promised identical pseudorandom streams.

## Model and life-cycle audit

[bottleneck.slim](bottleneck.slim) runs actual diploid, hermaphroditic
Wright–Fisher populations. Parents are chosen independently, so incidental
selfing is allowed. All parental fitnesses are one, and there is no new mutation,
migration or recombination. Every replicate is included after loss or fixation.
Replicates are isolated subpopulations sharing one seeded pseudorandom stream
per treatment.

The founders receive one shared neutral marker per replicate in tick 1's late
stage, after its reproduction; that is elapsed generation zero. Each subsequent
early stage sets the number of offspring in the next reproduction. Thus this
models a reproductive bottleneck, not random removal of current adults.
The [version-matched official method documentation](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/QtSLiM/help/SLiMHelpClasses.html)
explains this timing for `setSubpopulationSize`.

The script audits founder genotype/copy counts, neutral fitness, both engine
versions and the realized population size at **every** generation. The parser
checks complete replicate records, possible genotype counts and persistence of
absorbed states. Fixed markers remain countable via `convertToSubstitution=F`.

## Independent finite-model controls

Let `N_t` be offspring diploids, `K_t` marker copies, and `p_t=K_t/(2N_t)`.
Independent parental sampling gives:

```text
K_t | K_(t-1) ~ Binomial(2N_t, K_(t-1)/(2N_(t-1)))
E[p_t] = p_0
H_t = 2p_t(1-p_t)
E[H_t] = H_0 × product over j=1..t of (1 - 1/(2N_j)).
```

The last identity follows from the binomial first and second moments; it is
exact for this model. `H` is diversity calculated from all population copies,
with replacement, not the fraction of heterozygous individuals. At the first
bottleneck generation those expectations are **0.162** and **0.18**, respectively.
The probability of losing the allele immediately is `0.9^10 = 0.3486784401`.

[demo.py](demo.py) separately propagates the entire finite-state distribution
using binomial transition matrices and checks the diversity product formula.
These are numerical evaluations of an exact finite model, not diffusion or
large-population approximations. [test_demo.py](test_demo.py) independently
sums all ordered gamete draws using rational arithmetic in tiny populations,
and uses an 80-digit direct binomial formula to check numerical probabilities.
No SLiM-generated frequencies enter the analytical predictions.

## Actual checked output

[sample_summary.json](sample_summary.json) contains the actual run and is
checked byte for byte, rounded to nine decimal places. At generation 40 all
three treatments have N=100:

| History | Mean H | Monte Carlo SE | Finite-model E[H] | Lost / 1,024 | Finite-model loss probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | 0.144463037 | 0.004959052 | 0.147297622 | 375 | 0.360368380 |
| Early bottleneck | 0.089851172 | 0.005243810 | 0.089185218 | 752 | 0.728735874 |
| Late bottleneck | 0.090620996 | 0.005227509 | 0.089185218 | 755 | 0.747350525 |

In the early treatment, **713 populations had lost the allele by generation 5**;
all 713 still lack it at generation 40. Increasing census size at generation 6
cannot recreate that allele. Additional losses occur after recovery.

The exact-model difference in early versus late loss probabilities is about
0.01861. The observed loss fractions differ by only about 0.00293, with Monte
Carlo SEs 0.01381 and 0.01376. This batch does **not** resolve that small
ordering effect statistically; the finite-state calculation establishes the
counterexample to inferring allele retention from expected diversity alone.

Ensemble diversity declines without mutation, but individual trajectories and
finite-batch means can fluctuate upward. For example, the early treatment's
mean H rises from 0.105253906 at generation 5 to 0.105717920 at 6, while the
exact expectation falls from 0.106288200 to 0.105756759. This is not evidence
that growth restores variation. Reported SEs are between-replicate Monte Carlo
SEs, not uncertainty in the analytical model. A plug-in SE of zero when no
losses are observed does not prove zero loss probability.

## Tests and scope

The **28 tests** include full tiny-population rational enumeration, high-precision
binomial checks, finite-distribution moments, size ordering, allele-label
symmetry, first-generation loss, absorbing controls in the actual simulator,
every-generation size/copy checks, uncertainty-aware simulation comparisons,
fixed-seed repeatability, invalid inputs, CLI behavior and the checked output.
Five-SE comparisons are numerical sanity bounds, not preregistered hypothesis
tests or confidence claims about natural populations.

This is one biallelic locus with deliberately severe, illustrative size changes.
It does not measure multilocus allelic richness, infer demographic history,
model extinction risk or make conservation recommendations. New mutation or
migration could restore variation, but neither is present here. Census size
matches the ideal-model sampling size; that equality need not hold in nature.
This is a mechanism demonstration, not a full-paper reproduction.

Primary background: Nei, Maruyama and Chakraborty (1975),
[The Bottleneck Effect and Genetic Variability in Populations](https://doi.org/10.1111/j.1558-5646.1975.tb00807.x),
Evolution 29(1):1–10, published March 1, 1975. Simulator reference: Haller and
Messer (2023), [SLiM 4: Multispecies Eco-Evolutionary Modeling](https://doi.org/10.1086/723601).
The variable-size binomial derivation and example implementation here are
original; no paper text, figures or upstream recipes are copied.
The example is covered by the repository [MIT license](../../../LICENSE).
The separately installed SLiM simulator retains its own GPL license.
