# Reversible mutation restores variation after fixation

**2026-10-30 · SLiMulations · original synthetic experiment**

Without mutation, a neutral allele that disappears cannot return. With recurrent
mutation in both directions, monomorphic states are **nonabsorbing**, not
impossible. This small SLiM model starts every population without allele A and
follows recovery of variation under a two-state mutation process.

The conceptual reference is [Charlesworth and Jain (2014), *Purifying Selection,
Drift, and Reversible Mutation with Arbitrarily High Mutation Rates*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4256773/).
Their basic assumptions discuss reversible two-state mutation in finite,
randomly mating populations, and distinguish stationary ensemble distributions
from continual changes at individual sites. Here there is **no selection**.
The experiment, implementation and finite-generation derivations below are
original, not a reproduction of that paper. In particular, its diffusion
approximation is **not** substituted for our exact finite Wright–Fisher moments.

## Run and reproduce

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**, from the repository root:

```sh
slim -v
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/slimulations/reversible_mutation -v
python -m examples.slimulations.reversible_mutation.demo --check-sample
```

The last command reruns all treatments and compares the complete JSON report
byte for byte with [sample_summary.json](sample_summary.json). The wrapper also
checks both engine versions and verifies fitness equals one at every recorded
checkpoint. No third-party Python packages or real genetic data are needed.
An exploratory run uses:

```sh
python -m examples.slimulations.reversible_mutation.demo --replicates 32 --seed 123
```

Run the original Eidos model directly, including the actual small-N sample seed:

```sh
slim -s 202610301 -d N=20 -d INITIAL_COPIES=0 -d U=0.01 -d V=0.01 \
  -d REPLICATES=512 -d GENERATIONS=80 \
  examples/slimulations/reversible_mutation/mutation.slim
```

Raw output includes versions, `FIT,generation,min_fitness,max_fitness`, and
`DATA,generation,replicate,N,A_copies`. `GENERATIONS` accepts 0–200. Checkpoints
are generation 0, 1, the integer midpoint, and the endpoint, with duplicates
removed. There is no first-generation mutation before the installed founders.

This reuses the repository's pinned environment, not the newest SLiM release.
The [official release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1)
provides the manual. Existing [CI](../../../.github/workflows/tests.yml) builds
source revision `09fcf604a06307cf75098dfa0a2171a9b7852216` from an archive with
SHA-256 `e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
No dependencies, paid services, schedules, security settings or data downloads
are added. Byte-level reproducibility is for the pinned environment, not a
promise about future engine versions or arbitrary architectures.

## Model and implementation

Each replicate is a separate, constant-size diploid Wright–Fisher population
with one synthetic biallelic locus. Generations do not overlap. Parents are
sampled independently with replacement, allowing incidental selfing. All
genotypes have fitness one, Mendelian inheritance precedes mutation, and there
is no migration or recombination. Population sizes are **N=20 or N=80**.

Each inherited gene copy independently changes state once per generation:

- a → A with probability **u**.
- A → a with probability **v**.

The `modifyChild()` callback makes exactly one state-dependent Bernoulli draw
per child genome. It does **not** apply forward and reverse mutations
successively to the same copy. Background mutation is disabled because the
callback implements the entire two-state mutation process. It never rejects
offspring or changes parent choice.

Presence of an `m1` marker means A, and absence means a. Forward mutation adds
a new neutral marker at the single position only when no marker is present.
Reverse mutation removes the marker. Different marker identifiers all encode
the **same A state**, not different alleles or genealogical origins for analysis.
`convertToSubstitution=F` preserves the state when A becomes fixed. The script
asserts that every genome has at most one marker at recorded checkpoints.
These explicit state changes are a finite-alleles model, not an infinite-sites
or full nucleotide mutation model. No tree sequence is produced.

## Exact finite-population predictions

Let M=2N, p be the parental A frequency, and c=1−u−v. Independent inheritance
and state changes imply

```text
q(p) = u + c*p
K_next | p ~ Binomial(M, q(p))
p_next = K_next/M.
```

For u+v>0 and p0=0, the exact mean is

```text
E[p_t] = [u/(u+v)] * [1 - (1-u-v)^t].
```

This formula describes an ensemble mean, not an individual trajectory. The
reported main rates have 0<u,v<1, for which the finite chain has a unique
stationary distribution. Its mean is u/(u+v), but **generation 80 is not
asserted to be stationary**. For u=v=.01, E[p80]=0.400675575, not 0.5.

We measure within-population **gene diversity H=2p(1−p)**: the probability of
different states in two gene-copy draws with replacement. It is not the
observed fraction of heterozygous individuals or a without-replacement
sample correction. Averaging H differs from inserting an average frequency:

```text
2*E[p]*(1-E[p]) - E[H] = 2*Var(p).
```

For symmetric mutation u=v=μ and b=1−1/M, direct binomial conditioning gives

```text
E[H_(t+1)] = b * [2*μ*(1-μ) + (1-2*μ)^2 * E[H_t]]
H_stationary = 2*b*μ*(1-μ) / [1-b*(1-2*μ)^2].
```

The finite-N factor b matters. The stationary values are **0.303490017** for
N=20 and **0.431473055** for N=80. The finite-time comparison uses the recurrence,
not those limits. For μ=0, the recurrence reduces to neutral drift's diversity
decay; monomorphic founders remain monomorphic.

The Python oracle propagates the first four raw moments, evaluated in ordinary
binary64 arithmetic. Write m_j=E[p^j], m_0=1 and
`B_j = Σ_(l=0)^j choose(j,l) u^(j-l) c^l m_l`. Then

```text
m1_next = B1
m2_next = [B1 + (M-1)*B2]/M
m3_next = [B1 + 3*(M-1)*B2 + (M-1)*(M-2)*B3]/M^2
m4_next = [B1 + 7*(M-1)*B2 + 6*(M-1)*(M-2)*B3
           + (M-1)*(M-2)*(M-3)*B4]/M^3.

Var(p) = m2-m1^2
E[H] = 2*(m1-m2)
Var(H) = 4*(m2-2*m3+m4)-E[H]^2.
```

These are exact recurrences for the specified finite model, not a diffusion
approximation. Tests independently enumerate all possible allele counts and
their transitions for N=3 using rational arithmetic over multiple generations.

## Actual output

**512 replicates per treatment**, all starting at p0=0 and followed for 80
generations. Seeds **202610300–202610303**, in table order. Every replicate,
including a monomorphic one, enters the averages.

| Treatment | N | u, v | Observed mean p80 ± Monte Carlo SE | Observed mean H80 ± Monte Carlo SE | Exact E[H80] |
| --- | ---: | --- | ---: | ---: | ---: |
| No mutation | 20 | 0, 0 | 0 ± 0 | 0 ± 0 | 0 |
| Symmetric, small | 20 | .01, .01 | .381347656 ± .012959667 | .300195312 ± .007337941 | .301909913 |
| Symmetric, large | 80 | .01, .01 | .397314453 ± .007869694 | .415616760 ± .004013008 | .421162142 |
| Biased mutation | 20 | .02, .005 | .689941406 ± .012819195 | .259897461 ± .008275703 | .271577952 |

Mutation restores variation absent in the no-mutation control. At the same
symmetric mutation probabilities, larger populations retain more within-population
diversity. For the small symmetric treatment, inserting the replicate mean
frequency gives **0.471843243**, far above the actual mean diversity 0.300195312.
Different replicate populations should not be treated as one mating population.

At generation 80, **28/512** small symmetric populations have no A and **7/512**
have only A. Reversible mutation does not prevent temporary fixation. The
next-generation probability of leaving p=0 is `1-(1-u)^(2N)` and that of leaving
p=1 is `1-(1-v)^(2N)`.

Biased mutation has exact E[p80]=**0.694449756**, toward a stationary mean 0.8.
Its higher A frequency requires no fitness advantage. It does not show
adaptation, mutation directed toward a beneficial allele, or selection.
Monte Carlo SE describes uncertainty in replicate means, not empirical
parameter uncertainty or a prediction interval for an individual population.

## Verification and limits

The **23 tests** include independent full rational transition-matrix enumeration
of four moments and diversity variance; closed-form mean/diversity checks;
allele-complement symmetry; and actual SLiM means checked using theoretical,
not fitted, variances. Conservative six-SE bounds are regression tolerances,
not a claim about biological confidence.

Engine controls cover absorbing no-mutation boundaries, deterministic one-way
mutation, alternating fixation with u=v=1 (catching accidental double flips),
and u=v=.5, which resets offspring to `Binomial(2N,.5)` regardless of the
parental frequency. The latter checks both mean and variance using an
independent binomial fourth moment. Additional checks verify first-generation
creation probability, exact founders, versions, neutral fitness, repeated
seeds, all-replicate summaries, invalid inputs, parser guards and CLI output.

Rates are deliberately large **illustrative transition probabilities**, not
organism-specific nucleotide rates. There is no empirical calibration, selection,
sex structure, linkage, demography, mutation-rate evolution, inference, ancestry
analysis, or proof that every stochastic trajectory approaches the ensemble
mean. The experiment is not a molecular mechanism or a real-population forecast.

## Sources and licenses

- [Charlesworth and Jain (2014)](https://doi.org/10.1534/genetics.114.167973): reversible-mutation framework and statistical-equilibrium interpretation. Its diffusion formulas are not our exact discrete-generation oracle.
- [Haller and Messer (2023), SLiM 4](https://doi.org/10.1086/723601): simulator reference.
- [Official SLiM 4.0.1 manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf): `modifyChild()`, genome mutation editing, and `preventIncidentalSelfing`.
- [Pinned genome-editing implementation](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/genome.cpp): upstream API implementation, not copied into this example.

The new model, derivations, Python analysis, tests and prose are original and
use this repository's [MIT license](../../../LICENSE). Integer/version helpers
are reused from its neutral-drift example. Externally installed SLiM retains
its GPL-3.0 license. No upstream code, paper figures, private research,
credentials or real genetic data are redistributed.
