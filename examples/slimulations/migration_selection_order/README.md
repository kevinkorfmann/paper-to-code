# Migration and selection: the order can reverse the change

Original educational SLiM 4.0.1 model, November 13, 2026. Starting at p(A)=0.5,
we apply the same two processes in opposite orders: selection with reproductive
weights AA=2, aa=1, and immigration from an all-aa source with probability 0.3.
The expected final frequency is below 0.5 for selection then migration, but above
0.5 for migration then selection.

## What is simulated

This is **two clonal Wright–Fisher replacement stages**, not a full sexual life
cycle compressed into one generation. Each stage has N=100 individuals. All
individuals are homozygous at one neutral marker, which tracks the selected
clone. During the selection stage there is no immigration and parent sampling
is fitness-weighted. During the migration stage fitness is equal: each offspring
comes from the fixed source with probability m, or from a uniform local parent
otherwise. SLiM's actual WF cloning and migration machinery creates offspring.
No new mutation, recombination, mating, source feedback or density dependence.
The sampling units are N individuals, not 2N independent gametes. The shared
source is genetically constant, so it adds no fluctuating common environment.

Census occurs after both stages. There is finite resampling at *each* stage.
Reversing order changes the model; it is not merely relabeling a census point in
a stationary process. All 1,024 replicates contribute, including losses.

## Derivation and output

Let S(p)=wp/[1+(w-1)p] and M(p)=(1-m)p for this all-a source.
In an infinite population:

- S then M: M(S(1/2)) = 7/15 = 0.4666667.
- M then S: S(M(1/2)) = 14/27 = 0.5185185.

For a finite population, if q is the first-stage probability, intermediate
count J is Binomial(N,q). Conditional on J=j, the final count is
Binomial(N,r(j/N)), where r is the second operator. The exact mean is
sum_j P(J=j) r(j/N). Its variance uses the conditional binomial variance plus
the variance across intermediate counts. `prediction()` computes these sums,
not a deterministic curve mislabeled as a finite mean.

Actual output, CPython 3.14.5 / SLiM 4.0.1 (Eidos 3.0.1):

| Treatment | Simulated final mean | Exact finite mean | Deterministic limit |
| --- | ---: | ---: | ---: |
| Selection → migration | 0.466074 | 0.466667 | 0.466667 |
| Migration → selection | 0.516240 | 0.516666 | 0.518519 |
| Neutral, SM | 0.348945 | 0.350000 | 0.350000 |
| Neutral, MS | 0.348525 | 0.350000 | 0.350000 |
| No migration, SM | 0.663525 | 0.666667 | 0.666667 |
| No migration, MS | 0.666914 | 0.665180 | 0.666667 |

With selection last, S is concave for w>1, so intermediate sampling lowers its
expected output relative to S(E[p]). Even the no-migration treatment retains a
neutral replacement stage; putting that drift before selection can change the
finite mean. Neutral SM/MS have equal means but different final variances.
The complete-migration and absent-A controls end at zero exactly.
`sample_summary.json` contains all variances, Monte Carlo standard errors,
parameters and seeds 202611130–202611137. Means are checked against seven
predeclared theoretical standard errors, not selected for a favorable sign.

## Reproduce

From the repository root, use the pinned CPython 3.14.5 and SLiM 4.0.1 CLI:

```sh
python3.14 -m examples.slimulations.migration_selection_order.demo
python3.14 -m examples.slimulations.migration_selection_order.demo --check-sample
python3.14 -m unittest discover -s examples/slimulations/migration_selection_order -t . -v
```

The demo accepts `--slim /path/to/slim`. Python uses only its standard library.
The existing [CI workflow](../../../.github/workflows/tests.yml) builds official
SLiM source revision `09fcf604a06307cf75098dfa0a2171a9b7852216` after verifying
SHA-256 `e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
All package commands run from the repository root. The raw SLiM script accepts
N, K, M, W, ORDER (0=SM, 1=MS), REPLICATES and an explicit `-s` seed.

Twenty-four tests cover independent rational enumeration of both sampling
stages, exact moments, sign reversal, Jensen correction, neutral and boundary
controls, malformed data, invalid constants, deterministic seed replay, observed
variance tolerances and saved-output replay. Focused tests are not a substitute
for all repository CI jobs.

## Scope and sources

This is a deliberately strong-parameter mechanism example, not an empirical
migration-rate estimate, equilibrium forecast, or reproduction of either paper
below. Models using viability selection, diploid dominance, migrant competition,
one sampling stage, or a different census can have different recursions.

- [Zeng & Corcoran (2015), Genetics](https://doi.org/10.1534/genetics.115.178558)
  explicitly specify fertility selection followed by reproduction and juvenile
  migration in their structured-population model. This motivates stating stage
  order; we do not reproduce their linked-selection or coalescent analysis.
- [Haller & Messer (2023), SLiM 4](https://doi.org/10.1086/723601), simulator source.
  [Official version 4.0.1 source and manuals](https://doi.org/10.5281/zenodo.7331688).

Code and prose are original and MIT licensed under the repository license.
No paper figures, upstream code, private data or unpublished work are copied.
