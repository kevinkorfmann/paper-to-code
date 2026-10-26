# slimulations: recombination erodes linkage disequilibrium, not instantly

**Free recombination within meiosis does not necessarily erase an association
across the population in one generation.** This original two-locus SLiM
experiment starts with excess `AB` and `ab` haplotypes and follows their mixing.
With random-pairing founder proportions, recombination fraction `c=0.5` halves
signed linkage disequilibrium **in the infinite-population model**; it does not
immediately set it to zero. A second founder configuration shows why that
qualification matters.

## Model and controls

- Actual **SLiM 4.0.1 / Eidos 3.0.1**, with a **CPython 3.14.5** standard-library
  wrapper. These are reproducibility pins, not claims about current versions.
- Diploid Wright–Fisher model, **N=1000**, nonoverlapping generations, uniform
  independent parental draws, hermaphrodites, and incidental selfing permitted.
- Two neutral biallelic markers; no new mutation, migration, selection, gene
  conversion, sex differences, or spatial structure. Census size is constant.
- **128 isolated replicate populations** per treatment; eight generations at
  `c=0`, `0.1`, or `0.5`. Replicates use one seeded pseudorandom stream per run.
- The initial diplotype counts are `250 AB/AB`, `500 AB/ab`, `250 ab/ab`.
  These are exact random-haplotype-pairing proportions, not randomly sampled
  founders. Thus `p(A)=p(B)=0.5`, `x_AB=x_ab=0.5`, and initial `D=0.25`.
- A **one-generation founder control** instead makes all 1000 individuals
  `AB/ab`. Its initial haplotype counts and D are identical, but its diploid
  pairing is different.

The sites occupy adjacent SLiM positions 0 and 1, so there is exactly one
recombination interval. `initializeRecombinationRate(c)` specifies its actual
breakpoint probability, including at `c=0.5`; this is **not** a Poisson map
length needing conversion. The pinned
[official API documentation](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/QtSLiM/help/SLiMHelpFunctions.html)
and [engine implementation](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/chromosome.cpp#L430-L470)
explain that interpretation. Positions are an **abstract two-marker map**, not a
claim that realistic adjacent biological base pairs recombine at 50%.

Two shared neutral mutation objects per population label A and B. They are
installed after tick 1 reproduction; that observation is **t=0**. Tick 9 is
t=8. `convertToSubstitution=F` keeps alleles countable if fixed. All populations,
including any absorbed ones, remain in the summaries.

## Prediction and the founder-pairing caveat

Using haplotype frequencies in the order `(AB, Ab, aB, ab)`:

```text
p_A = x_AB + x_Ab
p_B = x_AB + x_aB
D   = x_AB - p_A*p_B = x_AB*x_ab - x_Ab*x_aB
```

D is a signed association among alleles on the **same haplotype**. It is not
`r²`, normalized `D′`, observed heterozygosity, or an association between the
two homologs of an individual. In an infinite randomly paired population,
Mendelian segregation and recombination give

```text
x_AB_next = x_AB - c*D       x_Ab_next = x_Ab + c*D
x_aB_next = x_aB + c*D       x_ab_next = x_ab - c*D
D_next = (1-c)*D            D_t = D_0*(1-c)^t
```

Recombination changes associations without changing the marginal allele
frequencies in this deterministic model. Finite populations also drift.

**The actual parental diplotypes matter.** Let `n_C` count coupling double
heterozygotes `AB/ab`, and `n_R` count repulsion double heterozygotes `Ab/aB`.
Only these individuals change haplotypes through recombination. Conditional on
these parental counts, the exact gamete probabilities are

```text
delta = c*(n_C-n_R)/(2*N)
q_AB = x_AB-delta           q_Ab = x_Ab+delta
q_aB = x_aB+delta           q_ab = x_ab-delta
```

For the main founders `n_C=N/2`, so `q_AB=.5-c/4`. For the all-`AB/ab` control
`n_C=N`, so `q_AB=(1-c)/2`. At `c=.5`, their gametic D values are respectively
`.125` and `0`, despite identical initial population D. Random choice of
parents does not retroactively randomize the homologs inside those parents.

There is also a finite-sampling distinction. Write `D(q)=q_AB-q_A*q_B` and
`M=2N`. The independent offspring gamete draws imply

```text
E[D_next | fixed parental diplotypes] = (1-1/M)*D(q).
```

This follows because `E[p_A_next*p_B_next]=q_A*q_B+D(q)/M`.
The JSON reports that exact **first-generation** expectation separately from
the infinite-population trajectory. Later parental pairing fluctuates; the
infinite recursion is not asserted to be an exact finite-population ensemble
mean. For the no-recombination control specifically, only `AB` and `ab` exist,
so `D=p(1-p)` and the exact drift expectation is
`E[D_t]=.25*(1-1/(2N))^t`: even without recombination, mean D is not constant.

## Actual checked output

All values below come from [sample_summary.json](sample_summary.json), not
hand-entered expectations. D is computed **within each population, then
averaged**; pooling first would mix within- and between-population associations.

| Founder pairing | c | Seed | Mean D, t=1 | Mean D, t=8 | MC SE of D, t=8 | Infinite-population D, t=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random-pairing proportions | 0 | 202610260 | 0.249867229 | 0.248994209 | 0.000122349 | 0.250000000 |
| Random-pairing proportions | 0.1 | 202610261 | 0.224907953 | 0.107046527 | 0.000749381 | 0.107616803 |
| Random-pairing proportions | 0.5 | 202610262 | 0.124534877 | 0.001381783 | 0.000656428 | 0.000976562 |
| All AB/ab; one generation only | 0.5 | 202610263 | 0.000196912 | — | — | — |

The alternative-founder t=1 MC SE is **0.000500199**, and its exact mean-D
prediction is zero. The three main exact t=1 mean-D predictions, including
finite sampling, are **0.249875**, **0.2248875**, and **0.1249375**. Free
recombination's main t=1 result is therefore clearly not an immediate loss of
population association. At t=8 its remaining mean is small relative to initial
D and the Monte Carlo variation; it is not evidence of a nonzero equilibrium.

MC SE is the sample standard deviation across populations divided by
`sqrt(128)`. It describes simulation variation under these chosen parameters,
not uncertainty in real biological recombination rates. Individual trajectories
need not be monotone, and a small signed mean does not imply every population
has zero D or zero squared association.

## Reproduce

From the repository root, with the pinned `slim` on PATH:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
slim -v
python -m unittest discover -s examples/slimulations/linkage_disequilibrium -v
python -m examples.slimulations.linkage_disequilibrium.demo > /tmp/ld-summary.json
diff -u examples/slimulations/linkage_disequilibrium/sample_summary.json /tmp/ld-summary.json
```

No third-party Python package, GPU, paid API, or external data is needed.
`--slim /path/to/slim` chooses an existing engine; other versions are rejected.
`--replicates`, `--generations`, and `--seed` allow exploration. The duration
option changes only the three main treatments; the alternative-founder control
always runs one generation. One replicate reports a null MC SE rather than an
invented uncertainty estimate.

The [CI recipe](../../../.github/workflows/tests.yml) builds the official CLI
from source revision `09fcf604a06307cf75098dfa0a2171a9b7852216`, verifying archive
SHA-256 `e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
This build uses a temporary directory and does not replace system tools.
Exact output is checked on the pinned reference platform; different SLiM
versions, compilers, or platforms may use different random streams.

The original [Eidos script](recombination.slim) can run directly:

```sh
slim -s 202610262 -d C=0.5 -d N=1000 -d REPLICATES=128 \
  -d GENERATIONS=8 -d FOUNDERS=0 \
  examples/slimulations/linkage_disequilibrium/recombination.slim
```

`FOUNDERS=0` means exact random-pairing proportions; `FOUNDERS=1` means all
`AB/ab`. N must be a multiple of four. Tagged `DATA` rows contain
`elapsed_generation,replicate_id,n_AB,n_Ab,n_aB,n_ab,n_coupling,n_repulsion`.
The wrapper rejects missing or duplicate observations, checks nonnegative
counts, haplotype totals, and double-heterozygote bounds, and verifies all
founder counts. These checks do not constitute a general diplotype-feasibility
solver for arbitrary externally supplied rows.

## Tests, sources, and limits

The **20 tests** in [test_demo.py](test_demo.py) cover:

- Independent exact-rational enumeration of homolog choice and crossover,
  both for arbitrary actual diplotypes and for random-pairing frequencies.
- Signed D, its determinant identity, geometric decay, preserved marginals,
  linkage-equilibrium and no-recombination controls, and the founder caveat.
- Exhaustive enumeration of small offspring samples to verify the finite-D
  bias independently of the simulation and recurrence code.
- Actual SLiM founder timing/counts; first-generation haplotype probabilities
  across both founder modes and all three c values; and second-generation
  probabilities conditional on the **realized stochastic parents**.
- Simultaneous Hoeffding checks for the haplotype means (alpha `1e-8` per
  family; 24 first-generation and 12 second-generation comparisons), plus
  separate six-MC-SE mean-D diagnostics. These SE diagnostics are not advertised
  as calibrated simultaneous significance tests. The fixed trajectory checks
  establish the intended qualitative contrast, not exact agreement with the
  deterministic curve.
- No creation of missing haplotypes at c=0, the exact finite no-recombination
  drift mean, within-population rather than pooled D, reproducible seeds,
  engine/version and input validation, parser failures, CLI, and checked JSON.

**Primary references:**

1. H. Tachida (1994), [*Decay of linkage disequilibrium in a finite island
   model*](https://doi.org/10.1017/S0016672300032742), *Genetical Research*
   64:137–144. The introduction on p.137 states the infinite random-mating
   decay factor `1-r`. We use that classical baseline, **not** the paper's
   structured-population extension. Its finite-island formulas are not treated
   as exact formulas for our diploid-parent sampling scheme.
2. R. B. Robbins (1918), [*Some Applications of Mathematics to Breeding
   Problems III*](https://doi.org/10.1093/genetics/3.4.375), *Genetics* 3:375–389,
   is the historical two-locus reference. The algebra above and the enumeration
   tests state this example's derivation explicitly.
3. B. C. Haller and P. W. Messer (2023),
   [*SLiM 4: Multispecies Eco-Evolutionary Modeling*](https://doi.org/10.1086/723601),
   plus the [official 4.0.1 release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1)
   and pinned API/engine sources linked above.

This is a small synthetic mechanism demonstration, not a named organism,
realistic chromosome, demographic inference, or full-paper reproduction.
Mutation, migration, selection, mating structure, and biological recombination
maps can change LD dynamics. The original script, wrapper, tests, and prose are
[MIT licensed](../../../LICENSE); separately installed SLiM retains its GPL
license. No upstream recipes, paper figures, private research, or real genetic
data are redistributed.
