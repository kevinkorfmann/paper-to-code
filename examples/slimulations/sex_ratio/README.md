# Unequal breeding sex ratios at the same census size

This original SLiM experiment compares 100 males + 100 females with
20 + 180 and its 180 + 20 mirror control. Every population contains 200
diploid individuals, yet unequal numbers of breeding parents accelerate
the loss of neutral variation. A second control distinguishes the frequency
among all gene copies from the equal contribution of the two parental roles.

The classical sex-ratio benchmark is `Ne = 4*Nm*Nf/(Nm+Nf)`: 200 for the
balanced treatment and 72 for either skewed treatment. It goes back to
Wright (1931), [Evolution in Mendelian Populations](https://doi.org/10.1093/genetics/16.2.97).
For a primary modern derivation and its assumptions, see the nonoverlapping
generation case in [Engen et al. (2007)](https://doi.org/10.1111/j.1558-5646.2007.00155.x).
Here we compute **exact finite-model moments**, rather than treating that
classical benchmark as an exact one-population diversity-decay formula.
This is a mechanism demonstration, not a reproduction of either full paper.

## Reproduce

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**. There are no
Python dependencies beyond the standard library. From the repository root:

```sh
slim -v
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/slimulations/sex_ratio -v
python -m examples.slimulations.sex_ratio.demo --check-sample
```

The runner validates the SLiM version and parses both engine versions from
Eidos `version()`. Use `--slim /path/to/slim` to select an executable. The
existing [CI recipe](../../../.github/workflows/tests.yml) builds official
SLiM source commit `09fcf604a06307cf75098dfa0a2171a9b7852216` (v4.0.1),
checking archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
This is a version pin, not a claim that 4.0.1 is the newest release.

The first three treatments run **512 replicate populations for 80 generations**,
using seeds **202611010, 202611011, 202611012**, respectively. The fourth
control uses seed **202611013**, 512 replicates and one generation.
[sample_summary.json](sample_summary.json) is actual executed output;
`--check-sample` compares it byte for byte, with derived numbers rounded to
nine decimal places. Raw engine observations are integer allele and genotype
counts. A smaller exploratory run is:

```sh
python -m examples.slimulations.sex_ratio.demo --replicates 32 --generations 20
```

The fourth, deterministic control always runs for one generation. Custom
Python calls to `run_counts()` can change sex counts; both must be multiples
of four for the founder construction. An exploratory run is not expected
to match the default sample.

## Model and initialization

This is a sexual Wright–Fisher model of one neutral autosomal locus. Each
offspring receives one gamete from a uniformly sampled female and one from
a uniformly sampled male, with replacement across offspring. There is no
selfing, new mutation, migration, selection, recombination, overlapping
generation or imposed reproductive skew within a sex. Each generation has
exactly the requested counts of males and females; these counts are audited
every tick. Fixed alleles remain explicit (`convertToSubstitution=F`).

The first three treatments start with pM=pF=0.5. Within each sex, exactly
one quarter of individuals are AA, half Aa, and one quarter aa. Initialization
occurs in tick 1's late event, after that tick's reproduction, so this is
generation zero. The script creates one shared marker mutation per replicate,
audits every founder genotype count, and checks actual parental fitness=1.
The [version-matched official manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf),
sections 6.2.1–6.2.2 (printed pp. 150–151), explains `initializeSex("A")`
and the male-fraction meaning of `addSubpop()`'s sex-ratio argument.

The fourth control starts with 20 aa males and 180 AA females. Its census
allele frequency is 0.9, but every offspring has one a and one A. All 512
replicates therefore have p=0.5 and observed heterozygosity=1 after one
generation. Its gene diversity `2p(1-p)` is 0.5. This frequency change comes
from equal parental-role contributions, despite unequal sex counts; it
does not require selection or an allele-frequency estimator.

## Executed results

Generation 80, including every replicate without conditioning on fixation:

| Males / females | Classical Ne benchmark | Mean census gene diversity | Monte Carlo SE | Exact finite expectation | A fixed / lost (of 512) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 / 100 | 200 | 0.409057373 | 0.004572182 | 0.409465512 | 0 / 0 |
| 20 / 180 | 72 | 0.286820361 | 0.008216209 | 0.288699236 | 43 / 45 |
| 180 / 20 | 72 | 0.282764331 | 0.008300267 | 0.288699236 | 39 / 52 |

Here gene diversity is `2*pC*(1-pC)` within each replicate, averaged across
replicates. It is not diversity computed from the pooled mean frequency
or the fraction of heterozygous individuals. SE is the sample standard
deviation divided by sqrt(512); it describes Monte Carlo uncertainty in
this model, not uncertainty about any natural population. Independent seeds
give different observations for the mirror treatments even though their
ensemble predictions are identical.

Let `pC=(Nm*pM+Nf*pF)/(Nm+Nf)` be census frequency, and let
`z=(pM+pF)/2` weight the two parental roles equally. At generation one:

| Males / females | Observed variance of pC | Exact variance of pC | Observed variance of z | Exact variance of z |
| --- | ---: | ---: | ---: | ---: |
| 100 / 100 | 0.000664424 | 0.000625000 | 0.000664424 | 0.000625000 |
| 20 / 180 | 0.000587010 | 0.000625000 | 0.001626755 | 0.001736111 |
| 180 / 20 | 0.000629917 | 0.000625000 | 0.001686521 | 0.001736111 |

The identical census variance at generation one is not a contradiction:
all treatments initially draw from pM=pF=0.5 and produce the same number
of offspring. The smaller sex's frequency varies more and receives half
the parental weight in subsequent reproduction. Conditional on current sex
frequencies, the expected next census frequency is z; it need not equal
the current pC. In contrast, z has conserved conditional expectation.

## Independent finite-population checks

Conditional on pM,pF, the A count in a new cohort of n individuals is the
sum of independent `Binomial(n,pM)` and `Binomial(n,pF)` variables. Male
and female offspring cohorts are conditionally independent. Track
`m=E[pM], f=E[pF], a=E[pM²], b=E[pF²], c=E[pM*pF]`. Then:

```text
u = (a+b+2c)/4
T = m+f-a-b
m' = f' = (m+f)/2
a' = u + T/(4Nm)
b' = u + T/(4Nf)
c' = u
```

With `w=Nm/(Nm+Nf)`, `E[pC]=w*m+(1-w)*f` and
`E[pC²]=w²*a+(1-w)²*b+2*w*(1-w)*c`. Thus
`E[2pC(1-pC)]=2*(E[pC]-E[pC²])`. The runner evaluates this recurrence
using exact rational arithmetic before converting the output to decimals.
At generation one from pM=pF=0.5, `Var(pC)=1/(8N)` and
`Var(z)=1/(8*Ne_benchmark)`. Later exact diversity differs from
`0.5*(1-1/(2*Ne_benchmark))^t`, including a small difference for the
balanced sexual model.

The **25 tests** include an independent oracle: for Nm=1, Nf=2, enumerate
all 64 ordered paternal/maternal gamete draws for each of the 15 possible
parental allele-count states. Exact rational transition probabilities
reproduce all five moments, and propagation of the full state distribution
agrees over three generations. This small oracle does not call SLiM or
the moment recurrence to construct its transition probabilities.

Further tests cover actual founder and sex-count audits, neutral fitness,
engine versions, the deterministic opposite-fixation control, absorption,
sex symmetry, conservation of z, observed mean/diversity curves within
five Monte Carlo SEs, first-generation sampling variances, deterministic
seed repetition, parser/parameter rejection, CLI behavior and sample output.
The statistical bounds are numerical sanity checks, not preregistered tests
or a general confidence guarantee.

## Limits and licensing

This is an idealized reproductive model, not an estimate of a species'
effective size. Sex-biased reproductive success within a sex, age structure,
sex-linked loci, mate limitation, assortative mating, variable census sizes
and environmental effects are outside its scope. The binary sex labels
refer only to SLiM's modeled reproductive roles. Individual trajectories
can gain or lose diversity and eventually absorb; the expectation describes
an ensemble. No private data, biological samples or upstream example code
are included.

The original Python, Eidos and documentation are covered by the repository's
[MIT license](../../../LICENSE). SLiM is a separately installed
[GPLv3 dependency](https://github.com/MesserLab/SLiM/blob/v4.0.1/LICENSE);
its implementation is not copied or relicensed here. Cite the simulator as
Haller & Messer (2023), [SLiM 4: Multispecies Eco-Evolutionary Modeling](https://doi.org/10.1086/723601).
