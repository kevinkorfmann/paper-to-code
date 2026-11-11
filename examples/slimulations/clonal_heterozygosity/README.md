# Clonal genotype fixation need not be allele fixation

An original synthetic SLiM experiment: a population can end up with just one
diploid genotype while retaining both alleles at a locus. Cloning copies both
homologs together. If the surviving genotype is `Aa`, allele frequency is 0.5,
observed heterozygosity is 1, and genotype diversity is 0.

This differs from the repository's founder-label reproductive-skew example and
its homozygous-clone selection examples. Here the mechanism is protection of
heterozygosity *within* a diploid clone, not unequal parental fitness or changing
selection. No empirical data or private research is used.

## Experiment and checked output

Each treatment has 512 independent populations of 40 diploids, run for 320
nonoverlapping generations. The main founders are exactly 20 `aa` and 20 `Aa`,
with p(A)=0.25. Each offspring is a clone with probability c. Otherwise its two
parents and transmitted homologs are sampled independently, allowing incidental
selfing. The comparison uses c=1, 0.5 and 0. Two controls start with 40 `Aa`
individuals and use c=1 or c=0.

Actual generation-320 output from `sample_summary.json`:

| Founders and reproduction | Mean gene diversity | Monte Carlo SE | Populations with both alleles | Populations consisting only of Aa |
| --- | ---: | ---: | ---: | ---: |
| Mixed founders, all clonal | 0.247070 | 0.011059 | 253/512 | 253/512 |
| Mixed founders, half clonal | 0.006782 | 0.002316 | 10/512 | 0/512 |
| Mixed founders, all sexual | 0.003586 | 0.001411 | 10/512 | 0/512 |
| All Aa founders, all clonal | 0.500000 | 0 | 512/512 | 512/512 |
| All Aa founders, all sexual | 0.008920 | 0.002576 | 13/512 | 0/512 |

Every purely clonal mixed-founder population in this batch had become
monomorphic for a genotype: 259 were all `aa`, 253 were all `Aa`, and none
contained `AA`. Both alleles survived in those 253 all-`Aa` populations. Cloning
cannot create a genotype absent from its founders.

This is a finite batch, not a claim of certain absorption by generation 320.
The exact expected genotype diversity in that treatment is still 0.000151512.
Its exact expected gene diversity is 0.250037878, close to the long-time limit
0.25. Sexual reproduction instead has exact expected gene diversity 0.006697375
at generation 320. A single simulation batch is not used to rank the small
difference between half-clonal and sexual reproduction.

All five treatments retain every replicate, including allele loss and fixation.
The JSON includes generation 0, 1, 80, 160 and 320 measurements, per-treatment
seeds 202611110–202611114, empirical standard errors, and exact expectations.

## Three different kinds of diversity

For genotype proportions q0, q1, q2 in order `aa`, `Aa`, `AA`:

- Allele frequency: p=(q1+2q2)/2.
- Gene diversity: G=2p(1-p), the probability that two independently drawn
  gene copies differ. It is calculated *within each population*, then averaged.
- Observed heterozygosity: H=q1, the fraction of heterozygous individuals.
- Genotype diversity: J=1−q0²−q1²−q2², the probability that independently drawn
  individuals have different one-locus genotypes. This is not founder-lineage
  diversity, multilocus diversity, or a count of genealogical clones.

An all-`Aa` population has (G,H,J)=(0.5,1,0). Calling it genetically fixed without
saying *what* is fixed would hide the main result.

## Exact finite-population checks

Conditioned on the parental genotype counts, each offspring has probabilities

```
P(aa) = c q0 + (1−c)(1−p)²
P(Aa) = c q1 + (1−c)2p(1−p)
P(AA) = c q2 + (1−c)p²
```

The next generation's counts are multinomial with N draws. This gives
E[p' | parents]=p. Write h_t=E[H_t] and g_t=E[G_t]. Since the dosage of one
offspring has variance 4p(1−p)−P(Aa), the exact recursion is

```
h_(t+1) = c h_t + (1−c) g_t
g_(t+1) = (1−1/N) g_t + h_(t+1)/(2N)
E[p_t] = p_0
Var(p_t) = p_0(1−p_0) − g_t/2
```

These are finite-population ensemble expectations, not a deterministic
large-population approximation. The genotype sampling distribution also makes
E[J_t]=J_0(1−1/N)^t when c=1.

For the main clonal founders, h_t=0.5 and

```
g_t = 0.25 + 0.125(1−1/N)^t.
```

For c=0, g_t=0.375(1−1/(2N))^t. In the no-mutation finite model, pure cloning
can eventually leave all `Aa`; random mating would immediately segregate that
genotype into `aa`, `Aa` and `AA`. The all-`Aa` clonal control stays unchanged
exactly, rather than merely in expectation.

## Run and verify

Reference environment: **SLiM 4.0.1, Eidos 3.0.1, CPython 3.14.5**, standard
library only. SLiM 4.0.1 is the repository's reproducibility pin, not the latest
release. The official site currently lists SLiM 5.2. The wrapper rejects other
SLiM versions rather than silently changing the seeded experiment.

From the repository root, with the pinned executable on PATH:

```bash
slim -v
python --version
python -m unittest examples.slimulations.clonal_heterozygosity.test_demo -v
python -m examples.slimulations.clonal_heterozygosity.demo
python -m examples.slimulations.clonal_heterozygosity.demo --check-sample
```

The existing `slimulations` CI job builds the official SLiM commit
`09fcf604a06307cf75098dfa0a2171a9b7852216`, verifies the source archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`, runs the tests,
and reproduces the JSON exactly under the pinned Python version. See the root
README and `.github/workflows/tests.yml` for the complete build commands.

Tests independently enumerate parent choices, transmitted homologs and ordered
offspring using exact rational arithmetic, then propagate tiny populations for
multiple generations. They also cover closed forms, allele relabeling,
absorbing boundaries, invalid inputs, malformed output, no creation of absent
genotypes under cloning, mixed reproduction, actual SLiM version checks, seeded
replay and the complete checked output. Nothing is skipped when SLiM is absent.

The summary checks means against the oracle using six empirical Monte Carlo
standard errors plus 1/R, where R is the number of replicates. The additive floor
allows rare-event genotype diversity when a batch happens to be fully absorbed.
This is a regression guard, not a calibrated confidence interval or proof.
Exact rational enumeration provides the independent mathematical test. The
frequency standard error also has an exact finite-model calculation in the JSON.

## Assumptions and limits

One neutral diploid locus, constant census size, uniform parental sampling,
no mutation, migration, viability selection, mitotic recombination or gene
conversion, and no overlapping generations. Sexual reproduction is monoecious
random mating, not forced outcrossing. Cloning probability is an independent
per-offspring probability, not an exact fraction of offspring. Mutation IDs
are used only to encode presence of A at the locus. They are not lineage labels.

These deliberately simple conditions isolate a mechanism. Allele preservation
under complete cloning is not universal protection from drift: all-`aa` is also
an absorbing outcome, and the result depends on initial heterozygosity. Adding
mutation, loss of heterozygosity or sex changes the long-term behavior. There is
no empirical estimate of a natural population's effective size or reproduction
rate, and this is not a reproduction of an island-model paper.

## Sources

- [Balloux, Lehmann and de Meeûs (2003), *The Population Genetics of Clonal and
  Partially Clonal Diploids*, Genetics 164:1635–1644](https://doi.org/10.1093/genetics/164.4.1635).
  Background on the contrast between allele retention within heterozygous
  clones and genotypic diversity. Our isolated, no-mutation experiment and its
  finite-state checks are original, not the paper's subdivided model.
- [Haller and Messer (2023), *SLiM 4: Multispecies Eco-Evolutionary Modeling*](https://doi.org/10.1086/723601).
- [Official SLiM site and documentation](https://messerlab.org/slim/) and
  [pinned SLiM 4.0.1 source](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216).
  The WF reproduction implementation draws the number of clones binomially,
  consistent with a per-offspring cloning probability.

Sources and versions checked on 2026-11-11. Code is distributed under the
repository's MIT license.
