# Assortative mating changes heterozygosity without a directional allele-frequency shift

This original synthetic SLiM experiment separates **observed heterozygosity**
(the fraction of Aa individuals) from **gene diversity**, 2p(1-p), computed from
allele frequency. It is a one-generation mechanism demonstration, not an
inference of selection, pedigree inbreeding, or a long-run equilibrium.

## Model

Each of 512 independent populations has 200 diploid founders: 100 aa and 100 AA.
All have equal fitness. For each offspring, SLiM chooses a first parent uniformly.
The `mateChoice()` callback assigns equal weights within each parental genotype,
with total probability q for the same genotype and 1-q for the opposite genotype.
Both groups have exactly equal size, so weights q and 1-q achieve those totals.
Parents are sampled with replacement. Incidental selfing is allowed, never
forced. The mating rule refers to genotype, not kinship. Each offspring inherits
one Mendelian gamete from each homozygous parent.

- q=0.1: disassortative mating
- q=0.5: random mating
- q=0.9: assortative mating
- q=0 and q=1: deterministic heterozygosity controls

There is one neutral synthetic locus, no new mutation, migration, viability
selection, or relevant recombination. The starting p(A)=0.5 is fixed exactly.
This balanced homozygous starting condition is deliberate. Unequal genotype
frequencies or other mate-choice rules can change marginal reproductive success
and allele frequencies. The result is **not** a claim that assortative mating
always preserves allele frequency. The callback only applies at tick 2 and the
simulation stops there; it does not define a multigeneration mating model.

## Exact independent benchmarks

Offspring genotype probabilities are P(aa)=q/2, P(Aa)=1-q, P(AA)=q/2.
Thus offspring genotype counts have a multinomial distribution with N trials.
For realized allele frequency p and observed heterozygosity Hobs:

- E[p]=1/2, Var(p)=q/(4N)
- E[Hobs]=1-q, Var(Hobs)=q(1-q)/N
- E[2p(1-p)]=1/2-q/(2N)

For the gene-diversity variance, let X be one offspring's allele frequency
contribution minus 1/2. Then E[X²]=q/4 and E[X⁴]=q/16. For delta=p-1/2,
E[delta⁴]=[Nq/16+3N(N-1)(q/4)²]/N⁴ and
Var(2p(1-p))=4[E[delta⁴]-E[delta²]²].

The tests independently enumerate all ordered offspring genotype vectors for
N=4 and N=6, with five q values, using exact rational arithmetic. They check
both first and second moments, plus invalid inputs, parsing, allele-label
symmetry, deterministic endpoints, seed replay and actual SLiM runs. Six-exact-
standard-error checks are loose regression guards, not reported p-values.

## Run and verify

Pinned runtime: **SLiM 4.0.1 / Eidos 3.0.1**, **CPython 3.14.5**. Python uses only
the standard library. See the repository's pinned SLiM build in
[CI](../../../.github/workflows/tests.yml). No package or network access is needed
after installing that runtime. From the repository root, with `slim` on PATH:

```sh
python -m unittest discover -s examples/slimulations/assortative_mating -v
python -m examples.slimulations.assortative_mating.demo
python -m examples.slimulations.assortative_mating.demo --check-sample
```

The fixed seeds are 202611080–202611084, listed by treatment in
[sample_summary.json](sample_summary.json). Output is generated from actual
SLiM genotypes. Exact replay is checked in CI on the pinned runtime. All
replicates are retained; no filtering for fixation or a desired result.

## Actual checked results

| Mating | Mean observed heterozygosity | Exact expectation | Mean allele frequency | Mean gene diversity |
|---|---:|---:|---:|---:|
| Disassortative | 0.900654 | 0.9 | 0.500356 | 0.499758 |
| Random | 0.501211 | 0.5 | 0.500605 | 0.498893 |
| Assortative | 0.099727 | 0.1 | 0.496895 | 0.497741 |
| Opposite only | 1 | 1 | 0.5 | 0.5 |
| Same only | 0 | 0 | 0.500703 | 0.497614 |

Expected allele frequency is 0.5 in every treatment, but realized means fluctuate.
The exact standard error of the assortative treatment's mean frequency is
0.001482. An approximately ninefold change in observed heterozygosity need not
represent a comparable loss of allelic variation. The small gene-diversity
changes here are finite-sample effects, quantified above.

## Sources and scope

- Crow & Felsenstein (1968), [The effect of assortative mating on the genetic
  composition of a population](https://doi.org/10.1080/19485565.1968.9987760),
  provides classical background on assortative mating and its distinction from
  inbreeding. This demo is not a reproduction of that paper's polygenic model.
- Haller & Messer (2023), [SLiM 4: Multispecies Eco-Evolutionary
  Modeling](https://doi.org/10.1086/723601), documents the simulation framework.

The code and derivations here are original, under the repository's MIT license.
No empirical or private data, natural-population estimates, or human mate-choice
inferences are used.
