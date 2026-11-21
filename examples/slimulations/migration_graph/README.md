# Same incoming migration, different arrival time

A neutral marker begins fixed in deme 0 and absent elsewhere. Compare four
equal diploid Wright–Fisher demes (40 individuals each) joined in either a
ring or a complete graph. In both cases a juvenile's parental source is its
own deme with probability 0.7, with total incoming fraction 0.3 divided
uniformly between the two ring neighbors or the three other demes. This is
SLiM's backward parent-source convention, not a count of adults emigrating.
There is no new mutation, selection, or recombination.

The exact expected frequency vector advances by applying the source
probability matrix to the current vector. Because each graph's matrix is
doubly stochastic and deme sizes are equal, the expected pooled frequency is
0.25 in both. This does not fix the realized pooled frequency in a replicate.

| Graph | Opposite deme, generation 1 (expected / observed) | Generation 2 (expected / observed) |
| --- | ---: | ---: |
| Ring | 0 / 0 | 0.045 / 0.043311 |
| Complete | 0.10 / 0.102246 | 0.16 / 0.159521 |

The ring cannot introduce the allele to the opposite deme in one generation;
at two generations there are two paths with probability 0.15 squared each.
This is a finite-step experiment, not an estimate of equilibrium
differentiation, real dispersal, or per-deme selection.

## Reproduce

From the repository root, on Linux with SLiM 4.0.1 on PATH and CPython 3.14.5
(standard library only):

```sh
python -m unittest examples.slimulations.migration_graph.test_demo -v
python -m examples.slimulations.migration_graph.demo
python -m examples.slimulations.migration_graph.demo --check-sample
```

The runs use seeds 202611211 (ring) and 202611212 (complete), with 256 isolated
replicate metapopulations per graph. Generation 0 is the initialized late tick
1, before dispersal affects the marker. The driver checks all 24 sample means
against the analytical values with a tolerance of six replicate-based Monte
Carlo standard errors plus 0.002. The tests independently enumerate the first
two exact transitions and check constant expectation through generation 30.
Output includes observed deme means, Monte Carlo SEs, and pooled means.

[SLiM 4.0.1 source and manuals](https://doi.org/10.5281/zenodo.7331688) ·
[Haller & Messer, SLiM 4](https://doi.org/10.1086/723601)

[Kimura & Weiss (1964), stepping-stone models](https://doi.org/10.1093/genetics/49.4.561) provide historical background; this is a distinct four-deme finite-step toy.
