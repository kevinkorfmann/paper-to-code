# A rare-allele advantage without a best heterozygote

An original one-generation SLiM experiment shows selection changing direction
with allele frequency. The heterozygote always has intermediate fitness, so this
is not overdominance. The fitness rule is explicitly frequency-dependent; no
spatial heterogeneity, migration, environmental fluctuation or mating preference
is needed for this demonstration.

## Model and exact prediction

There are N=100 hermaphroditic diploids at one biallelic locus, with random mating,
independent parent draws and Mendelian segregation. Incidental selfing is allowed.
There is no mutation, migration, recombination or density regulation beyond the
fixed Wright–Fisher census. Each treatment has 1,024 independent populations.

For parental allele frequency p and s=0.8, put a=s(1−2p). Genotype reproductive
weights are w(aa)=1−a, w(Aa)=1 and w(AA)=1+a. They are strictly positive for
0≤s<1. Fitness is recalculated after setting the founders and checked directly
against SLiM's cached individual fitnesses. At p=0.1 the weights are
(0.36,1,1.64); at p=0.9 their order reverses.

Given founder counts (n0,n1,n2), the probability a sampled gamete carries A is

```
q = [n1/2 + n2(1+a)] / [n0(1−a) + n1 + n2(1+a)].
```

This expression uses the actual founder genotype counts and does not require
Hardy–Weinberg equilibrium. Independent gametes give offspring genotype
probabilities ((1−q)^2,2q(1−q),q^2). Thus E[p_next]=q and
Var[p_next]=q(1−q)/(2N). The script compares simulated replicate means with these
exact conditional expectations, not with the result of another simulation.

## Actual checked output

| Treatment | Founder counts aa,Aa,AA | Seed | Mean p after one generation | Exact expectation |
| --- | --- | --- | ---: | ---: |
| Rare A | 81,18,1 | 202611120 | 0.218774 | 0.218033 |
| Intermediate A | 25,50,25 | 202611121 | 0.500986 | 0.500000 |
| Common A | 1,18,81 | 202611122 | 0.779966 | 0.781967 |
| Neutral rare A, s=0 | 81,18,1 | 202611123 | 0.100771 | 0.100000 |
| Absent A | 100,0,0 | 202611124 | 0 | 0 |
| Fixed A | 0,0,100 | 202611125 | 1 | 1 |

Monte Carlo standard errors of the mean are 0.000912 for rare/common selected A,
0.001105 for intermediate A and 0.000663 for neutral rare A. All replicates count.
The JSON records exact unrounded output, parameters, versions and analytical
standard errors. Seven-standard-error checks are regression guards, not a
hypothesis-testing procedure or a claim about natural populations.

## Reproduce

Use **CPython 3.14.5**, **SLiM 4.0.1 / Eidos 3.0.1**, and the Python standard
library. The repository CI builds a checksum-verified official SLiM revision.
From the repository root, with SLiM on PATH:

```sh
python -m unittest discover -s examples/slimulations/frequency_dependence -v
python -m examples.slimulations.frequency_dependence.demo
python -m examples.slimulations.frequency_dependence.demo --check-sample
```

The CLI also accepts `--slim /path/to/slim`. The Python `run` function exposes
founder counts, strength, seed and replicate count. No input data download or
third-party Python package is needed.

## Tests and limitations

Twenty tests cover independent rational enumeration over individual parental
pairs and Mendelian gametes, exact binomial moments, allele relabeling,
neutrality, the 1/2 fixed point, absence of overdominance, invalid inputs and
malformed output, seed replay, absent/fixed controls, non-HWE founders,
heterozygote segregation, direct Eidos validation and full sample reproduction.

This illustrates the **direction of one generation of selection**, not the
long-run equilibrium distribution or duration of polymorphism. A frequency of
1/2 is an expected fixed point, not a fixed stochastic outcome. Finite-population
drift remains present. If an allele is already absent, selection cannot recreate
it without mutation or migration. Fitness strength and linear dependence were
chosen for illustration; this does not infer an ecological mechanism or estimate
selection in any species. It is original teaching code, not a reproduction of a
published dataset or a novel biological result.

## Sources

- [Ayala and Campbell (1974), Frequency-Dependent Selection](https://doi.org/10.1146/annurev.es.05.110174.000555): classic conceptual background, not the source of these particular parameters.
- [Haller and Messer (2023), SLiM 4](https://doi.org/10.1086/723601): simulation framework.
- [Official SLiM 4.0.1 source and manual](https://github.com/MesserLab/SLiM/tree/v4.0.1): Wright–Fisher life cycle, `fitnessEffect()`, and `recalculateFitness()`.

All code and synthetic settings here are original; upstream code and data are
not redistributed.
