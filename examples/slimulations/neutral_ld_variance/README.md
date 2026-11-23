# Drift creates LD without a preferred sign

At a pair of synthetic neutral markers, each founder population has exactly
one quarter of each phased haplotype: AB, Ab, aB and ab. Its starting
linkage disequilibrium, D = p(AB) − p(A)p(B), is exactly zero. In one
Wright–Fisher generation, all 2N offspring haplotypes are inherited via
independent parental and Mendelian draws, with incidental selfing allowed.
There is no mutation, recombination, fitness difference, migration or
subsampling. The offspring counts are the population census.

| Diploids (N) | Mean signed D (Monte Carlo SE) | Mean D² (Monte Carlo SE) | Exact E[D²] |
| ---: | ---: | ---: | ---: |
| 10 | 0.001121826 (0.001192015) | 0.002909842 (0.000086712) | 0.002968750 |
| 40 | 0.001168594 (0.000630220) | 0.000814387 (0.000026317) | 0.000771484 |

Each size has 2,048 independent simulated populations. In the smaller case,
1,007 populations have positive D, 961 negative D and 80 exactly zero D.
The expected signed mean is zero for both sizes, yet D² is positive on
average and larger at smaller N. All outcomes, including populations with
a monomorphic locus or D = 0, contribute; D is not normalized to r².

## Exact one-generation control

Let n = 2N, and let vectors X and Y record the two allele indicators
in each transmitted haplotype. They are independent Bernoulli(1/2) vectors
under the specified founder counts and independent transmission. With
M = I − 11ᵀ/n, D = XᵀMY/n is an empirical covariance with denominator n.
Because M centers both vectors and is idempotent with trace n−1,
E[D] = 0 and E[D²] = (n−1)/(16n²). The denominator is n, not n−1.
The test suite independently enumerates exact multinomial count
probabilities for n = 1 through 12, computing D² by its frequency
definition instead of the implementation's haplotype determinant.

This is a one-generation null with a controlled founder state, not a
drift–recombination equilibrium prediction, a long-run LD estimator, or a
test for selection. Independence is a statement about the *transmitted
copies before observing their realized counts*, not a claim that every
offspring census has independent observed allele frequencies.

## Reproduce

From the repo root, with SLiM **4.0.1** on PATH and **CPython 3.14.5**
(standard library only):

```sh
python -m unittest examples.slimulations.neutral_ld_variance.test_demo -v
python -m examples.slimulations.neutral_ld_variance.demo
python -m examples.slimulations.neutral_ld_variance.demo --check-sample
```

Seeds: 20261123 for N = 10, 20261124 for N = 40. The test suite audits
the exact founder counts and every offspring census and checks simulation
moments against the mathematical null and Monte Carlo uncertainty.

- [Hill & Robertson, Linkage disequilibrium in finite populations (1968)](https://pubmed.ncbi.nlm.nih.gov/24442307/)
- [Haller & Messer, SLiM 4 (2023)](https://doi.org/10.1086/723601)
