# Neutral drift: correlated frequencies, uncorrelated disjoint changes

Run one neutral marker in 2,048 independent diploid populations of size 20.
All start at exactly 10 marked copies out of 40, or p₀ = 0.25. We census all
genomes at 0, 6, 12 and 18 generations, retaining populations where the marker
is lost or fixed. SLiM uses independent parent draws, including incidental
selfing, and fair Mendelian inheritance; there is no migration, mutation,
selection or observation sampling.

| Covariance (allele frequency squared) | Simulation | Monte Carlo SE | Wright–Fisher |
| --- | ---: | ---: | ---: |
| Levels at generations 6 and 12 | 0.027682190 | 0.001012501 | 0.026424694 |
| Changes over disjoint intervals 0–6 and 6–12 | 0.000385132 | 0.000538370 | 0 |
| Changes over overlapping intervals 0–12 and 6–18 | 0.021681213 | 0.001135976 | 0.022700617 |

Although neutral frequency levels share past drift, the expected future change
conditional on the entire observed past is zero. As a result, disjoint true
frequency changes are uncorrelated, not necessarily independent. Overlapping
changes have covariance equal to the variance accumulated in their shared
interval (6–12 in this case). Across replicate populations, frequency levels
at times t ≤ u have covariance Var(pₜ); subtracting the common fixed p₀ does
not change it.

With q = 1 − 1/(2N), neutral Wright–Fisher sampling gives
Var(pₜ) = p₀(1−p₀)(1−qᵗ). The covariance of two changes with shared
interval a–b is Var(p_b) − Var(p_a). To estimate it here we average the
per-population product of the corresponding changes, whose means are known
to be zero. The SE is the sample standard deviation of those products
divided by the square root of the number of independent replicates.
At the final census 603 markers were lost and 17 fixed.

The zero expectation is for actual frequencies in this null model, not noisy
estimates. An independent error at one census enters its adjacent changes
with opposite signs, creating negative covariance. Ascertainment, selection,
migration and finite samples need separate treatment. No genome-wide
selection effect is simulated or estimated.

## Reproduce

From the repository root, put SLiM **4.0.1** on PATH and run with
**CPython 3.14.5** (standard library only):

```sh
python -m unittest examples.slimulations.neutral_temporal_covariance.test_demo -v
python -m examples.slimulations.neutral_temporal_covariance.demo
python -m examples.slimulations.neutral_temporal_covariance.demo --check-sample
```

Default seed: `20261122`. The checked summary is produced by the actual
simulation. Unit tests also enumerate exact binomial transitions in a separate
small Wright–Fisher chain for all 100 interval pairs, test loss and
fixation, compare Monte Carlo estimates with the model null, and check that
malformed or incomplete census data are rejected.

## References

- [Haller & Messer, SLiM 4 (2023)](https://doi.org/10.1086/723601).
- [Buffalo & Coop, linked selection and temporal covariance (2019)](https://doi.org/10.1534/genetics.119.302581).
- [Buffalo & Coop, temporal allele frequency changes (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7456072/).
