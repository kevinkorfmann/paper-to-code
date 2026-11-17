# Partial turnover: less drift per tick is not less drift per generation

Original SLiM 4.0.1 / Eidos 3.0.1 nonWF experiment, CPython 3.14.5 standard library. A neutral population of N=40 homozygous clones initially has K=20 marked individuals. In one update, M newborn clones draw their parents independently and uniformly, with replacement, from the entire old census. Independently, exactly M distinct old individuals are sampled uniformly for death. Parents may die after reproduction. Newborns cannot die in that update. Census size returns to N.

Replacing 5 individuals gives mean squared frequency change **0.001467**, whereas replacing all 40 gives **0.006207**, across 4,096 independent populations per treatment. Exact predictions are 0.001482 and 0.006250. Replacing just one gives 0.000316 (expected 0.0003125). Allele frequency remains unbiased in expectation.

This is a **per-update** comparison: updates with different replacement counts are not equal reproductive generations. The M=1 case is a discrete neutral birth–death step; M=N is a clonal Wright–Fisher replacement. Intermediate M batches are simultaneous, not M sequential Moran events: newborns cannot become parents during the batch. Do not multiply the one-step variance by N/M to assert an exact generation-scale variance; allele frequencies and conditional variances evolve. There is no empirical effective-population-size estimate here.

## Model and exact check

No mutation, migration, sex, recombination or selection. All replicates count. The sampling unit is an individual clone, not 2N independent chromosome copies. For p=K/N, deaths D follow Hypergeometric(N,K,M), births B follow Binomial(M,p), independently, and K'=K-D+B. Thus

E[p'-p] = 0

E[(p'-p)^2] = p(1-p) [M + M(N-M)/(N-1)] / N².

The two terms are birth sampling and death sampling. At M=1 the expression is 2p(1-p)/N²; at M=N it is p(1-p)/N. `distribution` enumerates this finite count transition and its fourth moment supplies the Monte Carlo SE of the measured squared change. Seven-SE checks are diagnostic tolerances, not calibrated hypothesis tests. Independent tests enumerate labelled death subsets and ordered parent choices with rational probabilities, rather than reuse the formula.

The SLiM callback runs once globally at tick 2, clones parents drawn from a saved old-census vector, and sets exactly M old individuals' survival to zero. Others have survival one. Tick 1 and tick 2 census counts and homozygosity are checked. Six treatments cover M=1,5,40,0 and allele-absent/fixed controls at M=5. Seeds 202611170–202611175 are recorded in `sample_summary.json`. Constant founder counts are not sampling estimates. No age-dependent fertility, mortality, density regulation or natural-population claim is included.

## Reproduce

From repository root, use the existing checksum-pinned official SLiM 4.0.1 build in the CI workflow and CPython 3.14.5 (`.python-version`). These are reproducibility pins, not latest-version claims. Put `slim` on PATH:

```
python -m unittest examples.slimulations.partial_turnover.test_demo -v
python -m examples.slimulations.partial_turnover.demo
python -m examples.slimulations.partial_turnover.demo --check-sample
```

The demo also accepts `--slim /path/to/slim`. No additional Python packages. Full run takes seconds. Tests cover exact microstate enumeration, moments, complement symmetry, endpoint models, malformed output, impossible transitions, neutral fixed/absent controls and seeded replay. Actual output is saved, not an illustrative forecast.

## Sources

- Moran (1958), *Random processes in genetics*: https://doi.org/10.1017/S0305004100033193 . Historical birth–death model motivation. This is an original finite-batch experiment, not a reproduction of the paper's full analysis.
- Haller & Messer (2023), *SLiM 4*: https://doi.org/10.1086/723601 . The version-matched [SLiM manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf) documents the nonWF life cycle, `reproduction`, `addCloned` and survival via `fitnessScaling`.

Original code under the repository MIT license; no third-party implementation copied.
