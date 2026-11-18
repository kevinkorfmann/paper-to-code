# Equal allele frequency does not imply equal bottleneck-loss risk

An original SLiM 4.0.1 / Eidos 3.0.1 nonWF experiment with CPython 3.14.5 standard-library analysis. Forty diploids carry eight marked chromosome copies, so allele frequency is 0.1. Exactly five adults survive a uniformly random, genotype-independent cull without replacement. There is **no reproduction** before measurement.

If the eight copies are spread across eight heterozygotes, the allele disappears in **1,223/4,096 populations (29.86%)**. If the same eight copies are clustered in four homozygotes, it disappears in **2,327/4,096 (56.81%)**. Exact probabilities are **30.60%** and **57.29%**, with Monte Carlo standard errors of 0.72 and 0.77 percentage points. The expected frequency among survivors remains 0.1 in both cases; measured means are 0.100464 and 0.100000.

The survival draw samples whole diploid individuals. Losing a homozygous carrier loses both copies together. This is not selection against homozygotes, a fitness advantage of heterozygosity, or a reproductive Wright–Fisher bottleneck. The deliberately contrasting founder genotype configurations are conditioned upon, not claimed to be typical Hardy–Weinberg populations. There is no general claim that heterozygosity always improves persistence after subsequent reproduction.

## Exact finite-population calculation

Write H for heterozygotes, D for marked homozygotes, U=N-H-D for unmarked homozygotes, and S for adult survivors. The joint survivor genotype counts (h,d,u) have multivariate hypergeometric probability

`P(h,d,u) = choose(H,h) choose(D,d) choose(U,u) / choose(N,S)`, where `h+d+u=S`.

The probability of losing the marker is `choose(U,S)/choose(N,S)` (zero when U<S). It depends on the **number of carriers**, H+D, not solely the copy count H+2D. The mean surviving allele frequency is `(H+2D)/(2N)`. If X is dosage 0,1,2 in a uniformly chosen founder and its population variance is V, the survivor-frequency variance is `V (N-S) / [4 S (N-1)]`.

`genotype_distribution` evaluates that exact law. Independent tests enumerate every labelled subset for populations of 2–8 individuals and all valid genotype/survivor counts using rational probabilities. Other tests check the mean, variance, loss probability, boundaries, malformed output, seeded replay and live SLiM census controls. Seven-SE checks on simulation summaries are broad diagnostic tolerances, not calibrated hypothesis tests.

Six seeded treatments (202611180–202611185) include spread, clustered, an intermediate mixture (four heterozygotes plus two marked homozygotes), no cull, absent marker and fixed marker. All 4,096 replicate populations enter each summary, including losses. Founder and survivor genotype counts and census sizes are checked. Mutation, migration, recombination, selection and reproduction are disabled. Neutral marker copies may have different SLiM mutation IDs; only marker-type presence per homolog defines dosage in this single-site experiment.

## Reproduce

From the repository root, use the existing checksum-pinned official SLiM 4.0.1 build in `.github/workflows/tests.yml`, and CPython 3.14.5 (`.python-version`). These are reproducibility pins, not latest-version claims. Put `slim` on PATH:

```
python -m unittest examples.slimulations.carrier_bottleneck.test_demo -v
python -m examples.slimulations.carrier_bottleneck.demo
python -m examples.slimulations.carrier_bottleneck.demo --check-sample
```

The demo accepts `--slim /path/to/slim`. `sample_summary.json` is actual seeded output; the replay check compares all fields. No extra Python dependencies. The simulation takes seconds.

## Sources and scope

- Haller & Messer (2023), *SLiM 4: Multispecies Eco-Evolutionary Modeling*: https://doi.org/10.1086/723601 . Version-matched [manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf) for the nonWF life cycle, genotype access and survival via `fitnessScaling`.
- Allendorf et al. (2024), *What does effective population size tell us about loss of allelic variation?*: https://doi.org/10.1111/eva.13733 . Context for distinguishing allelic loss from heterozygosity and specifying sampling assumptions. This original conditioned-adult-survival construction does not reproduce that paper's reproductive model or estimate effective population size.

Original code under the repository MIT license. No third-party implementation or empirical/private data copied.
