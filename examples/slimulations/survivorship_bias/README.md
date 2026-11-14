# Neutral survivors can look as though they were favored

Original educational SLiM simulation, not a reproduction or empirical estimate.
A neutral marker starts in 2 of 20 clonal individuals (p=0.1). All 2,048 replicate
populations are retained. After 30 generations the all-replicate mean is 0.096558,
while the mean among the 329 populations still containing A is 0.601064.
No individual has a fitness advantage. Removing extinct alleles changes the
question being answered.

## Model and exact prediction

Each generation draws N clonal offspring independently and uniformly from N
parents. Marker carriers are AA, others aa; both homologs are cloned together.
Thus the sampling units are N individuals, **not 2N independent gametes**.
There is no mutation, recombination, migration, selection, sex, or population-size
change. Substitution conversion is disabled so fixed markers remain observable.

For carrier count K, K' conditional on K=i is Binomial(N,i/N). `distribution`
propagates the complete (N+1)-state transition matrix. Expected frequency stays
p0: E[p_t]=p0. If S_t=P(K_t>0), then

    E[p_t | K_t>0] = p0 / S_t.

This finite-generation identity includes fixed alleles among survivors. It is
not the mean conditional on still segregating (0<K_t<N), nor a trajectory
conditioned on eventual fixation. Because zero is absorbing, S_t cannot increase,
and the expected survivor-conditioned frequency cannot decrease. Individual
surviving trajectories can still decline.

For the rare treatment at generation 30, exact S=0.1568543134 and the conditional
mean is 0.6375342688. The unconditional mean is exactly 0.1 with Monte Carlo SE
0.0058747719 over 2,048 replicates. The observed 329 survivors include 100 fixed
populations and 229 segregating ones. Their separate segregating-only mean is
0.4268558952 (exact conditional expectation 0.4749161801). Finite-run ratios
fluctuate and are not forced to match their expectations.

Controls start at p=0.9, zero and one. Both absorbing boundaries remain fixed.
Checkpoints at generations 0, 5, 15, 30 use the same simulated trajectories;
they are not independent samples. Replicates, not generations, are sampling units.
No future-selection inference, genome-wide claim or permanent diversity implied.

## Reproduce

From repository root, use CPython 3.14.5 and official SLiM 4.0.1 on PATH (the
repository reproducibility pin, not a claim about latest release). See the root
workflow for the checksum-pinned SLiM source build. Python uses standard library only.

```sh
python -m unittest examples.slimulations.survivorship_bias.test_demo -v
python -m examples.slimulations.survivorship_bias.demo
python -m examples.slimulations.survivorship_bias.demo --check-sample
```

Four treatments use seeds 202611140–202611143. Actual output and exact predictions
are in `sample_summary.json`. The parser rejects missing/duplicate/out-of-range
rows, wrong founders, and escape from absorption. Tests include independent
parent-tuple enumeration, two-step rational probabilities, martingale and variance
identities, reflection, boundary/undefined conditional means, and actual SLiM replay.
Synthetic data only; no external source code copied.

## Sources

- Biddanda, Rice & Novembre (2020), [A variant-centric perspective on geographic
  patterns of human allele frequency variation](https://doi.org/10.7554/eLife.60107),
  Appendix 2: neutral conditional-survival frequency argument. This example isolates
  one finite population, not the paper's geographic analysis.
- Haller & Messer (2023), [SLiM 4](https://doi.org/10.1086/723601).
