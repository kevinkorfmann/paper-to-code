# Observation noise is not faster drift

Original one-generation SLiM 4.0.1 / Eidos 3.0.1 experiment, CPython 3.14.5 standard library. N=40 homozygous clonal individuals, one neutral marker, 4,096 independent populations per treatment, no mutation, migration, selection, sex or recombination. Uniform parental draws create binomial clonal Wright–Fisher offspring. The sampling unit is an individual, not 2N independent chromosomes. All replicates count.

At each census, observe S individuals uniformly **without replacement**, independently at the two times. Observations are nondestructive: all N individuals remain eligible to reproduce. Initial true frequency is fixed at K/N, not estimated from the first sample. Five seeded runs use S=5,20,40 at K=20 and absent/fixed controls at S=5. Sampling consumes random numbers, so treatments do not share identical biological paths. Within each run, sampled and full-census changes use the same populations.

## Run

From the repository root, install the repository's pinned official SLiM build (see the SLiM CI job), put `slim` on PATH, and use CPython 3.14.5:

```
python -m unittest examples.slimulations.temporal_sampling.test_demo -v
python -m examples.slimulations.temporal_sampling.demo
python -m examples.slimulations.temporal_sampling.demo --check-sample
```

The CLI also accepts `--slim /path/to/slim`. No Python dependencies. Seeds 202611150–202611154 are recorded in `sample_summary.json`. Run time is seconds for this example. Tests include exhaustive subset enumeration, a rational enumeration of all small-population offspring/sample paths, complement symmetry, invalid rows, fixed/absent controls and exact seed replay.

## Exact target

Let p=K/N, C be offspring marker count, and X0,X1 be observed frequencies. Conditional on the fixed founder census, C~Binomial(N,p). Each observation count is hypergeometric given its census. The reported statistic is **mean squared temporal change**, not sample variance with an estimated mean and not an Ne estimate.

The census MSD is p(1−p)/N. For a sample of S clones from N with frequency q, conditional sampling variance is q(1−q)(N−S)/[S(N−1)]. Since E[p1(1−p1)]=p(1−p)(1−1/N), independent observation errors yield

E[(X1−X0)^2] = p(1−p)/N + p(1−p)(N−S)(2−1/N)/[S(N−1)].

The cross terms vanish under the specified fixed-census, random nondestructive sampling model. `delta_distribution` independently enumerates offspring binomial probabilities and both hypergeometric observations; its fourth moment supplies Monte Carlo SEs for the MSD. The script checks observed MSDs within seven SEs of the formula (a diagnostic tolerance, not a calibrated hypothesis test).

## Actual output

At S=5, sampled MSD is 0.098457 versus census MSD 0.006283 in those same populations. Exact expectations are 0.094872 and 0.006250; sampled-MSD Monte Carlo SE is 0.001998. S=20 gives sampled MSD 0.018472 (expected 0.018910). Full census S=40 gives 0.006258 (expected 0.006250). Absent and fixed controls remain exactly zero.

Small observation samples add noise; they do not alter the expected biological drift rate. Sampling the entire census removes this observation component. These numbers are not universal corrections: destructive sampling, overlapping generations, sample-dependent reproduction, sexual diploidy, related samples, sequencing noise, selection and migration require their own models. No empirical population size or significance claim is made. This does not implement Waples' estimator or reproduce his results.

## Sources

- Waples (1989), *A Generalized Approach for Estimating Effective Population Size from Temporal Changes in Allele Frequency*: https://doi.org/10.1093/genetics/121.2.379 . Context for separating gene sampling from drift; our finite clonal model and derivation are specified above.
- Haller & Messer (2023), *SLiM 4*: https://doi.org/10.1086/723601 . SLiM 4.0.1 is the repository reproducibility pin, not a claim about the latest version.

Original code under the repository MIT license; no third-party implementation copied.
