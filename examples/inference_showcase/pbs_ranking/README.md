# Population branch statistic: normalization can reorder windows

**2026-11-22 · Inference showcase · scikit-allel 1.3.13**

With identical allele counts and windows, scikit-allel's default `allel.pbs`
places a different window first than `normed=False`. The default is PBSn1,
not simply PBS times a common scale.

## Synthetic panel and estimand

Twelve handwritten biallelic SNPs have complete counts in three populations
(A, B, C), with 20 haploid chromosomes per population per SNP. All twelve
SNPs appear in `ALT` in [demo.py](demo.py), and the output records the
counts used in each window. No seed, sampling, genomic coordinates,
haplotype phase, biological measurements or external input are involved.
Windows are four consecutive nonoverlapping blocks of **three variants**
(not fixed physical spans). A is focal, B and C are comparators.

The estimand here is the original PBS or normalized PBSn1 score for the
supplied windows and populations. The real library uses a ratio of sums for
each window's pairwise Hudson FST, clips FST to [0, 0.99999], then computes
T_ij = −log(1 − clipped FST_ij). It returns:

```text
PBS_A   = (T_AB + T_AC − T_BC) / 2
PBSn1_A = PBS_A / [1 + (T_AB + T_AC + T_BC)/2]
```

We call the native API for both statistics and separately call native
`moving_hudson_fst` to display pairwise inputs before clipping. The
independent test oracle counts pair differences using exact rational
arithmetic (without replacement within each population, across pairs
between populations). It sums the within- and between-population components
across sites before transforming distances.

## Actual result

| Window | Raw PBS A | Default PBSn1 A | Normalizer |
| --- | ---: | ---: | ---: |
| High background | 0.623637052 | 0.255984030 | 2.436234213 |
| Low background | 0.546289936 | 0.353290753 | 1.546289936 |
| Nonadditive | −0.523949641 | −0.219769216 | 2.384090234 |
| Equal counts | 0 | 0 | 1 |

The first row wins under raw PBS. The second wins after normalization:
the first row's larger total pairwise differentiation gives it a larger
divisor. That normalization changes the score across windows; it does
not alter the input observations.

In the nonadditive row, every pairwise FST is positive. Nevertheless,
T_BC exceeds T_AB + T_AC and the focal score is negative. These distances
cannot be represented as three nonnegative branches of a three-tip tree.
This is distinct from the fourth row, where identical counts across
populations yield finite-sample FST = −1/19, which the library clips to
zero before computing the scores. Negative PBS is not a negative time.

## Run and verify

From the repository root using **CPython 3.12.11**, **scikit-allel 1.3.13** and
**NumPy 2.3.5**:

```sh
python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
python -m unittest discover -s examples/inference_showcase/pbs_ranking -v
python -m examples.inference_showcase.pbs_ranking.demo --check-sample
```

Use an isolated virtual environment. Eleven tests cover input shape,
independent exact pairwise calculations, independent score calculations,
the default option, rank reversal, separate negative-score and
FST-clipping controls, joint REF/ALT swaps, swapping the two comparators
and byte-for-byte CLI replay.

The constructed panels are a diagnostic, not a calibrated outlier scan,
power comparison or validation on real genomes. Real interpretation
depends on the chosen comparator populations, allele alignment, windowing,
ascertainment, linked variation and demographic background. No calibrated
significance, confidence interval, split time or evidence of selection
is inferred.

Sources: [official scikit-allel PBS API](https://scikit-allel.readthedocs.io/en/stable/stats/selection.html#allel.pbs),
[version-tagged implementation](https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/selection.py),
[Yi et al. 2010, original PBS](https://pmc.ncbi.nlm.nih.gov/articles/PMC3711608/),
[Malaspinas et al. 2016, PBSn1 supplement](https://www.eva.mpg.de/documents/Nature/Malaspinas_Genomic_Nature_2016_Suppl_2351078.pdf).
