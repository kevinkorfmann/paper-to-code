# Pairwise missingness can make signed LD matrices invalid as correlations

This constructed example estimates signed Rogers–Huff LD, r, from
unphased diploid biallelic genotype dosages (0, 1, 2 ALT alleles). The
[scikit-allel API](https://scikit-allel.readthedocs.io/en/stable/stats/ld.html#allel.rogers_huff_r)
returns condensed pairwise r. Its
[pinned 1.3.13 implementation](https://raw.githubusercontent.com/cggh/scikit-allel/v1.3.13/allel/opt/stats.pyx)
skips a sample for a pair whenever either dosage is negative. The underlying
method is described by [Rogers & Huff (2009)](https://doi.org/10.1534/genetics.108.093153).

Three markers are observed in different groups of eight dedicated diploids
per pair and four additional diploids complete at all markers. The dedicated
blocks contribute signed dosage correlation +1 for AB, −1 for AC, and +1
for BC. The four common diploids have pairwise orthogonal centered dosages.
The genotype mask is deliberately structured for this counterexample.

| Input to the same estimator | Pairs' sample counts | r(AB), r(AC), r(BC) | Smallest eigenvalue |
| --- | ---: | --- | ---: |
| Masked, deleting missing calls per pair | 12, 12, 12 | +0.666667, −0.666667, +0.666667 | −0.333333 |
| Same four complete diploids for all pairs | 4, 4, 4 | 0, 0, 0 | 1 |
| Known complete genotype matrix before masking | 28, 28, 28 | +0.4, −0.4, +0.4 | 0.2 |

Each pairwise estimate is a valid correlation on its own overlap. Together
they do not define a correlation matrix from one set of genotypes: the
assembled unit-diagonal matrix gives a negative quadratic form for
(1, −1, 1), which no positive-semidefinite correlation matrix can do.
Identical overlap counts do not help when the sample identities differ.

These r estimates use genotype dosage correlations as a proxy for gametic
LD under the method's assumptions; the construction is not a sample from
a particular mating model. It illustrates a matrix algebra problem,
not a bug in individual pair estimates. It does not show that complete-case
analysis recovers truth under nonrandom missingness: here it changes the
sample to just four individuals and estimates zero. The known complete
matrix is a fixture-only counterfactual, not a recommendation to impute
missing calls to heterozygotes. No r² inference, uncertainty, population
history or real-data performance is claimed.

## Reproduce

Use CPython **3.12.11**, scikit-allel **1.3.13**, and NumPy **2.3.5**, as
pinned by the repo's existing
[`scikit_allel_fst/requirements.lock`](../scikit_allel_fst/requirements.lock).
No input downloads or seeds: all dosages are defined in the example.

```sh
python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
python -m unittest examples.inference_showcase.ld_pairwise_missing.test_demo -v
python -m examples.inference_showcase.ld_pairwise_missing.demo
python -m examples.inference_showcase.ld_pairwise_missing.demo --check-sample
```

The tests independently compute pair-specific covariance and variance with
rational arithmetic, audit sample overlaps and check the negative quadratic
form and the complete-data controls. Replay is tested byte for byte.
