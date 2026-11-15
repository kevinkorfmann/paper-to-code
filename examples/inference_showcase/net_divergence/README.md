# Between-population differences are not a split clock

A small scikit-allel 1.3.13 example of summary statistics used in population-genetic inference. It estimates pairwise differences, not a demographic model or a divergence time. All inputs are original, constructed biallelic chromosome counts; no private data or random generator is used.

## Estimand and inputs

At each aligned site, each population supplies `[reference_count, alternate_count]`. The arrays must share allele labels and site order. There must be at least two observed chromosomes per population and site. This deliberately bounded tutorial rejects missing/negative/fractional counts.

`mean_pairwise_difference_between` estimates the probability that one chromosome from each population differs (`dxy`). `mean_pairwise_difference` estimates the within-population probability using distinct pairs (`pi_a`, `pi_b`). We define the net comparison as `dxy - (pi_a + pi_b)/2`.

For biallelic population frequencies p and q, these population probabilities are p(1-q)+(1-p)q, 2p(1-p), and 2q(1-q). Their net comparison is (p-q)^2. With independent random chromosome samples, the sample estimators have these expectations. That is a sampling statement, not a claim about a particular observed table.

## Actual output

Each row below represents one synthetic site, not a genomic per-base estimate.

| Counts A; B | dxy | pi_a = pi_b | net |
|---|---:|---:|---:|
| [5,5]; [5,5] | 0.500000000 | 0.555555556 | -0.055555556 |
| [10,0]; [0,10] | 1.000000000 | 0.000000000 | 1.000000000 |
| [10,0]; [10,0] | 0.000000000 | 0.000000000 | 0.000000000 |
| [8,2]; [2,8] | 0.680000000 | 0.355555556 | 0.324444444 |

Identical allele counts still give substantial between-population differences because variation is shared. Their net estimate is -1/18 because within-population comparisons exclude self-pairs. This finite-sample negative value is not negative elapsed time and should not be silently clipped.

Conditioning on identical observed counts differs from independent sampling from identical population frequencies. An exact enumeration of two independent Binomial(10, 1/2) samples gives E[dxy]=1/2 and E[net]=0. For frequencies 1/5 and 4/5, enumeration gives E[dxy]=17/25 and E[net]=9/25.

## Reproduce

From the repository root, use Python 3.12.11 and the existing isolated, hash-locked environment:

```sh
python3.12 -m venv .venv-net-divergence
.venv-net-divergence/bin/python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
.venv-net-divergence/bin/python -m unittest discover -s examples/inference_showcase/net_divergence -v
.venv-net-divergence/bin/python -m examples.inference_showcase.net_divergence.demo --check-sample
.venv-net-divergence/bin/python -m examples.inference_showcase.net_divergence.demo
```

The checked JSON records Python 3.12.11, scikit-allel 1.3.13 and NumPy 2.3.5. Seeds: not applicable; the calculation and expectation enumeration are deterministic. Eighteen tests include independent explicit chromosome-pair enumeration across 1,089 count combinations, exact binomial expectation controls, allele/population/site symmetries, invariant and fixed-difference sites, invalid inputs and checked output. Duplicating chromosomes leaves dxy unchanged but changes the finite-sample within-population correction; copies are not new independent observations.

## Assumptions and limits

Independent, representative chromosome sampling supports the expectation interpretation. Relatedness, population structure, genotyping errors, allele misalignment, and nonrandom missingness can invalidate that interpretation. Across-site dependence matters for uncertainty; this example provides no confidence interval or significance test. It does not infer selection, ancestry, migration, mutation rate, or split time. A demographic interpretation requires additional assumptions and a fitted model.

For genomic Dxy, sum site differences and divide by an appropriate accessible-base denominator, including invariant callable positions. Variant-only averages and unequal callability can change the estimand. This example intentionally avoids window aggregation and missing-data corrections.

## Sources

- [Official scikit-allel diversity documentation](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.mean_pairwise_difference_between): definitions, count inputs and accessible-base normalization.
- [scikit-allel 1.3.13 diversity implementation](https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/diversity.py): the two library estimators used here.

The net-statistic derivation, fixtures and tests in this directory are original tutorial code, not a reproduction of an empirical result.
