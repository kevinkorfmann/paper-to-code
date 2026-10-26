# Hudson F_ST: estimate differentiation, then check the aggregation

**2026-10-26 · Inference showcase · real scikit-allel 1.3.13**

This original example calls `allel.hudson_fst` and
`allel.average_hudson_fst` on synthetic allele counts. It estimates
**pairwise Hudson differentiation for a specified locus panel** and a
delete-block jackknife standard error. It does **not** infer a split time,
migration rate, ancestry, selection, or a demographic history.

The useful trap: averaging per-locus F_ST ratios is not the same estimator
as dividing the summed numerators by the summed denominators. A constructed
rare-variant example makes that difference large.

## Input and estimand

[`synthetic_counts.json`](synthetic_counts.json) contains two scenarios, each
with 1,200 preselected biallelic loci and 80 independently sampled **haploid
chromosomes** per population. Each count row is `[reference, alternate]`;
rows identify the same loci in the same order in both populations.

We repeat baseline alternate frequencies `0.2, 0.3, 0.4, 0.5, 0.6, 0.7` 200
times. The null has `p_A = p_B = baseline`; the differentiated scenario has
`p_A = baseline - 0.15`, `p_B = baseline + 0.15`. Independent binomial draws
use NumPy's explicit PCG64 generator and **seed 20261026**. No loci are filtered
after sampling, and no real genetic data, reference genome, simulator,
downloaded dataset, or private research is used.

For sampled alternate frequency `p_i` and called chromosome count `n_i`, the
biallelic finite-sample components are

```text
numerator_l   = (p_A - p_B)^2
                - p_A(1-p_A)/(n_A-1) - p_B(1-p_B)/(n_B-1)
denominator_l = p_A(1-p_B) + p_B(1-p_A)
estimated F_ST = sum(numerator_l) / sum(denominator_l)
```

The actual inference uses scikit-allel, **not** our independent formula above.
Tests evaluate that formula separately with exact rational arithmetic.
Simulation probabilities and the population target are evaluation-only
metadata; `estimate()` receives counts and block size only.

On this deliberately chosen panel, the ratio of population-frequency moments
is `sum((p_A-p_B)^2) / sum(p_A(1-p_B)+p_B(1-p_A))`: **0** for the null and
**54/289 = 0.186851211…** for the differentiated scenario. This is a
panel-specific target, not a universal genome-wide parameter. The finite-data
ratio estimator is not exactly unbiased.

## Run

Use **CPython 3.12.11** in a separate environment, from the repository root:

```sh
python3.12 -m venv .venv-allel
. .venv-allel/bin/activate
python -c "import platform; assert platform.python_version() == '3.12.11'"
python -m pip install --no-cache-dir --require-hashes \
  -r examples/inference_showcase/scikit_allel_fst/requirements.lock
python -m unittest discover -s examples/inference_showcase/scikit_allel_fst -v
python -m examples.inference_showcase.scikit_allel_fst.demo --check-sample
```

The complete transitive package set is hash-locked. Main versions are
scikit-allel **1.3.13**, NumPy **2.3.5**, and Dask **2026.8.0**. Dask is a
package dependency; this example uses local NumPy arrays, not a cluster or a
paid service. The implementation and synthetic fixtures are original and
MIT-licensed; installed dependencies retain their own licenses.

[`sample_summary.json`](sample_summary.json) is actual execution output,
rounded to nine decimal places. `--check-sample` compares the complete JSON
byte for byte, including reference versions. Regenerate the input independently:

```sh
python - <<'PY'
import json
from examples.inference_showcase.scikit_allel_fst.demo import FIXTURE, generate_input
assert generate_input() == json.loads(FIXTURE.read_text())
PY
```

## Actual checked output

| Scenario | Population target | Estimated F_ST | Block-jackknife SE |
| --- | ---: | ---: | ---: |
| No frequency differentiation | 0 | -0.000204354 | 0.000438112 |
| Frequencies separated by 0.30 | 0.186851211 | 0.189656390 | 0.001956434 |

Negative finite-sample estimates are retained, not clipped to zero. The
null also has 819 negative per-locus estimates. Equal **observed** frequencies
do not force a zero estimate because of the finite-sample correction.

We use 20 equal blocks of 60 loci. Each block contains ten copies of every
baseline-frequency class, with independent draws. The reported SE measures
sampling uncertainty under this construction; it is not a posterior standard
deviation, a guaranteed confidence interval, or a coverage-validation result.
Changing to `--block-size 120` keeps the point estimate but changes the SE.

### Constructed aggregation counterexample

The separate deterministic panel has 20 common loci with counts `[80,20]`
versus `[20,80]`, then 180 rare loci with `[99,1]` versus `[100,0]`.

| Included loci | Ratio of sums | Mean of per-locus ratios |
| --- | ---: | ---: |
| 20 common loci only | 0.524658348 | 0.524658348 |
| Plus 180 rare loci | 0.463334645 | 0.052465835 |

Each constructed rare locus has numerator zero but denominator 0.01.
Giving every locus's ratio equal weight greatly lowers the mean. The
ratio-of-sums also changes when the panel changes; it is not immune to
ascertainment. These repeated counts illustrate arithmetic, **not** a
real-data result or a model of how rare variants generally behave. We do not
attach a biological standard error to this constructed panel.

## Assumptions and limitations

- The sampled chromosomes and loci are independent. Haploid allele counts
  require neither phasing nor a Hardy–Weinberg assumption here. Relatedness,
  inbreeding, linkage, population-label errors, and genotype errors are absent.
- Block-jackknife uncertainty relies on suitably independent blocks. Sixty
  loci is a convenient size for this synthetic design, not a validated block
  length for real genomes; variant count is not genetic distance.
- The wrapper deliberately accepts only aligned biallelic nonnegative integer
  count arrays, at least two calls per population/locus, and at least two
  complete equal blocks. Missing/single-call rows are rejected rather than
  silently dropped; general real-data QC is outside scope.
- Shared monomorphic loci have zero numerator and denominator. Their
  individual ratios are undefined, but they contribute zero to the pooled
  sums. An entirely shared-monomorphic panel or an uninformative jackknife
  block is rejected explicitly.
- This is method-of-moments differentiation estimation from a small
  frequency-defined fixture, not a coalescent simulation, selection scan,
  reproduction of the cited paper, or empirical performance benchmark.
- scikit-allel is in maintenance-only mode. Its stable documentation banner
  says 1.3.3; this example pins and tests the actual **1.3.13** release and
  links its versioned source below.

## Verification and sources

The **22 tests** cover exact input regeneration, changed/invalid seeds,
genuine package invocation, an independent rational component calculation,
an independent delete-block jackknife, null/positive controls, population
swap and allele relabeling, negative estimates, fixed differences,
monomorphic loci, invalid/missing counts, block validation and sensitivity,
truth isolation, the aggregation counterexample, CLI errors, and exact output.
The repository CI adds an isolated hash-locked job for these checks.

- Bhatia, Patterson, Sankararaman and Price (2013),
  [*Estimating and interpreting F_ST: The impact of rare variants*](https://doi.org/10.1101/gr.154831.113),
  Genome Research 23:1514–1521: estimator choice, locus aggregation, and
  ascertainment. The named Hudson estimator follows Hudson et al. (1992).
- [Official F-statistics API documentation](https://scikit-allel.readthedocs.io/en/stable/stats/fst.html)
  describes allele-count inputs, Hudson components and the block jackknife.
- [Exact scikit-allel 1.3.13 implementation](https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/fst.py)
  and [release metadata](https://pypi.org/project/scikit-allel/1.3.13/).
- Original [example code](demo.py), [tests](test_demo.py), and
  [dependency lock](requirements.lock).
