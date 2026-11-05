# Patterson D: copied sites are not independent evidence

This original example calls **scikit-allel 1.3.13** to compute Patterson's
four-population D statistic and its block-jackknife standard error. It uses
only deterministic, constructed biallelic allele counts, not empirical data or
a demographic simulation. No random operation or seed is used.

## Estimand, inputs and sign convention

The target is **normalized allele-sharing asymmetry D(A,B;C,D)** for a specified
four-population comparison. It is a test statistic for a tree/clade null under
appropriate population-genetic assumptions, not a gene-flow proportion, event
date, or donor-to-recipient direction. This demonstration does not establish
that admixture happened and reports no p-values.

Input shape is `(sites, four populations, REF/ALT counts)`. The original panel
contains eight sites and **20 haploid chromosomes per population**. Six sites
have ALT counts `[20,0,20,0]` and two have `[0,20,20,0]`. The second panel
repeats each original row ten times, contiguously. Each group of ten is an
exact copy of one original record, so it contains no new independent evidence.
An additional eight-site polymorphic panel exercises fractional frequencies.
All input definitions are in [demo.py](demo.py) and the ALT counts are recorded
in [sample_output.json](sample_output.json).

For frequencies `a,b,c,d`, the actual tool computes per-site components:

```
numerator   = (a-b)*(c-d)
denominator = (a+b-2*a*b)*(c+d-2*c*d)
D = sum(numerator)/sum(denominator)
```

The numerator and denominator have unbiased component estimators under the
model. Their ratio is not generally finite-sample unbiased. Do not replace
the ratio of sums with the mean of site ratios.

With population D fixed for the ancestral allele and the population order
used here, this is **BABA minus ABBA**, not the opposite convention sometimes
used for D. Swapping A/B changes the sign. Reversing REF/ALT labels consistently
in every population leaves the statistic unchanged. A positive sign alone does
not identify a gene-flow direction.

## Actual output

| Input and block definition | Records | Blocks | D | Jackknife SE | Z = D/SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original rows, one row per block | 8 | 8 | 0.5 | 0.327326835 | 1.527525232 |
| Ten copies per original, copy group per block | 80 | 8 | 0.5 | 0.327326835 | 1.527525232 |
| Same copies, incorrectly one row per block | 80 | 80 | 0.5 | 0.097435470 | 5.131601439 |

Nothing about the original data changed. The larger Z is manufactured by
the independence assumption. This is a controlled arithmetic example of
pseudoreplication, **not a model claiming real linkage consists of exact
copies**. In a real analysis, blocks need to reflect genomic dependence.
The tool's `blen` counts variants, not base pairs or genetic distance.
Eight artificial blocks do not justify asymptotic significance calibration.

For the original eight units, the leave-one-out estimates are `3/7` for each
of the six positive rows and `5/7` for each of the two negative rows. Using
`SE² = (B-1)/B * sum((D_leave_block_out - mean_leave_block_out)²)` gives exactly
`SE²=3/28`. Keeping copy groups together preserves those leave-one-out
values. Treating 80 copies as separate units gives `SE²=3/316`.

All-monomorphic input has a zero denominator and an undefined D. The output
records this as JSON `null`, not zero. A homogeneous informative panel has
zero jackknife SE and undefined Z, also recorded as `null`. Neither condition
should be interpreted as decisive evidence.

## Assumptions and limits

- Population samples and allele definitions must be appropriate and aligned.
  Distinct populations supply independent allele draws for the component
  calculation. Genotype error, contamination and sample overlap can invalidate
  this model. The example does not infer phase or relatedness.
- An admixture interpretation depends on the specified population tree and
  historical assumptions. Ancestral structure, a wrong topology, ascertainment,
  and technical artifacts can complicate or confound it.
- Block-jackknife uncertainty needs enough approximately independent genomic
  blocks. These constructed arrays demonstrate the numerical calculation,
  not biological calibration, confidence intervals, or real-genome accuracy.
- The strict wrapper rejects missing populations, negative/noninteger counts,
  and incomplete final blocks. It requires at least two complete equal-size
  blocks, avoiding silent loss of trailing rows in moving-window aggregation.
  Passing these checks does not validate a biological sampling design.
- D is not an admixture fraction. A separate tested construction has
  `A=alpha*C+(1-alpha)*B`, `B=D=0`, `C=1`: both `alpha=1/4` and `3/4` give
  D=1. This is an algebraic identifiability diagnostic, not realistic demography.

## Reproduce

Use CPython **3.12.11** and the existing hash-locked environment shared with
the Hudson-FST and Tajima-D examples:

```sh
python3.12 -m venv .venv-patterson
. .venv-patterson/bin/activate
python -c "import platform; assert platform.python_version() == '3.12.11'"
python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
python -m unittest discover -s examples/inference_showcase/patterson_blocks -v
python -m examples.inference_showcase.patterson_blocks.demo --check-sample
```

Recorded versions: CPython **3.12.11**, scikit-allel **1.3.13**, NumPy **2.3.5**.
The CLI emits JSON rounded to nine decimals and checks it byte for byte.
There are **26 tests**, including independent exact rational enumeration of
all 16 allele-draw patterns for each of **353 count configurations** across
one to three chromosomes per population. A separate exact jackknife oracle
checks the polymorphic panel. Tests cover duplication, population/allele swaps,
block order, ratio-of-sums aggregation, count scaling, undefined values, strict
input validation, nonmutation of inputs, CLI execution and checked output.
The existing scikit-allel CI job runs this example with its locked environment.

## Primary sources

- Patterson et al. (2012), *Ancient admixture in human history*, especially
  D statistics and ascertainment, pp. 1071–1073:
  https://doi.org/10.1534/genetics.112.145037
- Durand et al. (2011), *Testing for Ancient Admixture between Closely Related Populations*,
  including ancestral-structure limitations:
  https://doi.org/10.1093/molbev/msr048
- Official API documentation:
  https://scikit-allel.readthedocs.io/en/stable/stats/admixture.html#allel.average_patterson_d
- Version-pinned implementation:
  https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/admixture.py

All example code and explanatory text are original under this repository's
MIT license. scikit-allel is installed separately under its own MIT license.
No upstream figures, source files, or empirical genetic data are redistributed.
