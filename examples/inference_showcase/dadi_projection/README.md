# dadi projection: fewer segregating sites, same expected pairwise diversity

An original deterministic example using `dadi.Spectrum.project`, an input-processing
operation for demographic inference. It is not a fitted population history.

## Target and synthetic input

The target is the expected site-frequency spectrum (SFS) in a uniformly chosen
sample of 4 distinct chromosomes from 20, conditional on observed allele counts.
The main input contains 80 singleton sites and 20 sites with derived count 10.
All 100 are initially segregating. No random draw or random seed is used.

Actual checked output with dadi 2.4.4:

| Quantity | Original 20 chromosomes | Projected 4 chromosomes |
|---|---:|---:|
| Expected segregating sites | 100 | 34.266253869969 |
| Pairwise difference total | 18.526315789474 | 18.526315789474 |

A panel of 100 singleton sites contributes 20 expected segregating sites after
projection. This is not a particular sample containing exactly 20 variants.
Projection averages possible subsamples; fractional expected site counts are valid.

## Independent calculation

For a site with derived count k among n chromosomes, the probability of j derived
copies in an m-chromosome sample is

    choose(k,j) * choose(n-k,m-j) / choose(n,m).

The implementation checks dadi against exact rational combinations. Tests also
independently enumerate labeled chromosome subsets rather than calling this formula.
All site mass is conserved when the invariant corners are included. Masking those
corners removes sites expected to be invariant in the smaller sample; it does not
remove their contribution to pairwise diversity, which is already zero.

The pairwise statistic sums `2*j*(m-j)/(m*(m-1))` over the spectrum. A uniformly
chosen pair in a uniformly chosen smaller sample is still a uniformly chosen pair
from the original sample. Thus its conditional expected total is preserved.
This is a region total, not diversity per base: no callable sequence length is supplied.

A deliberately different control drops invariant outcomes and renormalizes the
remaining spectrum to 100 variants. This changes the target and inflates the pairwise
total. It is not the down-projection implemented by dadi.

## Assumptions and limits

Biallelic aligned, error-free chromosome calls and uniformly sampled chromosomes
are assumed. Counts are polarized, though the pairwise-diversity identity does not
require knowing the ancestral allele. Real projection does not repair nonrandom
missingness or ascertainment. Each site's marginal expectation is sufficient here;
there is no assumption that linked sites provide independent likelihood evidence.
No confidence interval, demographic parameter, selection test, or accuracy claim
is supplied. A constructed allele-count panel is not a simulated demographic sample.

## Reproduce

Use CPython 3.12.11 and the existing hash-locked dadi environment, from repo root:

```sh
python3.12 -m venv .venv-dadi
.venv-dadi/bin/python -m pip install --require-hashes -r examples/inference_showcase/dadi_size_change/requirements.lock
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dadi-projection-mpl
.venv-dadi/bin/python -m unittest examples.inference_showcase.dadi_projection.test_demo -v
.venv-dadi/bin/python -m examples.inference_showcase.dadi_projection.demo
.venv-dadi/bin/python -m examples.inference_showcase.dadi_projection.demo --check-sample
```

Twenty tests cover exhaustive one-site projections for n=2..12, independent subset
enumeration for n=2..8, exact pairwise identities for n=2..14, fixed/absent controls,
mass and first moment conservation, sequential projection, allele reversal,
linearity, invalid inputs and saved output. `sample_output.json` records actual
versions, inputs, expected spectra and output. No dependency implementation is copied.

## Sources

- [Official dadi projection documentation](https://dadi.readthedocs.io/en/latest/user-guide/manipulating-spectra/#projection).
- [Gutenkunst et al. (2009), Inferring the Joint Demographic History of Multiple Populations from Multidimensional SNP Frequency Data](https://doi.org/10.1371/journal.pgen.1000695).

The source describes the tool and projection; the constructed panels and controls
here are an original educational example, not a reproduction of the paper's analyses.
