# Tajima's D: an ascertainment filter can reverse the sign

This original example calls **scikit-allel 1.3.13**, rather than presenting a
home-made formula as the tool. It constructs allele counts for **20 haploid
chromosomes**, with no randomness, real genetic data or private research.

## Target and input

The target is **Tajima's D**, the standardized contrast between two estimates
of the population mutation parameter: total mean pairwise differences and
Watterson's `S/a1`. It is a summary/test statistic, not an estimate of selection,
population growth or a uniquely identified demographic history. The example
reports no p-value or confidence interval.

The input is a biallelic integer matrix `[reference count, alternate count]`.
The main panel has 80 singleton rows `[19,1]` and 20 intermediate-frequency
rows `[10,10]`. These are deliberate constructions, **not simulations from a
neutral or selected population**. The retained panel is derived from those
same rows by requiring minor allele count >= 2. It is not independently generated.
All fixture definitions are in `fixtures()` in [demo.py](demo.py).

## Actual output

| Panel | Segregating sites | Total pairwise differences | Watterson theta | Tajima D |
| --- | ---: | ---: | ---: | ---: |
| All constructed sites | 100 | 18.526315789 | 28.186961182 | -1.408065350 |
| Same panel, minor count >= 2 | 20 | 10.526315789 | 5.637392236 | 3.295285566 |
| 100 singleton sites | 100 | 10.000000000 | 28.186961182 | -2.650799074 |
| 100 intermediate sites | 100 | 52.631578947 | 28.186961182 | 3.562869547 |

The sign flips just by dropping rare variants. No change in population history
was introduced. A positive or negative D on an ascertained SNP panel cannot be
interpreted as a causal diagnosis from its sign alone.

Reversing reference/alternate labels or adding 200 monomorphic sites preserves
D to numerical tolerance. The latter does **not** mean the per-base mutation
parameter is unchanged: these are region-total estimates, with no accessible
sequence length supplied. With fewer than three segregating sites, the tool's
explicit `min_sites=3` policy returns NaN, represented as JSON `null`, never zero.
The full observed output is [sample_output.json](sample_output.json).

## Method and assumptions

For `n` complete chromosomes, define `a1=sum(1/i)` and `a2=sum(1/i²)` for
`i=1,...,n-1`. With `S` segregating sites and ALT count `k` at each site:

```
pi_total = sum(2*k*(n-k)/(n*(n-1)))
theta_W_total = S/a1
b1 = (n+1)/(3*(n-1))
b2 = 2*(n*n+n+3)/(9*n*(n-1))
c1 = b1-1/a1
c2 = b2-(n+2)/(a1*n)+a2/a1**2
e1 = c1/a1
e2 = c2/(a1*a1+a2)
D = (pi_total-theta_W_total)/sqrt(e1*S+e2*S*(S-1))
```

The classic null interpretation assumes a random sample from a neutral,
constant-size panmictic population under the infinite-sites model; the classic
variance uses the no-recombination model. Recombination, structure, changing
population size, selection, sequencing error and frequency-dependent ascertainment
can change interpretation or calibration. Neither the constructed spectra nor
the sign of D validate these assumptions. We do not use normal-tail thresholds.

The wrapper intentionally accepts only equal row totals of at least four
chromosomes and biallelic counts. The tool itself uses the maximum
row sum for `n`, while per-site pairwise diversity uses each site's called
count. We reject unequal row totals rather than silently treating them as a constant
sample. Completeness is guaranteed by the constructed fixtures; allele counts
alone cannot establish that the same complete sample was called at every site.
The counts do not specify haplotype linkage or phase, and this demonstration
does not infer either.

## Reproduce

Use CPython **3.12.11** and the already maintained, hash-locked scikit-allel
environment shared with the separate Hudson-FST showcase:

```sh
python3.12 -m venv .venv-tajima
. .venv-tajima/bin/activate
python -c "import platform; assert platform.python_version() == '3.12.11'"
python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
python -m unittest discover -s examples/inference_showcase/tajima_ascertainment -v
python -m examples.inference_showcase.tajima_ascertainment.demo --check-sample
```

Pinned output versions are scikit-allel **1.3.13**, NumPy **2.3.5**, and
CPython **3.12.11**. There is no seed because no random operation is performed.
Output is rounded to nine decimals and compared byte for byte. The existing
scikit-allel CI job runs both examples using the same locked environment.

The **26 tests** include an independent rational-arithmetic implementation,
explicit enumeration of chromosome pairs, all **996 unordered three-site
segregating spectra for n=4 through 12**, sign reversal, allele/site-order
invariance, monomorphic padding, minimum-site behavior, input rejection,
nonmutation of inputs, CLI execution and checked output. The rational reference
is a numerical check of the statistic, not an independent biological model.

## Primary sources

- Tajima (1989), *Statistical method for testing the neutral mutation hypothesis
  by DNA polymorphism*: https://doi.org/10.1093/genetics/123.3.585
- Official scikit-allel API documentation:
  https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.tajima_d
- Version-pinned implementation:
  https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/diversity.py

The code and explanation are original under this repository's MIT license.
No upstream source, figures or empirical genetic datasets are redistributed.
scikit-allel retains its own MIT license and is installed separately.
