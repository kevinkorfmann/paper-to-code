# VCFtools: exact Hardy–Weinberg tests and discrete calibration

This November 2, 2026 inference showcase calls **VCFtools 0.1.17 `--hardy`**
on original, constructed diploid genotype counts. It independently verifies
the resulting hypothesis tests with exact rational combinatorics.

The inferential target is **compatibility with Hardy–Weinberg genotype
proportions at one biallelic autosomal locus, conditional on the observed
allele count and number of called individuals**. This is statistical hypothesis
testing, not estimation of a demographic history or inbreeding coefficient.
A p-value is neither the probability that HWE is true nor a diagnosis of why
genotypes depart from it.

## Run

From the repository root, with **CPython 3.14.5**, a C++ compiler, `make`,
`curl`, and zlib headers available (tested on macOS and Ubuntu CI):

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -m examples.inference_showcase.vcftools_hwe.install
python -m unittest discover -s examples/inference_showcase/vcftools_hwe -v
python -m examples.inference_showcase.vcftools_hwe.demo --check-sample
```

The installer downloads the official release archive, verifies SHA-256
`b9e0e1c3e86533178edb35e02c6c4de9764324ea0973bebfbb747018c2d2a42c`,
and builds only its C++ CLI under ignored `.cache/vcftools-hwe`. It does not
install globally or build the Perl utilities. Configure can probe the system
Perl runtime. The release tag is `v0.1.17` (May 15, 2025), commit
`1c53c3c73be141103069965403e655536dda9c87`. The executable's reported version
is checked before use. There are no third-party Python packages, GPU calls,
API keys, or real genetic data. The checked Python version is the repository's
[root pin](../../../.python-version).

Inputs in [synthetic_input.json](synthetic_input.json) are deterministic
genotype counts; **seed: none**. A transient VCF expands those counts into
synthetic sample IDs, padding shorter rows with missing calls. The demo runs
`vcftools --vcf synthetic.vcf --hardy --out result`, reads its `.hwe` output,
and deletes the temporary files. A separately installed matching executable
can be supplied with `--vcftools /absolute/path/to/vcftools`.

## Actual checked output

Counts below are `(REF/REF, REF/ALT, ALT/ALT)`. The p-values are VCFtools output,
not substituted oracle results. The chi-square p-value is calculated separately
with one degree of freedom; VCFtools reports the chi-square statistic itself.

| Constructed case | Counts | VCFtools exact two-sided p | Asymptotic chi-square p |
| --- | --- | ---: | ---: |
| Balanced proportions | (25, 50, 25) | 1 | 1 |
| Deficit, 10 individuals | (4, 2, 4) | 0.07510446 | 0.05777957 |
| Same proportions, 100 individuals | (40, 20, 40) | 7.901338e-10 | 1.973175e-9 |
| Excess, same chi-square statistic 36 | (10, 80, 10) | 2.112246e-9 | 1.973175e-9 |
| Two ALT copies in one homozygote | (99, 0, 1) | 0.005025126 | 1.523971e-23 |
| Same ALT count, two heterozygotes | (98, 2, 0) | 1 | 0.9195424 |
| Balanced counts after removing 40 heterozygotes | (25, 10, 25) | 1.129777e-7 | 2.417564e-7 |

The rare-homozygote exact p-value is **1/199**; a very small asymptotic p-value
does not make the approximation reliable. Sample size matters even at identical
genotype proportions. Deliberately genotype-dependent removal produces a small
p-value without a biological departure in the original constructed counts.
The removal control represents the retained histogram, not a random-missingness
experiment or an estimate of real genotyping error.

The complete actual output, including inclusive one-sided deficit/excess tests,
is in [sample_summary.json](sample_summary.json). VCFtools prints p-values with
seven significant digits and expected counts with two decimals; comparisons
respect that precision. Values printed as 1 do not imply certainty.

## A complete conditional null, not a simulation estimate

For **20 diploid individuals and four ALT copies**, only three heterozygote
counts are possible:

| Genotype counts | Exact null probability | VCFtools two-sided p | Chi-square p |
| --- | ---: | ---: | ---: |
| (18, 0, 2) | 1/481 | 0.002079002 | 0.000007744216 |
| (17, 2, 1) | 72/481 | 0.1517672 | 0.04685418 |
| (16, 4, 0) | 408/481 | 1 | 0.6192568 |

At a preselected per-site threshold **0.05**, exact-test rejection probability
under this conditional null is **1/481 = 0.002079002**. The asymptotic test
rejects the first two configurations, giving **73/481 = 0.151767152**.
These are exhaustive probability sums, not Monte Carlo estimates or observed
false-positive rates in a genomic dataset. The exact test is conservative
here because its attainable p-values are discrete. This does not establish
power or calibration under population structure, genotype uncertainty or
related sampling.

## Independent calculation

Let `n` be the number of called diploids, `a` the ALT-copy count, and `h` the
heterozygote count. Compatible `h` has the parity of `a` and lies between zero
(or one) and `min(a, 2n-a)`. Counting placements of `a` labeled ALT alleles
among `2n` chromosome slots gives

```text
P(H=h | n,a) = choose(n,h) choose(n-h,(a-h)/2) 2^h / choose(2n,a).
```

[reference.py](reference.py) uses Python integers and `Fraction`, independently
of the recurrence used inside VCFtools. The two-sided test sums probabilities
**no larger than the observed configuration's probability**, including ties.
It is generally not twice the smaller one-sided tail, a mid-p test, or a
test based only on distance from the expected heterozygote count.
Both one-sided tails include the observed configuration.

VCFtools' reported expected genotype counts use plug-in allele frequency
`q=a/(2n)`: `n(1-q)^2`, `2nq(1-q)`, `nq^2`. These are distinct from the
conditional mean heterozygote count `a(2n-a)/(2n-1)` used by the exact null.
The independent Pearson baseline uses the plug-in counts and the chi-square
one-degree-of-freedom survival function `erfc(sqrt(statistic/2))`.

## Tests and limitations

The **32 tests** include all **454 genotype-count triples for 1–12 diploids**
compared against the actual executable, independent brute-force enumeration
of labeled chromosome placements for up to six diploids, rational null
calibration over a grid, the `1/(2n-1)` doubleton identity, inclusive tails,
sample/allele/phase invariance, missing calls, malformed input, monomorphic
degeneracy, sample-size controls, CLI failures and checked output. Integration
tests require the executable and fail if it is absent. CI also compares the
sample output byte for byte.

Assumptions: unrelated diploid samples, autosomal biallelic loci, accurate
genotype calls, and a common allele frequency under the null. Missing calls
are omitted; genotype-dependent missingness can invalidate the intended
sampling model. The wrapper rejects partial calls, haploids, multiallelic
genotypes, all-missing sites and more than 1,000 individuals. Monomorphic
controls have a degenerate exact p-value of 1 and an undefined Pearson
statistic; they carry no evidence of polymorphic HWE conformity.

There is no population structure adjustment, genotype-likelihood integration,
multiple-testing correction, causal diagnosis, demographic fitting, biological
parameter confidence interval, or real-data accuracy claim. HWE rejection
alone does not distinguish inbreeding, population mixture, selection, sampling
or genotyping problems. Non-rejection is not proof of equilibrium. This is a
small tool demonstration, not a full-paper reproduction.

## Primary sources and licenses

- Wigginton, Cutler & Abecasis (2005), **A Note on Exact Tests of Hardy-Weinberg
  Equilibrium**, especially equations 1–2 and the conditional-test discussion:
  [DOI](https://doi.org/10.1086/429864),
  [author-hosted paper](https://csg.sph.umich.edu/abecasis/publications/pdf/Am.J.Hum.Genet.vol.76-pp.887.pdf).
- Danecek et al. (2011), **The variant call format and VCFtools**:
  [DOI](https://doi.org/10.1093/bioinformatics/btr330).
- [Official VCFtools manual](https://vcftools.github.io/man_latest.html),
  `--hardy`; the online manual labels itself 0.1.16, so executable behavior
  was checked directly against the pinned 0.1.17 release.
- [Official release](https://github.com/vcftools/vcftools/releases/tag/v0.1.17)
  and [dependency license](https://vcftools.github.io/license.html).

This example's Python code, fixture and explanation are original and use the
repository's [MIT license](../../../LICENSE). VCFtools remains a separately
downloaded **LGPLv3** dependency; neither its source nor executable is committed.
No paper text, figures, upstream algorithm code, personal data or private
research is copied into the example.
