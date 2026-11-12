# PLINK 2 heterozygosity: F depends on the reference

This original example runs the actual PLINK 2 executable on one constructed
sample at 20 biallelic autosomal markers. The calls stay fixed while the supplied
reference allele frequencies change. It is a small algebra demonstration, not
a realistic inbreeding analysis or a biological data set.

## Estimand, input, and assumptions

PLINK `--het` reports the method-of-moments excess-homozygosity coefficient

```
F = (O_hom - E_hom) / (N - E_hom) = 1 - O_het / sum_j 2 p_j (1-p_j).
```

The sum covers this sample's nonmissing markers. We use the default estimator,
without the optional small-sample correction. This is a **ratio of sums**, not
an average of the per-marker F values. It is conditional on the reference
frequencies, marker ascertainment and observed genotypes. A negative estimate
means excess heterozygosity relative to that reference, not a negative
probability or a measured negative pedigree-inbreeding coefficient.

The synthetic calls are `[0,1,1,2]` repeated five times: 10 heterozygous and 10
homozygous calls. Genotypes are unphased, complete diploid ALT counts initially.
The frequencies are supplied assumptions, not estimates from this one sample.
The `.afreq` file's 10,000 allele observations are synthetic metadata, not an
external panel or additional evidence; the default calculation uses the supplied
frequencies without a finite-sample correction. No human information is used.

The baseline expectation is Hardy–Weinberg heterozygosity given each supplied
frequency. Practical interpretation requires reliable, ancestry-appropriate
frequencies and suitable markers, usually approximately independent and in HWE.
Our 20-marker fixture does not establish those conditions. Missingness controls
are deliberately genotype-dependent. Neither they nor changing the reference
change the individual's underlying complete calls or pedigree.

## Actual checked output

| Scenario | Observed calls | Observed heterozygotes | Expected heterozygotes | PLINK F |
|---|---:|---:|---:|---:|
| Every p = 0.5 | 20 | 10 | 10 | 0 |
| Every p = 0.1 | 20 | 10 | 3.6 | -1.77778 |
| Every p = 0.9 | 20 | 10 | 3.6 | -1.77778 |
| Mask all heterozygotes, p = 0.5 | 10 | 0 | 5 | 1 |
| Mask all homozygotes, p = 0.5 | 10 | 10 | 5 | -1 |
| Copy every marker, p = 0.5 | 40 | 20 | 20 | 0 |

The rare-reference exact value is -16/9; PLINK prints rounded decimal results.
Copying sites does not add independent information, even though counts double.
No confidence interval, hypothesis test, sample-quality verdict, autozygous tract,
or demographic cause is inferred here. In particular, an extreme F alone cannot
distinguish genotyping error, reference mismatch, ascertainment, or biology.

## Run

Use Linux x86-64, CPython **3.14.5**, and the repository's existing checksum-pinned
PLINK **2.0.0-b.1.2LM (24 October 2026)** installer. This is a deliberately pinned
version, not a claim to use the newest release. No random sampling is used, so
there is no seed. No additional Python packages are required.

```sh
python -m examples.inference_showcase.plink_king.install
python -m examples.inference_showcase.plink_heterozygosity.demo
python -m unittest examples.inference_showcase.plink_heterozygosity.test_demo -v
python -m examples.inference_showcase.plink_heterozygosity.demo --check-sample
```

The wrapper writes a VCF and frequency file into a temporary directory and runs
`--vcf input.vcf --read-freq freq.afreq --het cols=hom,het,nobs,f --threads 1
--memory 640 --out result`. The installer verifies both archive and executable
SHA-256. The wrapper checks the executable checksum before each run and verifies
the returned version, schema and finite values. PLINK is separately installed
under its own license and not redistributed.

The 21 tests cover exact rational values, independent two-allele HWE enumeration,
a 12-case tool grid, heterogeneous frequencies and ratio-of-sums behavior,
allele reversal, marker permutation, missing-versus-removed loci, doubled counts,
input validation, the synthetic fixture, CLI and byte-exact checked output.
`sample_summary.json` is actual output; `synthetic_input.json` records the original
complete inputs. Tests compare all reported counts with exact rational oracles,
using explicit tolerances for PLINK's printed precision.

## Primary sources

- [Official PLINK 2 --het documentation](https://www.cog-genomics.org/plink/2.0/basic_stats#het)
  and the pinned executable's `--help het` document the statistic and its caveats.
- [Chang et al. (2015), Second-generation PLINK](https://doi.org/10.1186/s13742-015-0047-8),
  the software paper. This example implements original fixtures and checks;
  it does not reproduce that paper's performance benchmarks.
