# Beagle: imputing hidden genotypes depends on the reference

This original synthetic example calls **Beagle 5.5 (27Feb25.75f)** to infer
alternate-allele dosages at untyped markers. It compares matched, reversed,
and uninformative reference haplotypes while keeping the observed target
genotypes, allele labels, map, and withheld truth fixed.

Primary method: Browning, Zhou & Browning (2018),
[A One-Penny Imputed Genome from Next-Generation Reference Panels](https://doi.org/10.1016/j.ajhg.2018.07.015)
([open primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6128308/)).
The [official Beagle page](https://faculty.washington.edu/browning/beagle/beagle.html)
identifies that paper as the imputation reference for version 5.5.
See also the [official manual](https://faculty.washington.edu/browning/beagle/beagle_5.5_17Dec24.pdf),
especially sections 3.1, 3.3, 3.4, and 5.

## Reproduce

The pinned reference platforms are **macOS arm64** and **Linux x86_64**.
Use **CPython 3.14.5**, with no third-party Python packages:

```sh
# From the repository root; reuse the existing root .venv if present.
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m examples.inference_showcase.beagle_imputation.install
python -m unittest discover -s examples/inference_showcase/beagle_imputation -v
python -m examples.inference_showcase.beagle_imputation.demo --check-sample
```

The installer downloads the official Beagle JAR and **Eclipse Temurin JRE
17.0.21+9**, verifies pinned SHA-256 hashes, and keeps them in the ignored
`.cache/beagle-imputation/` directory. It does not install system software or
change PATH. Runtime archives are approximately 43–47 MB; subsequent tests
and demos run offline. [install.py](install.py) records exact URLs and hashes,
including separate runtime hashes for both platforms. Java uses one worker
thread and a 256 MiB maximum heap. Normal extraction preserves the runtime's
license files; the archive filter rejects escaping paths and unsafe links.

Both the data generator and Beagle use seed **20261031**. The checked
[synthetic_input.json](synthetic_input.json) regenerates exactly. The checked
[sample_summary.json](sample_summary.json) is executed output, compared byte
for byte. Beagle prints rounded probabilities and dosages; the JSON summary
rounds derived statistics to nine decimal places. It excludes timestamps,
temporary paths, and elapsed times from Beagle's raw logs/VCF headers.

## Estimand and inputs

The estimand is each target individual's **expected number of ALT alleles**
at an unobserved diploid marker, conditional on observed phased genotypes,
reference haplotypes, genetic map, and model settings. DS ranges from 0 to 2;
it is not a directly observed genotype or an ancestry proportion.

The fixture has 200 biallelic markers spaced 1 kb apart, 80 reference
haplotypes (40 diploid records), and three targets: AA, AB, and BB. A is a
seeded binary haplotype; B is its complement. These labels are synthetic
haplotype classes, not nucleotide bases, ethnic groups, or human samples.
All targets have known phase. Only **51 markers** are written to the target
VCF; the other **149 markers / 447 target genotypes** are withheld. Truth
is used only when scoring output and is never supplied to Beagle.

Each reference marker has ALT frequency exactly 0.5 in all controls:

* **Matched:** reference chromosomes carry A or B throughout.
* **Flipped:** only reference alleles at hidden markers are complemented.
  REF/ALT labels stay A/G; this is an intentional incorrect haplotype panel,
  not allele harmonization. Typed reference records remain identical.
* **Uninformative:** each typed haplotype class has equally many copies of
  both hidden allele patterns. Observed tags cannot distinguish those copies.

The explicit map spans 0.199 cM (1 cM/Mb). Beagle receives `ne=10000`,
`err=0.0001`, `em=false`, `seed=20261031`, `nthreads=1`, `impute=true`,
`cluster=0.005`, `imp-states=1600`, `window=40`, `overlap=2`, and requests
AP1, AP2, and GP output. Its inputs are already phased; this example tests
imputation, not statistical phasing. Ne and mismatch probability are supplied
model parameters, not demographic estimates obtained by this demonstration.

## Actual output and independent controls

At the first hidden marker (position 2000), the withheld dosages are `[0,1,2]`:

| Reference | Beagle dosages | MSE over all 447 hidden genotypes | Reported DR2 at position 2000 |
| --- | --- | ---: | ---: |
| Matched | `[0,1,2]` | 0.000000000 | 1.00 |
| Flipped | `[2,1,0]` | 2.666666667 | 1.00 |
| Uninformative | `[0.92,0.96,1.00]` | 0.616000000 | 0.00 |

Every observed genotype remains unchanged in all three runs. Zero error and
endpoint probabilities here refer to **Beagle's printed precision**, not
mathematical certainty or an accuracy guarantee on other data.

An independent frequency-only prediction ignores tags and returns `2p=1`
for every target, with exact MSE `(1+0+1)/3 = 2/3`. A separate exact-rational
copying control enumerates all reference chromosomes matching a target
haplotype at every typed site, averages their hidden alleles with equal
weights, and sums the two haplotype means. Under **zero error and zero
recombination**, it gives `[0,1,2]`, `[2,1,0]`, and `[1,1,1]` respectively.
It is an explicitly simpler baseline, not a reimplementation or exact oracle
for Beagle's finite-state HMM with positive error/recombination. In the
uninformative case, Beagle differs from this baseline by up to 0.08 dosage;
the small MSE difference on three constructed targets is not evidence of
better inference or a calibrated uncertainty estimate.

The reversed result also exposes a metric limitation: signed correlation
with truth changes from **+1 to −1**, while empirical squared correlation
stays **1**. Beagle's DR2 is its model-based estimate of squared correlation,
not the probability that a genotype is correct, a measure of signed
agreement, or an external check of the reference panel. A high DR2 cannot
replace scrutiny of allele direction, reference construction, and errors.

## Tests, assumptions, and limits

The **32 tests** execute all three Beagle runs and verify fixture regeneration,
withheld-site isolation, unchanged typed input, constant reference frequencies,
exact hand-enumerated copying and MSE controls, signed correlation, complete
output coverage, dosage complement symmetry, preserved genotypes, DS/AP/GP
consistency at output precision, pinned dependencies, invalid input/output,
deterministic repetition, CLI behavior, and the checked sample.

The construction assumes biallelic diploid data, known phase, complete and
correct typed genotypes, aligned marker order/alleles, and a supplied map.
The target/reference haplotypes are deliberately repetitive and extremely
simple; they are not a demographic sample or a realistic benchmark. Markers
are linked, so 447 scored genotypes are **not 447 independent replicates**.
No intervals, uncertainty calibration, ancestry robustness, rare-variant
accuracy, clinical interpretation, or real-genome performance is claimed.
Hard calls, DS, and model GP are different outputs; the demo scores DS.

## Licenses

Our wrapper, generator, fixtures, tests, and prose are original and MIT
licensed. Beagle is a separately downloaded **GPL-3.0-or-later** executable;
its official source archive and license are linked on the Beagle page. Its
BGZIP support includes MIT-licensed Broad Institute code. No Beagle code or
binary is redistributed here. Eclipse Temurin/OpenJDK has
[GPLv2 with the applicable Classpath/Assembly exceptions](https://adoptium.net/about#licenses);
its [pinned official release](https://github.com/adoptium/temurin17-binaries/releases/tag/jdk-17.0.21%2B9)
retains its accompanying notices in the local cache. No paper text, figures,
private data, or real-world genetic data are copied.
