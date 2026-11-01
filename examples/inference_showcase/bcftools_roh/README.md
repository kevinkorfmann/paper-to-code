# BCFtools/RoH: autozygosity depends on likelihoods and allele frequencies

This original educational example runs **BCFtools/RoH 1.23.1**, linked to
**HTSlib 1.24**, on constructed diploid genotype likelihoods. It infers a sequence
of within-individual autozygous (AZ) versus Hardy–Weinberg (HW) states. The
estimand is conditional on supplied population allele frequencies and fixed HMM
transitions; the marker fraction called AZ is not a genome-wide inbreeding estimate.

The primary method is Narasimhan et al.,
[BCFtools/RoH: a hidden Markov model approach for detecting autozygosity from
next-generation sequencing data](https://doi.org/10.1093/bioinformatics/btw044),
published **January 30, 2016**, Bioinformatics 32(11):1749–1751.
See the [official tutorial](https://samtools.github.io/bcftools/howtos/roh-calling.html)
and [version-matched manual](https://github.com/samtools/bcftools/blob/1.23.1/doc/bcftools.txt).
This showcases an established inference tool, not a new inference method or a
reproduction of the paper's accuracy experiments.

## Run

From the repository root, use **CPython 3.14.5**, a C compiler, `make`, `curl`,
and zlib development headers. The Python code uses only the standard library.
The build is tested on macOS and in the Ubuntu 24.04 CI job.

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m examples.inference_showcase.bcftools_roh.install
python -m unittest discover -s examples/inference_showcase/bcftools_roh -v
python -m examples.inference_showcase.bcftools_roh.demo --check-sample
```

[install.py](install.py) downloads the official release archives, verifies their
SHA-256 hashes, and builds an isolated CLI in ignored `.cache/bcftools-roh`.
It does not install globally. GSL, plugins, Perl filters and optional network
codecs are disabled. Initial compilation needs network access; subsequent
inference runs are offline. The cached binary's hash and runtime versions are
checked before reuse. Compilers can produce different binary hashes; source
archives and tested output are pinned, not an identical cross-platform binary.

| Official source archive | SHA-256 |
| --- | --- |
| [BCFtools 1.23.1](https://github.com/samtools/bcftools/releases/download/1.23.1/bcftools-1.23.1.tar.bz2) | `01899a46f9420cdc1385d52fcfc84cce2806f9c996b787081a90d7dfc85eafa3` |
| [HTSlib 1.24](https://github.com/samtools/htslib/releases/download/1.24/htslib-1.24.tar.bz2) | `28a8de191381c7a97a35675ceac76fa1ea95e7b678d6a2e9d600a7874e4077de` |

The wrapper generates a temporary PL-only VCF and executes:

```sh
.cache/bcftools-roh/build/bcftools-1.23.1/bcftools roh \
  --AF-tag AF -a 0.02 -H 0.02 -b 0 --threads 0 synthetic.vcf
```

There is no transition fitting (`-V`), genetic map or supplied recombination
rate. Buffering is disabled for these tiny inputs. Consecutive positions make
the two transition probabilities 0.02 per marker in this deliberately narrow
reference model. The synthetic coordinate grid is not realistic SNP density.

## Data and independent model

[synthetic_input.json](synthetic_input.json) contains 180 markers for three
synthetic diploids, generated with `random.Random(20261101)`. ALT frequency is
0.5 at every marker. In HW, two alleles are independent Bernoulli draws; in AZ,
one allele is drawn and copied. The tract sample has designed AZ states at
markers 61–120. The other samples are all HW and all AZ controls.

For constructed genotype dosage `g`, likelihood entries are
`PL[c] = 30 * abs(g-c)` for `c=0,1,2`. These are informative likelihood vectors,
not simulated sequencing reads or empirically calibrated measurement errors.
Only PL and supplied AF reach the tool; truth states and genotype calls remain
outside its VCF input. Check regeneration without changing the fixture:

```sh
python -c "import json; from examples.inference_showcase.bcftools_roh.demo import FIXTURE, generate_fixture; assert json.loads(FIXTURE.read_text()) == generate_fixture()"
```

For ALT frequency `f` and genotype likelihoods `L[g] = 10**(-PL[g]/10)`,
the independent [reference.py](reference.py) uses:

```text
P(data | HW) = (1-f)^2 L[0] + 2f(1-f) L[1] + f^2 L[2]
P(data | AZ) = (1-f) L[0] + f L[2]
initial probabilities = (0.5, 0.5)
transition matrix = [[0.98, 0.02], [0.02, 0.98]]
```

It computes a log-space Viterbi path and scaled forward–backward marginals.
Tests independently enumerate all hidden paths with 70-digit Decimal arithmetic
on short inputs, and enumerate allele configurations for exact rational emission
checks. This reference is limited to consecutive markers, symmetric transitions
and the stated prior; it is not a replacement for BCFtools' general distance or
map handling.

## Executed results

[sample_summary.json](sample_summary.json) is actual executed output, rounded to
nine decimal places and checked byte for byte.

| Input | Designed AZ markers | Inferred AZ markers | Called interval |
| --- | ---: | ---: | --- |
| Tract, original PLs | 60 | 61 / 180 | 60–120 |
| Outbred control | 0 | 0 / 180 | none |
| Fully AZ control | 180 | 180 / 180 | 1–180 |
| Tract, every PL divided by ten | 60 | 0 / 180 | none |

The tract has one boundary error. Weakening PLs preserves every most likely
genotype but changes the inferred path: genotype calls alone omit evidence
strength. The controls remain 0 and 180 AZ markers after this weakening.

A separate 40-marker experiment holds PLs fixed: eight heterozygote-like rows,
24 homozygous-reference-like rows, then eight heterozygote-like rows. Supplied
ALT frequency 0.5 gives 24 AZ calls (markers 9–32); frequency 0.01 gives zero.
For its homozygous-reference row, the AZ/HW emission ratio falls from
**1.996007988** to **1.010080615**. A common homozygote provides less evidence
of autozygosity than one less expected under HW.

All **1,164** state calls across the main, weakened, frequency and four-marker
diagnostic inputs match the independent Viterbi implementation. Agreement with
the same specified model is a correctness control, not empirical accuracy.

## Quality-field diagnostic in the pinned version

We also check the tool's reported quality fields rather than assuming they are
calibrated probabilities. For PL rows `[30,0,30], [0,30,60], [0,30,60], [30,0,30]`
at frequency 0.5, both implementations call all four markers HW. However:

| Quantity | Four values in marker order |
| --- | --- |
| BCFtools 1.23.1 printed site quality | 3.0, 35.2, 25.4, 25.4 |
| Independent same-site phred error for the HW call | 35.237592189, 25.373701378, 25.373701378, 35.237592189 |

The independent marginals also match exhaustive Decimal enumeration. Here the
printed qualities after the first match the preceding site's independent values
at printed precision. This observation is specific to this version and setup;
the [upstream HMM](https://github.com/samtools/bcftools/blob/1.23.1/HMM.c) and
[RoH caller](https://github.com/samtools/bcftools/blob/1.23.1/vcfroh.c) provide
the version-matched implementation context. We retain raw reported qualities
and **do not interpret either site or region Q fields as calibrated confidence**.
No upstream source is patched. State-path agreement is checked separately.

## Tests, limitations and licenses

The 25 tests cover the real CLI, independent exhaustive and rational oracles,
likelihood scaling, allele reversal, posterior symmetry, all controls above,
fixture regeneration, omission of truth from inference input, version/checksum
pins, invalid input/output rejection, repeatability and checked CLI output.

Only three synthetic diploids and a few constructed controls are used. Fixed
transition rates, a symmetric prior, independent markers conditional on state,
complete biallelic input and supplied population frequencies are strong
assumptions. Real data add LD, genotype-error misspecification, ancestry mixture,
frequency estimation error, missing sites and varying density. The toy does not
estimate pedigree relationships, genome-wide F_ROH, time to a common ancestor,
confidence calibration or general detection accuracy. Our wrapper rejects flat
PL vectors because the tool skips them, and expects every consecutive marker.

All tracked example code and text are original under the repository
[MIT license](../../../LICENSE). The cited paper is CC BY 4.0; no paper figures,
text passages or upstream implementation are copied. Downloaded dependencies
remain outside Git and retain their licenses. This build disables GSL and its
BCFtools runtime reports MIT/Expat; see the full
[BCFtools license](https://github.com/samtools/bcftools/blob/1.23.1/LICENSE),
including the different conditions for GSL-enabled builds. HTSlib includes
MIT/Expat and component-specific notices, including BSD terms for CRAM;
see its [license](https://github.com/samtools/htslib/blob/1.24/LICENSE).
