# ADMIXTURE projection: ancestry estimates depend on the references

**2026-10-30 · Inference showcase · actual ADMIXTURE 1.4.0**

This original example invokes ADMIXTURE's **projection mode** (`-P`) on five
synthetic diploid individuals. It estimates a two-source mixture coefficient
while holding the supplied allele-frequency profiles fixed. It does **not**
learn ancestral populations, select K, reconstruct a pedigree, date admixture,
or infer local ancestry tracts.

The useful lesson: unchanged genotypes can produce doubled mixture estimates
when the supplied reference profiles are brought closer together. Identical
profiles make the coefficient unidentifiable, even though the program prints
numbers and reports convergence.

## Estimand and synthetic input

For individual i and marker j, the model is

```text
p_ij = q_i p_Aj + (1-q_i) p_Bj
G_ij ~ Binomial(2, p_ij)
```

The estimand `q_i` is the **source-A global mixture coefficient conditional on
K=2 and these particular source profiles**. In this generator it is the
probability of choosing source A for an independently sampled allele. It is
not an observed fraction of inherited chromosome segments, a recent pedigree
fraction, or a posterior probability that a person belongs to a population.
A and B are abstract synthetic sources, not named human populations.

[`synthetic_genotypes.json`](synthetic_genotypes.json) records:

- Five individuals generated with q values `0, 0.25, 0.5, 0.75, 1`.
- 2,000 preselected, independent biallelic markers. Four `(p_A, p_B)` pairs,
  `(0.1,0.8), (0.2,0.7), (0.15,0.65), (0.35,0.9)`, repeat 500 times.
- Two independent Bernoulli draws per genotype using CPython's
  `random.Random(20261030)`. Dosage strings count copies of allele 2.

The wrapper writes original PLINK BED/BIM/FAM files and `study.2.P.in` in a
private temporary directory, then executes:

```sh
admixture -P --seed=41 -C=1e-10 -j1 study.bed 2
```

`run_projection()` receives **only dosages, reference frequencies, and an
optimizer seed**. Generating q values are evaluation-only metadata, never
supplied to ADMIXTURE. Reference frequencies are deliberately known exactly
here: we are isolating projection, not estimating a reference panel. BED allele
orientation is verified by a separate asymmetric closed-form integration
control; swapping allele coding and complementing P leaves q unchanged.

## Run

Use **Linux x86-64** with kernel 3.17+ and **CPython 3.14.5**, from the repository root:

```sh
python3.14 -m venv --without-pip .venv-admixture
. .venv-admixture/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m examples.inference_showcase.admixture_projection.install
python -m unittest discover -s examples/inference_showcase/admixture_projection -v
python -m examples.inference_showcase.admixture_projection.demo --check-sample
python -m examples.inference_showcase.admixture_projection.demo
```

No third-party Python packages, PLINK executable, GPU, paid API, real genetic
data, or reference-panel download is needed. The installer downloads only the
[official versioned ADMIXTURE archive](https://dalexander.github.io/admixture/download.html),
verifies its SHA-256 and the executable's SHA-256, and extracts only the named
binary into ignored `.cache/`. Each inference invocation rechecks the binary.
The hashes are reproducibility pins, not upstream digital signatures:

```text
archive: 5520c4f18525f49ab2c715d711b40f4a0818ef1485361a94ec8954d1364e6f19
binary:  dbab7f695dc159bddd72a1137f11cb2888ca2939565e6123fbb80e001c4e613f
```

The reference uses one thread, initialization seeds **41 and 99**, and a
log-likelihood-change threshold of `1e-10`. [`sample_summary.json`](sample_summary.json)
is actual output, including the fixture checksum, versions, and six-decimal
values printed by the executable. `--check-sample` compares the complete
summary byte for byte; this is a pinned-environment check, not a promise for
arbitrary future software or architectures. Regenerate the input independently:

```sh
python - <<'PY'
import json
from examples.inference_showcase.admixture_projection.demo import FIXTURE, generate_input
assert generate_input() == json.loads(FIXTURE.read_text())
PY
```

## Actual output: change the references, not the genotypes

| Generating q | Correct-reference estimate | Halved-contrast estimate |
| ---: | ---: | ---: |
| 0 | 0.000010 | 0.000010 |
| 0.25 | 0.280409 | 0.560818 |
| 0.50 | 0.470208 | 0.940415 |
| 0.75 | 0.778415 | 0.999990 |
| 1 | 0.999990 | 0.999990 |

The second fit replaces `p_A` with `(p_A+p_B)/2`, leaving `p_B`, all genotypes,
and the optimizer seed unchanged. Algebra explains the doubling:

```text
p_B + q_original (p_A-p_B)
  = p_B + q_narrow ((p_A+p_B)/2-p_B)
when q_narrow = 2 q_original.
```

This equality is feasible only up to `q_original=0.5`. The two higher-q
individuals hit the upper boundary instead. The estimates printed near 0/1
reflect the executable's numerical offsets, not proof of pure ancestry.
The correct-profile independent mathematical MLEs are
`[0, 0.280409242, 0.470207698, 0.778415125, 1]`; interior estimates agree with
the tool to its printed precision. The discrepancy from generating truth
includes finite-genotype sampling and is **not** a benchmark error estimate.

### No contrast: numbers without identification

Set both supplied profiles to `0.5` at every marker. Now `p_ij=0.5` for every
q. All q values have exactly the same likelihood. For the second synthetic
individual, ADMIXTURE prints **0.630516** with seed 41 and **0.426764** with
seed 99. Both runs give **−13862.943611** for the log likelihood, matching the
analytical value `−2 × 5 × 2000 × log(2)` (binomial constants omitted).
Neither numerical coefficient is an identified ancestry estimate. This
example deliberately treats a flat likelihood as `None` in its independent
oracle rather than assigning an arbitrary answer.

### Convergence is not an optimality certificate

An additional stress input has two individuals, one with all-zero dosages and
one with all-two dosages at 40 markers, each with `(p_A,p_B)=(0.2,0.8)`.
The mathematical maximizers are `[1,0]`, with log likelihood **−35.702968**.
The pinned default block algorithm reports convergence but returns
**[0.998559, 0.001664]**, log likelihood **−35.889384**: a demonstrably
suboptimal result. The test records this limitation; it does not disguise it
by loosening the oracle or treating the convergence message as sufficient.
We have not determined the internal cause in the proprietary executable.
This does not invalidate the separately checked main fits, and it does not
establish the behavior of other ADMIXTURE versions or algorithms.

## Independent controls and scope of testing

The **28 tests** include:

- A bisection maximizer of the concave one-dimensional binomial likelihood,
  independent of ADMIXTURE's block/EM implementation, checked against an
  exact rational constant-profile solution and a 60-digit Decimal calculation.
- Actual executable recovery of q=0.25 for an asymmetric constructed dosage
  panel, plus the boundary stress case above.
- Independent likelihood evaluation, multiple initialization seeds, reference
  sensitivity, flat-likelihood analysis, and verification that output P remains
  fixed rather than being refitted.
- Allele-complement invariance, source-label exchange, marker permutation and
  individual-order equivariance; exact BED bytes and a separate decoder.
- Seeded fixture regeneration, invalid input/output rejection, executable
  checksum checking, CLI behavior, and checked sample output.

Those are numerical, algebraic and integration controls, **not** validation
of historical population labels or accuracy on real genomes. The documented
boundary behavior is an expected regression finding, not a failed CI test.

## Assumptions and limitations

The fitted likelihood uses diploid biallelic calls, independent alleles
conditional on q/P, and independent markers; this synthetic construction
satisfies those assumptions. It excludes missing calls, phasing issues,
relatedness, linkage, inbreeding, genotype error and reference-panel sampling
error. Source profiles, allele alignment, and K are assumed correct in the
main fit, then deliberately misspecified in controls.

No bootstrap, standard errors, posterior distribution, admixture dates,
confidence intervals, or selection test is computed. Real-world structure
need not follow a two-source discrete mixture. These conditional coefficients
are not categorical identities, ethnicity labels, or a complete ancestry
history. Good optimization cannot repair an unidentifiable or misspecified
reference model; independent controls are still needed when a program says
it converged.

## Primary sources and licensing

- Alexander, Novembre & Lange (2009), **Fast model-based estimation of ancestry
  in unrelated individuals**, *Genome Research* 19:1655–1664.
  [Primary paper / DOI](https://doi.org/10.1101/gr.094052.109);
  [author-hosted manuscript](https://dalexander.github.io/preprints/admixture-preprint.pdf).
  The genotype likelihood and model-based global coefficient are the method
  being exercised, not reimplemented for the reported estimates.
- [Official ADMIXTURE 1.4 manual](https://dalexander.github.io/admixture/admixture-manual.pdf),
  §2.14 (fixed-P projection), §2.5 (seed), §2.7 (termination), and §2.3 (LD).
  [Official version history](https://dalexander.github.io/admixture/news.html).
- [Official PLINK BED format](https://www.cog-genomics.org/plink/1.9/formats#bed)
  documents the file layout, not this example's inference algorithm.
- Alexander & Lange (2011), [Enhancements to the ADMIXTURE algorithm for
  individual ancestry estimation](https://doi.org/10.1186/1471-2105-12-246),
  states that binaries are freely available, source code proprietary, with
  no restrictions on non-academic use.

All example Python, text, and synthetic fixtures are original and covered by
the repository's MIT license. **ADMIXTURE is a separately downloaded proprietary
executable; it is not relicensed or redistributed here.** No upstream source,
paper figure, private research, real person's genotypes, or credentials are
included. Citations do not imply endorsement.
