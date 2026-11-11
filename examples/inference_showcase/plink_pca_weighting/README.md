# PLINK 2 PCA: copying markers can swap the leading axis

This original, deterministic example runs the real PLINK 2 executable on a tiny
synthetic diploid genotype panel. It shows why **relative marker weighting matters**:
copying an existing genotype pattern can change PC1 even when the two nonzero
eigenvalues and the two-dimensional sample subspace stay the same.

## Sources and scope

- Method background: Patterson, Price & Reich (2006),
  [Population Structure and Eigenanalysis](https://doi.org/10.1371/journal.pgen.0020190).
  We use PCA, not the paper's significance tests or demographic interpretation.
- [Official PLINK 2 PCA documentation](https://www.cog-genomics.org/plink/2.0/strat#pca)
  defines the calculation in terms of a variance-standardized relationship matrix
  and explains why LD pruning matters. Its warnings distinguish broad population
  structure from batch effects, family structure and duplicated samples.
- [Official relationship-matrix documentation](https://www.cog-genomics.org/plink/2.0/distance#make_rel)
  describes allele-frequency standardization. The pinned executable's
  `--help pca make-rel` also documents the exact flags used here.

This code, fixture, calculation and tests are original. No human genotypes or
private research data are used. The external PLINK binary is not redistributed.

## Estimand and input

The estimand is the leading eigenvectors and eigenvalues of the **specified
sample-by-sample variance-standardized relationship matrix**. It is conditional
on the chosen samples, markers and supplied allele frequencies. It is not an
ancestry proportion, population label, migration rate, number of ancestral
populations, or an estimate of pedigree kinship.

For complete diploid ALT dosages `g_ik` and supplied ALT frequencies `p_k`,

```
R_ij = (1/M) sum_k (g_ik - 2p_k)(g_jk - 2p_k) / [2p_k(1-p_k)].
```

[synthetic_genotypes.json](synthetic_genotypes.json) contains 16 named synthetic
samples and three markers. The generator enumerates the Cartesian product of
`(0, 1, 1, 2)` with itself. The two resulting dosage vectors, A and B, each have
four 0s, eight 1s and four 2s. Their centered vectors are orthogonal. The baseline
has two copies of A and one copy of B at distinct positions with distinct IDs.
These are copied dosage patterns, not independent loci. The two entries labelled
1 in each factor are separate sample slots; several complete genotypes are
identical, and no unrelatedness or biological-population claim is made.

Every marker has ALT frequency exactly 1/2. We explicitly supply these toy sample
frequencies in a PLINK `.afreq` file. PLINK otherwise refuses to estimate PCA
frequencies from fewer than 50 samples. Supplying the known construction is
appropriate for this controlled algebra check, **not a recommendation to evade
that safeguard on a real small cohort**. No reference-population precision is
claimed. VCF contains unphased, complete calls, since this PCA does not need phase.

## Actual output and analytic explanation

The checked [sample_summary.json](sample_summary.json) is produced by the pinned
CLI, not by substituting the oracle for the tool.

| Panel | Counts A, B | PLINK eigenvalues as printed | PC1 axis |
|---|---:|---:|---|
| Baseline | 2, 1 | 10.6667, 5.33333 | A |
| Copy B three more times | 2, 4 | 10.6667, 5.33333 | B |
| Keep one of each pattern | 1, 1 | 8, 8 | Not uniquely identified |
| Copy every baseline marker | 4, 2 | 10.6667, 5.33333 | A |

The first two PC1 vectors have absolute normalized inner product **0.000000**.
Thus PC1 changes to an orthogonal direction, even though the two eigenvalues
remain the same at printed precision. Keeping both PCs preserves the same
subspace. This is not merely an arbitrary sign flip.

Let `u=A-1`, `v=B-1`. Then `u.u=v.v=8`, `u.v=0`, and for a copies of A and b of B,

```
R = [2a uu' + 2b vv'] / (a+b).
lambda_A = 16a/(a+b), lambda_B = 16b/(a+b).
```

The exact baseline eigenvalues are 32/3 and 16/3. Changing relative multiplicity
reverses their associated directions. Uniform copying preserves the average
matrix. Equal weights give an eigenvalue tie: any orthonormal rotation within
this two-dimensional subspace is equally valid. Comparing raw PC1 entries in
that case would be misleading, so the tests compare the projector instead.

Tool GRMs are read as binary doubles and checked against a separately written
`Fraction` matrix calculation. JSON diagnostic errors are rounded to six decimals
and print zero here; they are **not assertions of zero floating-point error**.
Tests require matrix error below 1e-12, projector error below 2e-6, and eigenvalue
error below 5e-5 to allow PLINK's printed precision. Eigenvectors are normalized
before comparison and signs are removed only for alignment calculations.

## Run and verify

Reference environment: Linux x86-64, **CPython 3.14.5**, **PLINK
v2.0.0-b.1.2LM (24 Oct 2026)**. There are no Python package dependencies.
The shared installer pins the official archive and executable SHA-256 hashes
and is reused unchanged from the KING example. There is no RNG seed: the data
and exact PCA are deterministic, and `--approx` is not used. One thread and
640 MiB requested memory are explicit in every CLI call.

From repository root:

```bash
python --version  # 3.14.5
python -m examples.inference_showcase.plink_king.install
python -m examples.inference_showcase.plink_pca_weighting.demo
python -m examples.inference_showcase.plink_pca_weighting.demo --check-sample
python -m unittest examples.inference_showcase.plink_pca_weighting.test_demo -v
```

The installer requires network access only for the official, checksum-verified
binary. Once installed, all data generation, calculations and tests are offline.
The demo generates temporary VCF/frequency files and invokes:

```bash
plink2 --vcf input.vcf --read-freq freq.afreq --pca 2 \
  --make-rel square bin --threads 1 --memory 640 --out result
```

The 25 tests include 16 marker-weight combinations, all eight allele-reversal
combinations of the baseline, sample/marker permutations, exact rational GRMs,
analytic eigenvalues, vector residuals, the tied-space projector, uniform-copy
invariance, a separate non-1/2-frequency panel, checked input/output and malformed
input rejection. The existing `plink-king` CI job runs this example too.

## Assumptions and limitations

The construction assumes error-free, aligned, autosomal biallelic diploid calls,
complete observation, fixed frequencies and equal weighting of each marker row.
The HWE genotype proportions here are imposed combinatorially, not inferred.
The baseline is already redundant and is not described as LD-pruned. Keeping one
copy of each known pattern is an explicit control, not an implementation of
PLINK's LD-pruning algorithm. Marker positions are arbitrary and PCA ignores
their physical distances in this setup.

This is an algebraic sensitivity demonstration, not a simulated demographic
history, real-genome benchmark, ancestry classifier or validation of population
structure inference. It supplies no confidence interval or significance test.
Real analyses must also consider ascertainment, missingness, allele-frequency
uncertainty, relatives, technical artifacts and the distribution of LD. The
lesson is narrower: a visually dominant sample axis can reflect the weighting
of correlated markers, and preserving eigenvalues alone does not preserve axes.
