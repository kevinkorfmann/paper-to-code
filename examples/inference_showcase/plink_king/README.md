# KING-robust kinship with PLINK 2: a useful estimate, not a universal pedigree oracle

**2026-10-29 · Original synthetic example · Actual PLINK 2 execution**

This example estimates pairwise diploid kinship from unphased genotypes using
PLINK 2's implementation of KING-robust. It compares a small synthetic pedigree
with unrelated and duplicate controls, then deliberately breaks the shared-ancestry
condition. **A real parent–child pair can get a negative estimate in that last
case.** The data are generated here, not downloaded human genetic data.

## Estimand, inputs and sources

The biological estimand is the probability that one randomly sampled allele
from each individual is identical by descent, relative to the founder population.
With unrelated, noninbred founders, the pedigree values are 0.25 for parent–child
and full siblings, 0.5 for duplicates and 0 for unrelated pairs. The **estimated
score is not constrained to be a probability**.

Primary method: Manichaikul et al. (2010),
[Robust relationship inference in genome-wide association studies](https://doi.org/10.1093/bioinformatics/btq559),
particularly Eq. 11. See the
[official PLINK 2 KING documentation](https://www.cog-genomics.org/plink/2.0/distance#make_king)
and [.kin0 output specification](https://www.cog-genomics.org/plink/2.0/formats#kin0).

The tool receives a generated VCF with **9 diploid samples and 10,000 biallelic
loci**. No pedigree or population-frequency information is passed to it.
PLINK's command uses the between-family estimator for every pair, regardless
of the simulated relationships:

```text
plink2 --vcf synthetic.vcf --make-king-table counts \
  cols=id,nsnp,hethet,ibs0,ibs1,kinship --threads 1 --memory 640 --out result
```

For jointly observed loci, writing `g` for alternate-allele count and `H` for
heterozygote count, the estimator is

```text
phi_hat = 1/2 - sum((g_i - g_j)^2) / (4 * min(H_i, H_j)).
```

The **minimum**, not the average, matters. Counts and an independent exact
rational implementation check all 36 output pairs. The executable's printed
scores have six significant digits, so comparison uses a magnitude-aware rounding
bound (at most `5e-6` for this fixture). Negative values are preserved. Zero heterozygosity in
either member makes this formula undefined, not evidence of unrelatedness.

## Constructed experiment

[`demo.py`](demo.py) generates founder alleles independently, using Python's
`random.Random(20261029)`. Two main founders have alternate-allele probability
0.3 at every locus. Two children inherit one randomly chosen allele from each
parent at each locus. A third independent founder at the same frequency is the
unrelated control, and a copy of the first founder is the duplicate.

Two additional unrelated founders have probabilities 0.1 and 0.9. Their child
inherits one allele from each. These deliberately extreme, constant frequency
panels are a stress test, **not a model of any named human population**.

All markers are independent by construction. Chromosome `1` and consecutive
VCF positions are only file-format labels: they do **not** imply a simulated
linked chromosome or physical recombination model. The main input has no
missingness, genotype error, mutation or inbreeding. A separate small test checks
pairwise removal of missing calls. Monomorphic loci are retained and add nothing
to the heterozygote/distance counts.

## Actual checked output

[`sample_output.json`](sample_output.json) is the captured default run, not an
expected-output template. Selected values are:

| Constructed pair | Pedigree kinship | PLINK score | Opposite homozygotes / 10,000 |
| --- | ---: | ---: | ---: |
| Main parent–child | 0.25 | 0.248417 | 0 |
| Full siblings | 0.25 | 0.246165 | 219 |
| Duplicate | 0.5 | 0.5 | 0 |
| Same-population unrelated | 0 | 0.00613972 | 845 |
| Different-population unrelated | 0 | −3.64086 | 6,575 |
| Low-frequency parent–mixed child | 0.25 | −0.661108 | 0 |
| High-frequency parent–mixed child | 0.25 | −0.655011 | 0 |

Parent–child and siblings have similar kinship estimates. Opposite homozygotes
provide extra information: the error-free parent–child pair cannot have one.
This is not a general guarantee of relationship classification.

The last two rows illustrate PLINK's documented warning about relatives whose
parents come from very different populations. Negative scores here are **not
negative IBD probabilities**. Clipping them to zero would hide the failed
interpretation. In an independent rational enumeration, the probability limit
of this score for the 0.1/0.9 parent–child construction is `−23/36 ≈ −0.638889`,
while pedigree kinship stays 0.25. This is a ratio-of-expectations limit, not the
finite-sample expected score and not an additional observed tool result.

An exhaustive, equally weighted **256-locus Mendelian control** at founder
frequency 0.5 gives PLINK scores exactly 0.25 for parent–child, 0.25 for full
siblings and 0 for unrelated parents. This is an algebra/implementation check,
not independent validation on real data.

## Reproduce

Reference environment: **Linux x86-64, CPython 3.14.5**, standard library only.
The external executable is **PLINK v2.0.0-b.1.2LM 64-bit Intel (24 Oct 2026)**,
from the dated official non-AVX2 Linux archive. No GPU, pip dependencies,
credentials or paid service is needed. The tool reserves 640 MiB, its supported
minimum, and runs one compute thread.

From the repository root:

```sh
python3.14 -m venv --without-pip .venv-king
. .venv-king/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m examples.inference_showcase.plink_king.install
python -m unittest discover -s examples/inference_showcase/plink_king -v
python -m examples.inference_showcase.plink_king.demo --write-vcf /tmp/king-input.vcf
python -m examples.inference_showcase.plink_king.demo --check-sample
```

The installer downloads only the official public archive, verifies its SHA-256
before extraction, and writes the executable and bundled library-license notice
under the ignored `.cache/` directory. It does not install system packages.
Every inference call verifies the executable bytes and version. Other operating
systems/builds are outside this pinned reference environment.

Archive SHA-256:
`b639901022d6b553d422eb571cfee8ff476b4f74babf258a81117b911ff45e41`

Executable SHA-256:
`50ff05c332e8a4ff75f9bdbb9f513b0eab639e604c986ecd8b84b7f664253db2`

Generated default VCF SHA-256:
`95e61152b8342dc1e4cab092fc23f52cbcf349c0d4f1746c3a4533110c3e3aa5`

Try a different finite sample without changing the checked reference:

```sh
python -m examples.inference_showcase.plink_king.demo --seed 20261030 --loci 20000
```

## Tests and limits

The **25 tests** cover all-pair count/score agreement, exact balanced pedigree
values, independent weighted Mendelian enumeration, the asymmetric-heterozygosity
fixture that distinguishes Eq. 11 from Eq. 9, duplicate and unrelated controls,
the ancestry counterexample, joint missingness, sample-composition/order,
locus-order and allele-label invariance, replicated loci, undefined/invalid
inputs, checksum rejection, large-negative-score rounding, deterministic input/output
and the CLI.

These checks establish the stated constructed behavior, not a general accuracy
benchmark. No confidence intervals, ancestry estimates, IBD tracts, relationship
classifier or full pedigree reconstruction are provided. Real applications must
consider variant ascertainment, LD, genotype error, missingness, ancestry and
inbreeding. Artificially repeating loci leaves the point score unchanged but
does not create independent evidence. The independent-locus simulation does not
validate uncertainty for real chromosomes.

Original generator, wrappers, tests and prose are covered by the repository's
MIT license. PLINK is an independently downloaded GPL-3.0 dependency and its
linked libraries retain their bundled license terms. No upstream executable,
source code, paper figure, private research or real genetic data is redistributed.
