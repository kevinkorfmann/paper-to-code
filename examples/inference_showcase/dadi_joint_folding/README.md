# dadi: joint folding is not independent minor-allele counting

Original deterministic educational example, November 13, 2026. This calls real
**dadi 2.4.4 `Spectrum.fold()`**. It demonstrates preparation of a demographic
inference input, not a fitted demographic history, selection test or accuracy benchmark.

## Input, target and actual output

Each synthetic panel has 20 biallelic sites and two populations with 10 called
haploid chromosomes each. A row records counts of the **same labeled allele**
in both populations. There is no missing data, phase, randomness or seed.
No real or private data are used. These are constructed count tables, not
samples from a simulated evolutionary history.

The target is the empirical joint frequency spectrum after removing ancestral
polarity while preserving cross-population allele alignment. dadi identifies
complementary counts `(i,j)` and `(10-i,10-j)`, choosing the orientation with
lower total count. Equal-total ties split their mass equally; a self-complementary
cell retains all its mass. Jointly monomorphic corners are masked by default.

Actual results in [sample_output.json](sample_output.json):

- 20 rows `(1,1)` become 20 sites in `(1,1)`.
- 20 rows `(1,9)` become 10 sites in `(1,9)` and 10 in `(9,1)`.
- Both panels collapse to 20 in `(1,1)` if each population independently chooses
  its minor allele. That different summary discards the allele alignment.
- 20 fixed differences `(0,10)` remain 20 globally segregating sites after
  joint folding (10 in each complementary cell). Independent minor counts put
  these in `(0,0)`, the same cell as shared absence. That comparison alone does
  not remove any sites, but treating that cell as monomorphic would lose them.

Thus absence of an ancestral allele assignment need not make same-versus-opposite
frequency patterns indistinguishable. Independently folded marginal counts are
not a substitute for dadi's jointly folded input. This is a distinction between
summaries, not a claim that another software package is wrong.

## Assumptions and limits

Alleles must be aligned across populations and sites. Sample sizes are fixed,
complete and supplied explicitly; the wrapper limits them to 2..100 chromosomes.
The code rejects noninteger, negative, out-of-range and malformed counts.
A shared reference allele is a valid temporary orientation for joint folding;
it is not automatically the ancestral allele. Folding removes polarity, not
ascertainment bias, calling error or unmodeled structure. No independence
assumption is needed just to count these sites; downstream composite likelihoods
and uncertainty estimates require their own modeling and dependence treatment.
No demographic parameter, confidence interval or p-value is estimated here.

## Reproduce

From the repository root, use the existing isolated, hash-locked dadi environment:

```sh
python3.12 -m venv .venv-dadi
.venv-dadi/bin/python -m pip install --require-hashes -r examples/inference_showcase/dadi_size_change/requirements.lock
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/paper-to-code-matplotlib XDG_CACHE_HOME=/tmp/paper-to-code-cache
.venv-dadi/bin/python -m examples.inference_showcase.dadi_joint_folding.demo
.venv-dadi/bin/python -m unittest discover -s examples/inference_showcase/dadi_joint_folding -v
.venv-dadi/bin/python -m examples.inference_showcase.dadi_joint_folding.demo --check-sample
```

Validated with CPython 3.12.11 and the versions in the output. CI reuses the
existing dadi job and lockfile. The 18 tests include all 121 one-site cells for
samples `(10,10)`, all 48 cells for `(5,7)`, an explicit per-site orientation oracle,
complement invariance, population transpose, ties, masks, mass, linearity,
invalid inputs and byte-stable sample output. Dyadic half-counts are exactly
representable. The tests check the documented transformation, not biological
validity of a fitted model. No upstream implementation code is copied.

## Sources and licenses

- [dadi official guide: folding and masking](https://dadi.readthedocs.io/en/latest/user-guide/manipulating-spectra/#folding).
- [Gutenkunst et al. (2009), demographic inference with multidimensional frequency spectra](https://doi.org/10.1371/journal.pgen.1000695).
- [Pinned dadi 2.4.4 release](https://pypi.org/project/dadi/2.4.4/).

Example code and prose are original under the repository MIT license.
Separately installed packages retain their own licenses. No paper text, figures,
external datasets or upstream source are redistributed.
