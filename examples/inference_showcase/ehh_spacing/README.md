# EHH: identical allele sharing is not identical physical extent

Original educational example using **scikit-allel 1.3.13**. Eight synthetic,
correctly phased chromosomes share a core allele at the first marker. As three
more biallelic markers are added, their shared-prefix groups split from 8 to
4+4, then 2+2+2+2, then eight singletons.

## Estimand, input and result

`allel.ehh_decay(..., truncate=False)` measures the fraction of **distinct
chromosome pairs** matching from the first through each successive marker.
For this core-conditioned sample it returns **1, 3/7, 1/7, 0**. If the input
contains different core alleles, the function does not condition on one allele
for you; subset the chromosomes explicitly when that is the estimand.

The array is markers × chromosomes. All input values and positions are generated
in `panel()` and `run()` in `demo.py`; there are no external data, simulation
priors, random draws or seeds. This is not an inferred genealogy or mutation age.

The same matrix at positions `[0,100,200,300]` bp gives a one-sided, finite-window
trapezoidal area of **750/7 = 107.142857 bp**. Positions `[0,1000,2000,3000]` bp
leave EHH unchanged but give **7500/7 = 1071.428571 bp**. Changing units on the
first panel to kb gives **0.107142857 kb**, which is the same physical area,
not a biological difference. Using marker indices gives **15/14 marker
intervals**, not base pairs. Coordinates are relative to the core.

The area is an **original NumPy trapezoidal calculation** using scikit-allel's
EHH values. It is not a call to the library's integrated haplotype score (iHS),
XP-EHH or a full iHH scan. There is no threshold truncation, extrapolation beyond
the observed window, gap correction, genetic-map conversion, opposite flank,
allele comparison or frequency-bin standardization. It must not be interpreted
as a calibrated selection statistic. We make no p-value, selection coefficient,
confidence interval or empirical accuracy claim.

## Useful controls

- Removing marker row 2 changes the observed prefixes. At the last retained
  marker, EHH becomes 1/7 rather than zero. Sparse markers can conceal differences.
- Ending the window at marker row 2 leaves terminal EHH 1/7; the finite observed
  area is not the total area of an unobserved tail.
- Duplicating every chromosome raises final EHH from 0 to 1/15, because the
  duplicated copies are now distinct sample indices. Repeated samples are not
  new independent information. The estimator uses pairs without replacement,
  unlike a raw sum of squared empirical haplotype frequencies.
- Chromosome order, consistent allele relabeling and position translation leave
  the relevant quantities unchanged. Uniform position scaling scales area only.

## Assumptions and limits

Known phase, aligned homologous sites, accurate complete biallelic calls, and
ordered coordinates with stated units are assumed. Missing or unsupported
calls are rejected rather than treated as an allele. Biological interpretation
also depends on recombination, demography, sampling, allele frequency and marker
ascertainment. A long shared haplotype alone does not diagnose selection.

The eight chromosomes and four markers are deliberately constructed to make the
counting transparent, not to represent a natural population or validate any
selection model. Physical spacing and genetic recombination distance differ.
The 100000-entry size cap is only a small-demo guard, not a performance promise.
The independent exact oracles are intended for small test panels, not genomes.

## Reproduce

From the repository root, use CPython **3.12.11** and the existing isolated,
hash-locked environment (NumPy **2.3.5**):

```sh
python3.12 -m venv .venv-ehh
.venv-ehh/bin/python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
.venv-ehh/bin/python -m unittest discover -s examples/inference_showcase/ehh_spacing -v
.venv-ehh/bin/python -m examples.inference_showcase.ehh_spacing.demo
.venv-ehh/bin/python -m examples.inference_showcase.ehh_spacing.demo --check-sample
```

Twenty tests compare the library to two independent rational oracles: explicit
chromosome-pair enumeration and prefix-class counting. They exhaust all **4672
binary matrices with three markers and 2–4 chromosomes**, and cover the spacing,
sampling, truncation and invalid-input controls. `sample_output.json` records
actual execution and is checked exactly under the pinned environment. CI extends
the existing scikit-allel job, without installing new dependencies.

## Sources and license

- Sabeti et al. (2002), *Detecting recent positive selection in the human genome
  from haplotype structure*. [Primary paper](https://doi.org/10.1038/nature01140),
  [author-hosted full text](https://reich.hms.harvard.edu/sites/reich.hms.harvard.edu/files/inline-files/Sabeti.2002_0.pdf).
- [Official EHH API/source documentation](https://scikit-allel.readthedocs.io/en/stable/_modules/allel/stats/selection.html#ehh_decay).
  The hosted stable docs identify an older documentation build; the executed
  implementation is pinned to 1.3.13 and tested directly.
- [Official iHS documentation](https://scikit-allel.readthedocs.io/en/stable/stats/selection.html#allel.ihs)
  explains the gap, edge, map and standardization features not implemented here.

Code, tests and explanations are original and under the repository MIT license.
No paper text, figures, upstream implementation or real genomic data are copied.
Dependencies retain their own licenses.
