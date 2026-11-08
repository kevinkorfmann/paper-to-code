# Admixture f3: a positive result does not exclude admixture

An original, deterministic scikit-allel 1.3.13 example. No random seed is used.

## Estimand and tool

We call `allel.patterson_f3(C, A, B)`. The unnormalized statistic estimates the locus-average product `(p_C-p_A)(p_C-p_B)`. This is neither an admixture fraction nor an event date. Under the appropriate population-tree/drift null, a significantly negative result provides evidence against an unadmixed target history. A positive value cannot exclude admixture. No significance test or p-value is claimed here.

The tool corrects the plug-in product by subtracting `p_hat_C(1-p_hat_C)/(n_C-1)`. This removes target sampling bias when the populations are independently sampled as binomial chromosomes. The example also reports f3-star, the ratio of summed numerators to summed target heterozygosities. When target heterozygosity is zero, f3-star is undefined and represented as JSON null, while unnormalized f3 remains defined.

## Constructed input and result

Two biallelic sites each have 20 chromosomes per population. Sources have ALT counts 4 and 16 at both sites. Three target panels have counts `[10,10]`, `[4,16]`, or `[0,20]`. All have the same across-site mean target frequency of 0.5, but progressively larger dispersion. Actual unnormalized tool outputs are -0.103157895, -0.008421053 and +0.16. These are deliberately constructed count panels, not stochastic drift simulations, demographic reconstructions or real-genome benchmarks.

The separate exact population calculation explains the mechanism. A 50:50 mixture of source frequencies 0.2 and 0.8 starts at p=0.5 with f3=-0.09. A mean-zero target shift delta changes the expected product to `-0.09 + E[delta^2]`. Thus post-admixture target drift can mask the negative signal. Endpoints 0 and 1 illustrate the extreme fixation limit, with population f3=+0.16. The deterministic count-panel correction is not itself an estimate of drift magnitude. Source evolution and complex demographic models are not modeled.

## Reproduce

From the repository root, use CPython 3.12.11 and the existing isolated, hash-locked environment (NumPy 2.3.5, scikit-allel 1.3.13):

```sh
python3.12 -m venv .venv-f3
.venv-f3/bin/python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
.venv-f3/bin/python -m examples.inference_showcase.f3_drift.demo
.venv-f3/bin/python -m examples.inference_showcase.f3_drift.demo --check-sample
.venv-f3/bin/python -m unittest discover -s examples/inference_showcase/f3_drift -v
```

The exact CPython patch version is required for the checked output's version record. No external data, downloads beyond locked packages, or private information are used. Input fixtures are generated directly in `demo.py`; `sample_output.json` is actual output.

## Independent checks and limits

Nineteen tests include a distinct ordered-pair U-statistic oracle, all 147 count configurations for three small sample-size triples, exact binomial expectation over nine probability/sample-size cases, allele reversal, source swapping, invalid data and undefined normalization. The oracle enumerates `(C_i-A_j)(C_k-B_l)` for distinct target chromosomes i and k instead of reusing the tool's correction formula. It is deliberately limited to at most eight chromosomes per population. Tests also demonstrate that the naive product has upward bias `p_C(1-p_C)/n_C`.

Counts assume homologous, aligned biallelic sites, independently sampled chromosomes, no missing populations, genotype errors, relatives or overlapping target/source samples. This strict wrapper rejects fewer than two chromosomes per population; it is not a general missing-data pipeline. Real analyses need appropriate ascertainment, population choices, dependence-aware uncertainty and enough loci. Repeated constructed sites are not independent evidence. No jackknife, confidence interval or selection inference is made.

## Sources and originality

- Patterson et al. (2012), *Ancient Admixture in Human History*, Genetics 192:1065–1093, especially three-population test and Appendix A. https://doi.org/10.1534/genetics.112.145037
- Official scikit-allel API: https://scikit-allel.readthedocs.io/en/stable/stats/admixture.html#allel.patterson_f3

The demo and tests are original educational code under this repository's MIT license, not copied source or a reproduction of a complete study. The installed library is invoked as a dependency.
