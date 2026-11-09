# Rogers-Huff LD: pooling can create genotype correlation

An original deterministic scikit-allel 1.3.13 example. No random seed or external data.

## Estimand and tool

`allel.rogers_huff_r` takes biallelic diploid ALT dosages (0, 1, 2) and returns genotype correlation for each pair of variants. The method uses unphased genotypes to estimate linkage disequilibrium under appropriate population assumptions. Here the directly verified quantity is the correlation of the supplied dosages, not a recombination rate, physical distance, gametic-phase correlation in an arbitrary structured sample, or evidence of selection. A positive pooled result does not establish physical linkage.

## Input and actual result

Each of two groups has 625 constructed diploids at two sites. Group A has exact HWE dosage proportions `(16,8,1)/25` at both sites, with ALT frequency 0.2. Group B reverses those proportions, giving frequency 0.8. Each group's 3-by-3 joint table is the outer product of its two dosage-count vectors, so the dosages are exactly independent within each group. The fixture is an exact frequency table, not a random population sample or evolutionary simulation.

The library gives within-group r approximately zero (A: -4.66e-8; B: -7.45e-7, float32 roundoff). Combining all 1,250 unchanged individuals gives **r = 0.529411852 and r-squared = 0.280276909**. The exact values are 9/17 and 81/289. Group label predicts both dosages. This is real statistical dependence in the pooled data, not an arithmetic failure of the tool.

The law of total covariance explains it: within-group dosage covariance is zero, but between-group covariance of dosage means is 9/25. The pooled dosage variance is within-group 8/25 plus between-group 9/25, or 17/25. Thus pooled r=(9/25)/(17/25)=9/17. No recombination parameter or genomic position enters this construction. The same genotype association can occur for unlinked loci in a structured sample.

Controls swap the ALT label at one site (r reverses, r-squared does not), pool groups with identical frequencies (r stays zero), and include perfect association and a constant locus. Correlation with a constant locus is undefined and recorded as JSON null, never zero. Group-centering in a test removes the constructed correlation, but this is not a general ancestry-correction pipeline. Near-zero float32 residuals are kept in the output rather than silently rounded away.

## Reproduce

From the repository root, using CPython 3.12.11:

```sh
python3.12 -m venv .venv-ld-structure
.venv-ld-structure/bin/python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
.venv-ld-structure/bin/python -m examples.inference_showcase.ld_structure.demo
.venv-ld-structure/bin/python -m examples.inference_showcase.ld_structure.demo --check-sample
.venv-ld-structure/bin/python -m unittest discover -s examples/inference_showcase/ld_structure -v
```

The existing dependency lock pins scikit-allel 1.3.13 and NumPy 2.3.5. Actual inputs, version record and output are in `sample_output.json`; fixtures are in `demo.py`. Exact patch versions are needed for the checked version record. No phase inference, downloads beyond dependencies, private data or randomness are used.

## Checks and limitations

Twenty tests include an exact rational pair-difference oracle that does not reuse centered covariance sums, all 502 binary 3-by-3 tables with at least two observations, weighted homozygote-only tables, the total-covariance identity, genotype-order/replication/allele-label invariants, input validation and undefined cases. Float32 tool results are compared to exact values with explicit tolerances. The strict wrapper accepts only complete integer count tables, 0..1000 per cell, at least two diploids. Missing calls and genotype uncertainty are not modeled.

Biological interpretation requires reliable calls, comparable sampling and attention to relatedness and structure. Within-group HWE and independence are imposed here, not inferred. The pooled sample is not panmictic. A realistic LD analysis needs many markers, distances, sampling-aware uncertainty and the appropriate population target. This two-site construction supplies no p-value, confidence interval, demographic fit or real-genome accuracy claim.

## Sources and originality

- Rogers and Huff (2009), *Linkage Disequilibrium Between Loci With Unknown Phase*, Genetics 182:839-844. https://doi.org/10.1534/genetics.108.093153 . The API doc calls this method “2008,” while the paper was published in 2009.
- Official scikit-allel API: https://scikit-allel.readthedocs.io/en/stable/stats/ld.html#allel.rogers_huff_r
- Mangin et al. (2012; online 2011), *Novel measures of linkage disequilibrium that correct the bias due to population structure and relatedness*, Heredity 108:285-291. https://doi.org/10.1038/hdy.2011.73 . This motivates the structure caveat; the full corrected estimators are not implemented here.

Code and tests are original educational work under the repository's MIT license. This invokes the installed tool and does not copy its implementation or reproduce a complete paper.
