# A zero adjusted statistic can mean an effect is not identifiable

A synthetic tskit sequence has eight haploid samples and two 10-base
windows, each with one biallelic site. The first site perfectly tags the
two groups (four samples each). The second site splits each group in
half. We assign a numeric trait equal to ten times the group indicator
plus two times the second marker.

[tskit's trait_linear_model](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.trait_linear_model)
regresses the trait on inheritance and optional covariates. At a
biallelic site, with span normalization disabled, the site statistic is
the square of the fitted genotype coefficient. With one site per
window, the two output values are directly comparable. The unadjusted
squared coefficients are 100 and 4. With group as a covariate, they
are 0 and 4. The first zero is the tool's convention for a genotype
perfectly collinear with the covariate, not evidence of no biological
effect. The second genotype can still be compared within each group.

The score is unsigned, is not a p-value, and sums over sites within a
window. The example is noiseless and not a recipe for choosing
covariates or inferring causal effects in real cohort data.

Run with CPython 3.12.11, tskit 1.0.4 and NumPy 2.5.5:

```sh
python -m unittest -v
python demo.py --check-sample
```

No random draws or external data. See [checked output](sample_output.json).
