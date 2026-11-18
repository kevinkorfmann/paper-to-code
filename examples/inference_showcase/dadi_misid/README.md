# Ancestral-state errors: folding removes the nuisance parameter and its information

Original deterministic inference showcase, November 18, 2026. Uses the real
**dadi 2.4.4** `Numerics.apply_anc_state_misid` and `Inference.ll_multinom` APIs.

## What is fitted?

One ancestral-state misidentification probability, conditional on a supplied
unfolded spectrum shape. A grid search evaluates probabilities 0–0.5 in steps
of 0.01, profiling out the overall spectrum scale with dadi. This is not a
joint demographic fit, an uncertainty assessment, or an inference of selection.
The grid maximum is not a claim of arbitrary numerical precision.

The synthetic input contains five sampled chromosomes and four segregating-site
bins. The supplied shape is `[120,60,40,30]`; deliberate observed integer counts
are `[102,56,44,48]`. These counts exactly equal 80% of the supplied shape plus
20% of its reversal. They are a same-model construction, not independent
biological validation or a random sample. There is no seed. The constructor's
20% value is metadata and never passed into the fitting function.

## Actual result

- Unfolded: unique grid maximum at **0.20**, log likelihood **−11.834546770**.
- Fixing error probability at zero gives **−18.147837273**.
- After folding, both input spectra become `[150,100]`. All **51** probabilities
  have the same score to numerical precision. Returning the first optimizer
  value as an estimate would hide non-identifiability.
- A symmetric supplied shape also has no orientation information, even before
  folding. Multiplying the supplied shape by seven leaves profiled scores unchanged.

The mechanism is `g_i=(1-e)f_i+e f_(n-i)`. Pairing complementary bins removes `e`:
`g_i+g_(n-i)=f_i+f_(n-i)`. The midpoint of even sample sizes is unchanged too.
This is why folding tolerates simple ancestral/derived swaps but prevents
estimating their probability. It does not fix ascertainment, genotype error,
missingness or demographic misspecification. The model assumes one common
error rate across bins and no additional errors. Real polarization errors can
be frequency/context dependent. An incorrect supplied demographic shape can
confound error-rate inference.

The likelihood is dadi's profiled Poisson working likelihood. Linked genomic
sites ordinarily require composite-likelihood uncertainty methods. No interval,
p-value, Bayes factor or biological accuracy claim is supplied. Do not compare
absolute folded and unfolded likelihoods as evidence: the data categories differ.

## Reproduce

From the repository root, using **CPython 3.12.11** in an isolated environment:

```sh
python3.12 -m venv .venv-dadi
. .venv-dadi/bin/activate
python -m pip install --require-hashes -r examples/inference_showcase/dadi_size_change/requirements.lock
python -m unittest examples.inference_showcase.dadi_misid.test_demo -v
python -m examples.inference_showcase.dadi_misid.demo
python -m examples.inference_showcase.dadi_misid.demo --check-sample
```

The existing dependency lock pins dadi 2.4.4 and NumPy 2.5.4. No new dependencies.
The 19 tests include independent Poisson log-likelihood calculations for all 51
grid values, 77 rational-mixture comparisons, odd/even sample-size folding
controls, scale invariance, endpoint masking, invalid inputs and saved-output
replay. Test controls do not establish real-data accuracy.

## Primary sources

- [Official dadi misidentification API](https://dadi.readthedocs.io/en/latest/autoapi/dadi/Numerics/#dadi.Numerics.apply_anc_state_misid)
- [Official folding documentation](https://dadi.readthedocs.io/en/stable/user-guide/manipulating-spectra/)
- Gutenkunst et al. (2009), [Inferring the Joint Demographic History of Multiple Populations from Multidimensional SNP Frequency Data](https://doi.org/10.1371/journal.pgen.1000695).

Original glue code and constructed fixture; no source implementation copied.
