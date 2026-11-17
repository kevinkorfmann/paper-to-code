# dadi’s profiled likelihood separates shape and scale

An original, deterministic example using **dadi 2.4.4** to fit one multiplicative scale to a supplied site-frequency-spectrum shape. No demographic parameters are optimized and no diffusion equation is solved here.

## Input and estimand

Five sampled chromosomes define four unfolded polymorphic bins, for derived-allele counts 1, 2, 3 and 4. The original fixture contains **[60, 30, 20, 15]** sites, an exact-count construction proportional to the neutral constant-size shape `m_i = 1/i`. It is neither a random draw nor empirical data. Ancestral polarity, complete calls and aligned bins are assumed. Monomorphic corners are masked.

The fitted quantity is a scale `a` conditional on the supplied shape and active bins. Expected counts are `a * m_i`. For the independent Poisson working model,

```text
log L(a) = sum_i [k_i log(a m_i) - a m_i - log(k_i!)]
a_hat = sum_i k_i / sum_i m_i
```

The sum includes only bins masked in neither model nor data. The derivative is `S/a - M`, and the second derivative is `-S/a^2`, so this is the unique positive maximum when `S>0` and `M>0`. `dadi.Inference.optimal_sfs_scaling` fits this factor. `ll_multinom` returns the Poisson score after profiling it.

## Actual output

| Supplied shape | Fitted scale | Fitted mean counts | Profiled Poisson log likelihood |
|---|---:|---|---:|
| `0.1 / i` | 600 | [60,30,20,15] | -10.289303946 |
| `1 / i` | 60 | [60,30,20,15] | -10.289303946 |
| `10 / i` | 6 | [60,30,20,15] | -10.289303946 |
| flat `[1,1,1,1]` | 31.25 | [31.25,31.25,31.25,31.25] | -28.268875594 |

Multiplying the model by 10 divides its fitted scale by 10, leaving the fitted means and score unchanged. This is a change of normalization, **not evidence for a tenfold biological change**. Shape still matters: the flat spectrum cannot fit the four counts exactly. The unscaled Poisson scores differ, as recorded in the checked JSON.

In an additional sensitivity control, only the singleton count is doubled to 120. The `1/i` model's fitted scale becomes **88.8**. Masking that bin in both model and data returns **60**, based on the remaining bins. This illustrates a changed fitting target, not advice to discard inconvenient data. Its absolute likelihood is **not comparable as model evidence** to the unmasked likelihood because the observations differ.

### Why the function name can be confusing

For fixed data with total `S`, the profiled Poisson log likelihood equals the conditional multinomial log likelihood plus `-S + S log S - log(S!)`. This constant does not depend on model shape. Thus model rankings on the same active data agree, but the absolute scores are not the same. The tests verify this identity independently. It does not license score comparisons between differently masked or differently sized datasets.

## Reproduce

Use CPython 3.12.11 and the existing isolated hash-locked dadi environment:

```sh
python3.12 -m venv .venv-profile-scale
.venv-profile-scale/bin/python -m pip install --require-hashes -r examples/inference_showcase/dadi_size_change/requirements.lock
.venv-profile-scale/bin/python -m unittest discover -s examples/inference_showcase/dadi_profile_scale -v
.venv-profile-scale/bin/python -m examples.inference_showcase.dadi_profile_scale.demo --check-sample
.venv-profile-scale/bin/python -m examples.inference_showcase.dadi_profile_scale.demo
```

Reference versions: CPython 3.12.11, dadi 2.4.4, NumPy 2.5.4. No seed applies. JSON numbers are rounded to nine decimal places. Nineteen tests include 372 count/normalization combinations checked against an independent rational scale and `math.lgamma` Poisson oracle, direct likelihood maxima, the multinomial constant, mask intersection, zero-count bins, permutations, invalid inputs and output replay.

## Assumptions and limitations

This wrapper accepts at least two aligned polymorphic bins, nonnegative integer counts, strictly positive finite model entries, and positive total active count. It rejects all-zero active spectra, duplicate/invalid masks and zero-support model cases rather than attempting their boundary limits. Inputs are interpreted as unfolded; folded spectra are not accepted as a separate mode.

For truly independent Poisson counts, the formula is an ordinary likelihood. SFS-based analyses of linked genomic sites generally use it as a **composite likelihood**. Linkage, nonrandom missingness, ancestral misidentification, model misspecification and ascertainment matter for inference and uncertainty. This tutorial reports no confidence intervals or likelihood-ratio significance. The scale is interpretable as a biological theta only under a specified model normalization, mutation model and units. Converting to effective population size additionally requires the appropriate mutation rate and callable sequence length. No population size, growth rate, demographic history or biological accuracy is estimated here.

## Sources

- [Official dadi fitting guide, likelihoods](https://dadi.readthedocs.io/en/latest/user-guide/simulation-and-fitting/#likelihoods): Poisson and optimally scaled comparisons.
- [Official Inference API and source display](https://dadi.readthedocs.io/en/latest/autoapi/dadi/Inference/): `optimal_sfs_scaling` and `ll_multinom` behavior, also checked against the installed hash-locked 2.4.4 source.
- [Gutenkunst et al. (2009)](https://doi.org/10.1371/journal.pgen.1000695): demographic inference from joint frequency spectra and composite-likelihood context.

All fixtures, analytic controls and example code are original, not a reproduction of a published empirical fit.
