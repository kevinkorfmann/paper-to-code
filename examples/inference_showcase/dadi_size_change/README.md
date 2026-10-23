# dadi: estimating a size change conditional on its time

An original, small CPU example using **dadi 2.4.4** to fit a demographic
parameter from a synthetic, unfolded site-frequency spectrum (SFS).
It illustrates a useful distinction: **estimating a size ratio conditional on a
history is different from discovering that history**.

## Estimand, input, and model

The estimand is `nu = Ncurrent / Nancestral`, in a one-population model that
starts at neutral mutation-drift equilibrium, changes size instantaneously,
then stays at the new size. The change time `T` is **supplied**, not estimated;
its units are `2*Nancestral` generations. The observed input is the count of
variants with each derived-allele count `i=1,...,19` in **20 haploid genomes**.
Ancestral alleles are assumed known. The monomorphic bins 0 and 20 are masked.

The [fixture](synthetic_sfs.json) contains **2,307 segregating sites**. It was
generated with independent Poisson draws for each SFS bin around dadi's
expected spectrum, using `nu=3`, `T=0.2`, `theta=500`, and NumPy PCG64 seed
**20261023**. It is an idealized count experiment, not a simulated genome,
coalescent replicate, real genotype dataset, or private research. The generator
and its parameters are included; the fitting functions read only the SFS,
sample size, and explicitly assumed time, never the generating truth.

dadi solves a diffusion model for the expected SFS. This wrapper uses the
installed `Demographics1D.two_epoch` model, log extrapolation across grids
`[40,50,60]`, and dadi's NLopt BOBYQA interface. Three deterministic initial
ratios `[0.5,2,6]` are optimized directly within `[0.2,10]`, with `T` fixed.
The nuisance scale is profiled at every likelihood evaluation:

```text
m_i(nu, T) = expected SFS at theta=1
theta_hat = sum(observed_i) / sum(m_i)
log L = sum_i [observed_i * log(theta_hat*m_i)
               - theta_hat*m_i - log(observed_i!)]
```

This is dadi's `ll_multinom` fitting convention: a Poisson log likelihood
evaluated at the optimal scale, whose shape optimum corresponds to a
multinomial fit. The printed values retain Poisson constants. In the usual
diploid convention `theta=4*Nancestral*mu*L`; without `mu` and a suitable
sequence length, the fitted scale does not identify an absolute population
size. We do not estimate confidence intervals here.

## Actual checked output

[sample_summary.json](sample_summary.json) records all starts and controls.

| Fit/control | Assumed T | Estimated nu | Profiled theta | Profile log likelihood |
| --- | ---: | ---: | ---: | ---: |
| Synthetic observed counts | 0.2 | 2.728412 | 514.188227 | -77.643537 |
| Same counts, deliberately wrong time | 0.8 | 2.386620 | 409.592893 | -73.399148 |
| Same counts, finer grids `[60,70,80]` | 0.2 | 2.721641 | 513.372544 | -77.531787 |
| Noiseless expected counts, same solver | 0.2 | 3.000000 | 500.000000 | -57.378574 |

All three initial ratios agree within `1e-4` in each fit. Refining the grids
changes the estimated ratio by **0.2481%**. These grids therefore have visible
numerical error; agreement between starts does not remove discretization error.
The noiseless control checks internal recovery using the **same solver** and
fractional expected counts. It is not an independent validation of diffusion
accuracy, and its likelihood is not comparable to the noisy-data likelihood.

The wrong-time model fits this particular noisy sample **better**, despite
having the wrong generating time. Its different ratio makes the dependence
on assumptions concrete. Neither the likelihood difference nor the improved
fit establishes a true history, statistical significance, or a tool benchmark.
No seed was searched to obtain a desired result; the date-based default is fixed.

## Reproduce

Run from the repository root with **CPython 3.12.11**, separately from the
standard-library and other inference environments:

```sh
python3.12 -m venv .venv-dadi
. .venv-dadi/bin/activate
python -c "import platform; assert platform.python_version() == '3.12.11'"
python -m pip install --require-hashes -r examples/inference_showcase/dadi_size_change/requirements.lock
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -m unittest discover -s examples/inference_showcase/dadi_size_change -v
python -m examples.inference_showcase.dadi_size_change.demo --check-sample
```

dadi imports Matplotlib even though this example does not plot; its font cache
is directed to the ignored local cache directory. The environment pins dadi
**2.4.4**, NumPy **2.5.4**, SciPy **1.18.1**, NLopt **2.11.0**, and all transitive
dependencies with distribution hashes. [.python-version](.python-version)
pins the interpreter; [requirements.in](requirements.in) records direct inputs
to the complete [lock](requirements.lock). CPU execution is explicit, and no
model weights, genetic datasets, service, or GPU are needed.

Regenerate the synthetic input and explore a different assumed time:

```sh
python -m examples.inference_showcase.dadi_size_change.demo --generate-input > /tmp/dadi-input.json
diff -u examples/inference_showcase/dadi_size_change/synthetic_sfs.json /tmp/dadi-input.json
python -m examples.inference_showcase.dadi_size_change.demo --time 0.4
```

The 15 tests include an independent closed-form profiled Poisson likelihood,
the neutral equilibrium `theta/i` spectrum (0.5% solver tolerance), neighboring
likelihood checks, agreement across starting values, same-solver noiseless
recovery, invariance of the fitted ratio to doubling counts, corner masking,
wrong-time sensitivity, grid refinement, fixture regeneration, truth exclusion,
invalid inputs, CLI behavior, and checked output. Sample floats use `2e-5`
absolute/relative tolerance across CPU libraries; integer counts and version
strings must match exactly. These tolerances are regression checks, not
statistical confidence intervals. CI runs the real tool and all controls.

## Assumptions and limits

- One neutral, panmictic diploid population; equilibrium before a single size
  change; no selection, migration, structure, ancestral-state error, missingness,
  or ascertainment bias. Parameters use the standard diffusion approximation.
- Independent Poisson bins are the data generator here. In genomic data,
  linkage induces correlations: the usual SFS likelihood is composite, and
  calibrated uncertainty requires additional work such as block bootstrapping.
- This one-parameter example conditions on a fixed change time. It does not fit
  a flexible history, jointly resolve time and size, or estimate an absolute
  effective population size. All sites share the assumed demographic model.
- Same-model synthetic input is a controlled demonstration, not an assessment
  of robustness to real data. One seed provides no bias or accuracy estimate.
- Three starts and a grid refinement check are diagnostics, not a proof of a
  global optimum or full numerical convergence. The wrapper rejects boundary
  solutions and disagreement between starts instead of presenting them as a fit.

## Primary sources

- Gutenkunst, Hernandez, Williamson & Bustamante (2009),
  [*Inferring the Joint Demographic History of Multiple Populations from
  Multidimensional SNP Frequency Data*](https://doi.org/10.1371/journal.pgen.1000695),
  published **October 23, 2009**. Source of dadi's diffusion/SFS inference
  framework; this example does not reproduce the paper's empirical analyses.
- Official dadi documentation:
  [simulation and fitting](https://dadi.readthedocs.io/en/latest/user-guide/simulation-and-fitting/),
  [model specification and units](https://dadi.readthedocs.io/en/latest/user-guide/specifying-a-model/),
  and [data/Spectrum input](https://dadi.readthedocs.io/en/latest/user-guide/importing-data/).
  Documentation may evolve; the executable environment is pinned to 2.4.4.

The wrapper, tests, explanation, and synthetic fixture are original. dadi and
its dependencies are installed from their package distributions; upstream
implementation code is not copied into this repository.
