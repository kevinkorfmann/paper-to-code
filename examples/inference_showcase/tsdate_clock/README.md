# tsdate: ancestor ages depend on the mutation clock

Daily inference showcase **2026-10-21**. This example calls the actual
**tsdate 0.2.7** inference engine to estimate three ancestor ages conditional
on a known synthetic genealogy. It does not infer topology or reuse yesterday's
tsinfer pipeline. It is an original small workflow around an existing tool.

The `variational_gamma` method approximates marginal posterior node ages using
expectation propagation. Its primary reference is Pope et al. (2026),
[Tracing the evolutionary histories of ultra-rare variants using variational dating of large ancestral recombination graphs](https://doi.org/10.64898/2026.01.07.698223),
a preprint. See also the developers'
[method citation](https://tskit.dev/tsdate/docs/stable/citation.html),
[API](https://tskit.dev/tsdate/docs/stable/python-api.html), and
[usage guide](https://tskit.dev/tsdate/docs/stable/usage.html).

## Synthetic experiment

Four present-day haploid samples have a fixed bifurcating topology. Two pairs
coalesce at deliberately chosen ages 100 and 200 generations; their ancestors
coalesce at age 500. These ages specify the simulator and are kept for
evaluation only. No demographic model or real genetic data is involved.

The six branches cover a 1,000,000-base sequence. Each independently receives
a Poisson number of mutations with mean `mu × length × branch_duration`, using
mu=10^-7 per base per generation. Positions are independently uniform; all
144 realized events occur at different sites. Binary derived states are mapped
to the known branches, so genotype phase and ancestral states are error-free.
The original generator uses NumPy PCG64, seed **20261021**.

The readable [synthetic_input.json](synthetic_input.json) freezes that input.
`build_undated` constructs inference tables from the topology and mutations
without reading true ages or the simulation rate. Temporary node heights are
0,0,0,0,1,1,2, marked **uncalibrated**; mutation times are unknown. Tests verify
that scaling those placeholders by 10,000 does not determine the answer.

We run 25 expectation-propagation iterations, with time rescaling disabled.
The method's empirical prior is estimated internally; no true population size
or true node ages are supplied. We then rerun inference assuming twice the
mutation rate to expose the clock's scale dependence.

## Install and run from the repository root

Use the separate **CPython 3.12.11** environment. All transitive dependencies
and distribution hashes are pinned in [requirements.lock](requirements.lock).
Direct versions are in [requirements.in](requirements.in).

```sh
python3.12 -m venv .venv-tsdate
.venv-tsdate/bin/python -c "import platform; assert platform.python_version() == '3.12.11'"
.venv-tsdate/bin/python -m pip install --require-hashes -r examples/inference_showcase/tsdate_clock/requirements.lock
.venv-tsdate/bin/python -m examples.inference_showcase.tsdate_clock.demo --check-sample
.venv-tsdate/bin/python -m unittest discover -s examples/inference_showcase/tsdate_clock -v
.venv-tsdate/bin/python -m examples.inference_showcase.tsdate_clock.demo --mutation-rate 2e-7
```

On Windows use `.venv-tsdate\Scripts\python.exe`. Initial import compiles CPU
Numba kernels and can take a few minutes. Optional `TSDATE_ENABLE_NUMBA_CACHE=1`
avoids recompiling in this environment; no GPU, model download, or service is
used. The checked scientific stack is tsdate 0.2.7, tskit 1.0.3, NumPy 2.5.4,
SciPy 1.18.1 and Numba 0.68.0.

## Actual results

| True age (evaluation only) | Posterior mean | Posterior SD | Mean with twice the assumed rate |
| ---: | ---: | ---: | ---: |
| 100 | 95.952075 | 20.863321 | 47.976037 |
| 200 | 272.461330 | 34.294280 | 136.230665 |
| 500 | 545.793297 | 47.236757 | 272.896649 |

Times are generations. These are **unconstrained approximate posterior moments**
from the fit object, rather than topology-constrained output times. All
576 genotype calls and all edges are preserved. Doubling the assumed rate
halves means and quarters variances to numerical tolerance. The sizeable
error for the 200-generation ancestor is retained, not hidden or tuned away.

[sample_summary.json](sample_summary.json) records the actual run. Twelve tests
cover input generation, truth withholding, placeholder invariance, genotype and
topology preservation, valid age ordering, posterior uncertainty, clock scaling,
response to increased mutations, invalid clocks, and the checked report.
`--check-sample` compares floats with relative and absolute tolerance 10^-6 to
allow platform-level floating-point differences; categorical fields are exact.

## Limits

This is conditional genealogical dating, **not population-size inference**.
The topology, mutation placement, clock, and sequence length are assumed known.
There is no recombination, recurrent mutation, missing data, phasing error,
ancient sampling, or topology uncertainty. A single chosen four-tip tree is
neither a realistic population sample nor an accuracy/calibration benchmark.
The reported SDs omit uncertainty in those assumptions. Without external
clock information, mutation counts cannot identify an absolute generation
scale on their own. The preprint and software describe much broader methods;
this example reproduces no full-paper results.
