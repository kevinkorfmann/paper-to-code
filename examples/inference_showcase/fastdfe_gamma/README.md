# fastDFE: target-site counts change the inferred DFE

Inference showcase **2026-10-25**. An original CPU example calling the real
**fastDFE 1.4.1** tool on synthetic neutral and selected site-frequency spectra
(SFS). It estimates a deleterious gamma distribution of fitness effects (DFE),
then changes only the declared selected-site target count. No empirical data,
private research, upstream implementation, or paper figures are redistributed.

## What is estimated

The fitted parameters are the **mean population-scaled selection coefficient**
`S_d < 0` and gamma shape `b > 0` for **new mutations at selected sites**:

```text
S = 4*Ne*s
-S ~ Gamma(shape=b, scale=abs(S_d)/b)
E[S] = S_d
```

This is not the distribution among surviving polymorphic variants. It also
does not identify an absolute `s` without an effective population size `Ne`.
The beneficial fraction is fixed to `p_b=0`; the unused beneficial mean is
fixed to `S_b=1`. Ancestral-state error `eps=0` and semidominance `h=0.5` are
assumed. Only `S_d` and `b` are optimized. Divergence data and bootstrapping
are explicitly disabled.

Inputs are unfolded neutral/selected polymorphic counts for derived-allele
counts 1 through 19 in **20 haploid genomes**, plus the number of mutational
target sites in each category. A target includes monomorphic sites, not just
variant records. Here neutral/selected targets are **1,000,000 / 2,000,000**.
The wrapper supplies the monomorphic count in bin zero and sets the unused
fixed-derived/divergence bin to zero.

fastDFE obtains the mutation scale and frequency-bin nuisance corrections from
the neutral SFS, then uses Poisson-random-field expectations to fit the selected
SFS. The same per-site mutation rate and demographic distortion are assumed
to apply to both categories. These nuisance corrections are not a reconstructed
demographic history. With the actual noisy neutral counts, the per-site scale
is `theta_hat=0.000991053555`.

## Synthetic input and actual output

The [portable fixture](synthetic_spectra.json) was generated from a constant,
panmictic neutral expectation `theta*L/i` with `theta=0.001`, and a fastDFE
selected expectation with `S_d=-20`, `b=0.5`. Independent Poisson draws of
polymorphic-bin counts use NumPy PCG64 seed **20261025**, drawing neutral bins
first, then selected bins. There are **3,516 neutral** and **3,206 selected**
polymorphic sites. This is a count experiment, not a forward or coalescent
genome simulation. Truth is stored separately for evaluation; fitting reads
only counts and declared targets.

[sample_summary.json](sample_summary.json) records the real tool output:

| Input/control | Mean S_d | Shape b | Selected Poisson log likelihood |
| --- | ---: | ---: | ---: |
| Observed counts, stated targets | -15.99298 | 0.577457 | -71.78978 |
| Same counts, selected target doubled to 4,000,000 | -282.92459 | 0.352551 | -82.08297 |
| Same counts, finer numerical grid | -15.99688 | 0.577233 | -71.79199 |
| Noiseless expected counts, same solver | -20.00000 | 0.500000 | -58.55156 |

The target perturbation preserves **every neutral and selected polymorphic
count**, the neutral target, and the estimated mutation scale. Only selected
monomorphic sites are added. The inferred probability of `|S| >= 100` changes
from **0.009288** to **0.478234**. This demonstrates sensitivity to a declared
input under this model and fixture, not a universal quantitative response.
The wrong target fits this fixture worse; no significance test is claimed.

The noiseless control contains fractional expected counts and recovers its
generating parameters with the **same solver**. It is an internal diagnostic,
not independent accuracy validation; its likelihood should not be compared
to the noisy-data likelihood. No seeds were searched for favorable results.

## Numerical choices and verification

All inference runs use deterministic Nelder-Mead through `fastdfe.BaseInference`,
with three explicit initial `(S_d,b)` pairs: `(-5,0.8)`, `(-50,0.3)`, and
`(-150,1.2)`. Bounds are `S_d in [-10000,-0.1]` and `b in [0.1,5]`.
The wrapper rejects failed fits, boundary optima, or disagreement between starts.
Optimizer settings are `xatol=1e-7`, `fatol=1e-8`, `maxiter=600`; there is one
run per specified start and no randomly chosen starts or worker pools.
This derivative-free method avoids finite-difference line-search failures in
the small two-parameter fit. Successful termination, agreement between starts,
and neighboring-likelihood checks are still required.

The semidominant DFE-to-SFS kernel uses midpoint integration over logarithmic
intervals `[-1e8,-1e-6]` and `[1e-6,1e3]`, with a connecting interval around
zero. The positive grid is retained for the tool's kernel; beneficial mass is
fixed to zero. At **400 intervals per sign**, the integration range captures
the negative gamma tail even at the fitting-bound corners. Refinement to
**800** changes fitted mean magnitude by **0.0244%** and shape by **0.0388%**.
These are measured discretization sensitivities, not statistical uncertainties
or a proof of full numerical convergence. Kernel precomputation is cached;
fitted answers are always recomputed.

Eighteen tests cover:

- The independent neutral `theta*L/i` expectation and harmonic-number mutation
  scale; reflected gamma CDF identities using `exp` and `erfc`, without using
  another gamma-CDF implementation as the reference.
- An independently evaluated Poisson likelihood, agreement between starting
  points, neighboring-parameter likelihood checks, same-solver noiseless
  recovery, and invariance to doubling both counts and targets.
- The target-only perturbation, grid refinement, tail coverage at bound
  corners, valid DFE probabilities, input immutability, exclusion of truth
  metadata from inference, fixture regeneration, invalid inputs, CLI, and
  checked output.

Printed floats have nine significant digits. Cross-platform sample checking
uses `rtol=1e-4`, `atol=1e-5`; versions, counts, and other non-float fields must
match exactly. These regression tolerances are not confidence intervals.

## Run from the repository root

Use **CPython 3.12.11** in a separate environment. PyPI metadata for fastDFE
1.4.1 requires Python `>=3.11,<3.14`; the root's Python 3.14.5 is unsuitable.

```sh
python3.12 -m venv .venv-fastdfe
. .venv-fastdfe/bin/activate
python -c "import platform; assert platform.python_version() == '3.12.11'"
python -m pip install --require-hashes -r examples/inference_showcase/fastdfe_gamma/requirements.lock
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1
python -m unittest discover -s examples/inference_showcase/fastdfe_gamma -v
python -m examples.inference_showcase.fastdfe_gamma.demo --check-sample
```

The tool imports plotting packages even though this demo does not plot. The
cache is directed to an ignored local directory. The complete dependency lock
pins fastDFE **1.4.1**, sfsutils-popgen **1.1.3**, NumPy **2.5.4**, SciPy **1.18.1**,
pandas **2.3.3**, and all transitive dependencies with distribution hashes.
[requirements.in](requirements.in) is the direct input to
[requirements.lock](requirements.lock). The non-yanked 1.4.1 release is used;
versioned documentation and the installed API were checked rather than relying
on the moving `latest` documentation. No GPU, VCF, paid service, or download of
sequence data is needed.

Regenerate the fixed input or explore another declared target size:

```sh
python -m examples.inference_showcase.fastdfe_gamma.demo --generate-input > /tmp/fastdfe-input.json
diff -u examples/inference_showcase/fastdfe_gamma/synthetic_spectra.json /tmp/fastdfe-input.json
python -m examples.inference_showcase.fastdfe_gamma.demo --target-factor 1.5
```

## Assumptions, limitations, and dependency license

This demonstration assumes a deleterious gamma family, correct neutral/selected
classification, known ancestral states, semidominance, shared mutation rate,
and valid target sizes. It excludes beneficial mutations, linkage, structure,
sequencing error, ascertainment bias, and explicit demographic changes.
The independent Poisson generator does not represent linked genomic sampling.
In real data the SFS likelihood is composite, and calibrated uncertainty needs
additional treatment of linkage and model/input errors. No intervals, adaptive
substitution estimates, real-data recommendations, or broad accuracy/throughput
benchmark are provided. A single synthetic seed cannot quantify bias or coverage.

The wrapper, fixture, tests, and explanation are original MIT-licensed repository
material. **fastDFE 1.4.1 is GPL-3.0-or-later**, installed separately from its
package distribution, and is not relicensed by this repository. No GPL
implementation code is copied or vendored here. The repository MIT license does
not replace the licenses of fastDFE or other installed dependencies.

## Primary sources

- Sendrowski & Bataillon (2024), [*fastDFE: Fast and Flexible Inference of the
  Distribution of Fitness Effects*](https://doi.org/10.1093/molbev/msae070),
  published online **April 5, 2024**, *Molecular Biology and Evolution* 41(5).
  This example illustrates use of the tool, not reproduction of its paper.
- Versioned official documentation: [DFE inference](https://fastdfe.readthedocs.io/en/v1.4.1/reference/Python/inference.html),
  [SFS expectations](https://fastdfe.readthedocs.io/en/v1.4.1/reference/Python/simulation.html),
  [parameterizations](https://fastdfe.readthedocs.io/en/v1.4.1/reference/Python/parametrizations.html),
  and [API](https://fastdfe.readthedocs.io/en/v1.4.1/modules/inference.html).
- [Official package metadata](https://pypi.org/project/fastdfe/1.4.1/) and
  [upstream repository and license](https://github.com/Sendrowski/fastDFE).
