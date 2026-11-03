# DAMI: calibrating decay and Softplus sensitivity together

A small original implementation of the **scalar GDN calibration component** in
Feng et al., [DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context
Retrieval in Gated Linear Attention](https://arxiv.org/abs/2610.41994v1),
preprint first posted **October 30, 2026**.

DAMI sets geometric timescales across heads and a low–high–low Softplus-sensitivity
profile across layers. Its full procedure collects model activations sequentially,
using each calibrated layer's output to calibrate the next. This example instead
uses a fixed, constructed activation sample. It implements the scalar calibration,
not the full model initializer or the paper's architecture-specific extensions.

## Run

With **CPython 3.14.5**, from the repository root:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/dami_calibration -v
python -m examples.dami_calibration.demo --check-sample
```

On Windows, use `.venv\Scripts\Activate.ps1` to activate the environment.
There are no third-party dependencies, downloads, trained weights, data files,
paid services or GPU requirements. There is no randomness or seed. The pinned
interpreter matches the repository's [.python-version](../../.python-version).

[model.py](model.py) implements the component, [demo.py](demo.py) runs constructed
controls, and [sample_output.json](sample_output.json) contains actual executed
output rounded to nine decimal places. CI checks that file byte for byte.

## The two targets

The scalar equations from [§§3.3–3.5, equations 2 and 8–11](https://arxiv.org/html/2610.41994v1#S3)
are, with `b = dt_bias`, `a = A_log`, and a supplied sample `Z`:

```text
g(z) = -exp(a) * softplus(z + b)
retention(z) = exp(g(z))
mean_Z sigmoid(z + b) = target_sensitivity
mu = mean_Z softplus(z + b)
a = -log(target_timescale) - log(mu)
```

The bias is solved by bisection. Increasing rates give descending timescales
across heads. Our three-head example uses `[64,16,4]` and supplied layer targets
`[0.25,0.75,0.25]`. The component is original code expressing these equations;
Softplus, geometric spectra and root finding themselves are not new inventions.

## What the executed controls show

Our calibration sample is `[-2,-0.5,0.5,2]`. At timescale 16:

| Configuration | Softplus sensitivity | Mean log retention | Geometric-mean retention |
| --- | ---: | ---: | ---: |
| Joint calibration to 0.25 | 0.25 | -0.0625 | 0.939413063 |
| Change only bias to the 0.75 solution | 0.75 | -0.348952900 | 0.705426355 |
| Joint calibration to 0.75 | 0.75 | -0.0625 | 0.939413063 |

Changing the bias alone breaks the decay target. Recalibrating both parameters
restores it. These are arithmetic checks on constructed inputs, not training
results.

### Mean log retention is not mean retention

Substitution into the calibration equations gives `mean(g) = -1/tau`.
Consequently the **geometric** mean retention is `exp(-1/tau)`. The arithmetic
mean generally differs. Jensen's inequality implies
`mean(exp(g)) >= exp(mean(g))`, strictly when the log retentions vary.

An independently derived exact control uses activations `[-log(3),log(3)]`,
sensitivity 0.5 and `tau = 2/log(16/3)`. It yields `b=0`, `a=0`, and individual
retentions `[0.75,0.25]`. Their arithmetic mean is **0.5**, while their geometric
mean is **sqrt(3)/4 = 0.433012702**. Thus matching a mean logarithm does not fix
the mean of exponentiated values. No independent-draw assumption is used in
this finite-sample calculation.

### Softplus sensitivity is not full-gate sensitivity

Holding the calibrated parameters fixed,

```text
|dg/dz| = exp(a) * sigmoid(z+b)
|d retention/dz| = retention(z) * exp(a) * sigmoid(z+b).
```

The mean full log-decay slope at timescale 16 is **0.045576455** for target 0.25,
but **0.024489223** for target 0.75. The ordering reverses after joint calibration.
Calling the target simply “memory sensitivity” would conceal the multiplier.
The output therefore labels it `mean_softplus_sensitivity`.

### Calibration does not establish recall

Repeating the four calibration activations four times yields explicit decay
**0.367879441**, equal to `exp(-16/16)`. Adding 2 to every activation without
recalibrating changes the 16-step envelope to **0.034378789**. Sample calibration
does not guarantee the same behavior under a shifted distribution.

A separate scalar write control starts with state 1. The first explicit retention
is **0.994957086**. With key 1, update coefficient 1 and incoming value 0, the
subsequent delta write cancels that remaining state exactly. A slow explicit
decay envelope does not prevent write interference.

## Numerical contract and verification

There are **38 tests**, including:

- An independent 80-digit Decimal Newton solver versus production bisection,
  on 36 sample/target/timescale combinations.
- A closed-form constant-input oracle and the exact `[0.75,0.25]` retention control.
- Both target identities, translation/permutation/replication invariance,
  timescale rescaling, geometric spacing and bias monotonicity.
- Finite-difference Softplus, log-decay and retention derivatives.
- Independent product and repeated-cycle envelopes, prefix causality after
  calibration, shifted-input and delta-write failure controls.
- Extreme inputs, underflow, invalid inputs, an ill-conditioned root, immutability,
  near-flat accepted targets, tiny-Softplus/large-multiplier composition, CLI
  behavior and exact sample-output checks.

Calibration intentionally accepts only finite `|z| <= 100`, timescales in
`[1e-6,1e12]`, and sensitivities in `[1e-6,1-1e-6]`. It rejects a root whose
computed mean sigmoid derivative is below `1e-10`, since a nearly flat equation
can identify its bias poorly in binary64. This guard is a numerical safeguard,
not a formal bias-error bound. It also checks a `2e-13` sensitivity residual.
Other functions reject nonfinite inputs or overflow; tiny explicit retentions
may underflow to zero, while the log path remains available. Gate evaluation
combines its multiplier and Softplus in log space in the far negative tail to avoid premature underflow
of the Softplus factor. Tiny log decays can still underflow at binary64 limits.

For finite inputs, a bracket follows directly from monotonicity:
`[logit(s)-max(Z), logit(s)-min(Z)]`. The grid helper requires at least two heads,
as the cited spacing equation divides by `H-1`. Equal endpoints are supported.
The log path starts at zero and accumulates one-step logs without any future
input access. Calibration is an offline operation on the supplied sample;
this prefix check is **not** a causal or leakage-free training procedure.

## Scope, license and privacy

No layerwise model forward passes, learned projections, attention readout,
training, architecture-specific adaptation, optimized kernels or retrieval
benchmark are implemented. The delta-write control is a single scalar algebraic
counterexample, not a full GDN model. The geometric grid and supplied profile do
not prove an optimal spectrum, a power law, or generalization to unseen inputs.
Nothing here establishes speed, memory-capacity or language-model gains.

All code, fixtures and explanations here are original under the repository's
[MIT license](../../LICENSE). Only constructed numbers are used. No upstream
source code, paper figures or prose passages, model weights, private messages,
private research, credentials or real personal data are redistributed. The
paper's arXiv perpetual non-exclusive distribution license is separate from
this implementation's MIT license. The citation identifies the source idea,
not an endorsement by its authors.
