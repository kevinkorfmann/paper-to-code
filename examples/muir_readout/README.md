# Muir readout: the useful row depends on retrieval lag

Original educational implementation, 2026-11-07. Reference: Tushaar
Gangavarapu, Gil Tabak and Clemens JS Schaefer,
[Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout](https://arxiv.org/abs/2611.08376v1),
preprint submitted **5 November 2026**, specifically Proposition 3.6,
[Equations 6–7](https://arxiv.org/html/2611.08376v1#S3.SS1).

Muir motivates a query-dependent gain on memory rows with different retention
and interference levels. Here we implement only its supplied-moment readout
objective and optimum, not the recurrent architecture. All code and toy inputs
are original. No paper implementation was copied. The primary paper is CC BY 4.0;
this repository's MIT license applies to our code.

## Run

From repository root, use **CPython 3.14.5**, standard library only:

```sh
python -m examples.muir_readout.demo
python -m examples.muir_readout.demo --check-sample
python -m unittest discover -s examples/muir_readout -t . -v
```

No installation, training, randomness, seed or downloaded data is needed.
`sample_output.json` is actual output, rounded to twelve decimal places for
readability. Tests use unrounded values.

## Supplied model

For each row j, let c[j] be key energy (positive, summing to one), m[j] be
positive interference energy, a[j] be surviving target signal, and gamma[j]
be its readout gain. The signal variance s is positive. The objective is

```
L(gamma) = s * (1 - sum_j gamma[j]*c[j]*a[j])**2
           + sum_j gamma[j]**2*c[j]*m[j]
gamma_mse[j] = (s / (1 + s*sum_i c[i]*a[i]**2/m[i])) * a[j]/m[j]
```

The separate unit-signal solution replaces the common scalar by
`1/sum_i c[i]*a[i]**2/m[i]`. It has zero bias but need not minimize total MSE.
The readout is `(gamma * query) @ state`.

These equations assume centered interference with diagonal row covariance and
the stated signal moments. We treat those moments as known. We do not derive
or estimate them from a trained network. `retention**lag` is a deterministic
toy survival law, not a general identity for random gates. All positive moments
must lie in [1e-100,1e100]; this illustrative binary64 solver is not intended for
arbitrarily ill-conditioned inputs. Zero surviving signal is allowed for MSE,
while unit-signal normalization then has no solution.

## Actual result and controls

Our two rows have c=(0.5,0.5), m=(0.1,1), and retention=(0.5,0.95).
At lag 1 the optimal gains are (1.850995,0.351689). At lag 10 they become
(0.008281,0.507728): the fast, low-noise row has almost no signal left.
The MSE-optimal scaling of each competing direction is fitted analytically,
so controls do not lose merely because of an arbitrary overall scale.
At lag 10, inverse-noise weighting gives MSE 0.983448, versus 0.847998 for
matched readout. Forcing unit signal increases the latter to 5.578871.

A separate **original finite-mixture control** minimizes mean MSE over equal
probabilities for the two lags. Set b_l=c*a_l and D=diag(c*m). Its normal equations are

```
(D + s*E[b_l b_l.T]) gamma = s*E[b_l]
```

The exact shared-gain MSE is 0.664709, versus 0.675363 when mean survival is
substituted into the single-lag formula. The difference is the omitted
between-lag signal variance. An oracle told the lag uses separate gains and
reaches 0.609099. These are losses under the constructed moment model, not
language-model evaluations. This comparison does **not** refute the paper's
Corollary 3.8, which optimizes interference under a horizon-averaged
unit-signal constraint, a different objective.

## Verification and limits

29 tests cover independent exact-rational optima, all 512 signal/noise sign
configurations in a four-row finite ensemble, quadratic optimality gaps,
finite-difference gradients, unit-signal versus MSE objectives, fixed-direction
scale controls, exact 2x2 mixture inversion, the Jensen variance gap,
permutation, malformed input, numeric extremes, CLI and checked output.
The finite ensemble supplies constant-magnitude keys so its moment closure is
exact; it does not validate the assumptions for learned keys.

This is an oracle calculation with supplied lag/moments. It omits learned gates,
low-rank projections, state-update dynamics, chunkwise kernels, backpropagation,
training, throughput and model benchmarks. There is no claim of a full-paper
reproduction, speedup, improved learned recall or demonstrated context length.
