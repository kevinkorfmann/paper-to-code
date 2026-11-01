# Switching Linear Attention: different experts for different coordinates

An original educational implementation of the **ungated SwiLA write and
read recurrence**, equations 3.3–3.5 of Lee et al.,
[Switching Linear Attention](https://arxiv.org/abs/2609.39034v1).
The primary arXiv v1 was first posted **September 30, 2026**; this example
was added November 1. The [paper's equations and Appendix B](https://arxiv.org/html/2609.39034v1)
define a mixture of linear regressors with a separate expert assignment
for each output coordinate. Mixture regression and online EM themselves
predate this sequence-layer application.

The idea is to weight each memory's update by how well it already predicts
the incoming value, together with an input-dependent prior. Separate query
priors combine the updated memories at retrieval. Our tiny example supplies
all projections, priors, learning rates and initial states directly; none
is learned. It is not a full-paper reproduction or a trained-model comparison.

## Reproduce

**CPython 3.14.5**, standard library only, no randomness or seed. From the
repository root, using the version pinned by [.python-version](../../.python-version):

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/switching_linear_attention -v
python -m examples.switching_linear_attention.demo --check-sample
```

[sample_output.json](sample_output.json) contains the actual executed output.
The check compares it byte for byte; floating-point results are rounded to
12 decimal places. The existing [CI workflow](../../.github/workflows/tests.yml)
runs both commands in its standard-library job. No new packages, services
or model downloads are required.

## Equations and ordering

Let `W[j][d]` be a row vector for expert j and output coordinate d. Each
expert has a D×D matrix; the state is J×D×D. For a supplied key k, value v,
write-prior logits Lk and learning rates beta:

```text
delta[j,d] = v[d] - dot(W_old[j][d], k)
pi_k[:,d] = softmax(Lk[:,d])
r[:,d] = softmax(Lk[:,d] - 0.5*delta[:,d]^2)
W_new[j][d] = W_old[j][d] + beta[j,d]*r[j,d]*delta[j,d]*k
```

Responsibilities r are the posterior weights of a unit-variance Gaussian
mixture. The code's logit expression equals normalizing
`pi_k[j,d]*exp(-delta[j,d]^2/2)`. Normalization is **across experts separately
for each coordinate**, not across coordinates and not once for the whole
value vector. All errors and responsibilities use the same old state.
This is the paper's one-step online likelihood-gradient update, not a
fully converged batch EM M-step.

After writing, a query q with separately supplied logits Lq reads:

```text
pi_q[:,d] = softmax(Lq[:,d])
output[d] = sum_j pi_q[j,d]*dot(W_new[j][d], q)
```

Retrieval uses query priors, not the value-conditioned write posterior.
`run()` implements an inclusive causal scan: write token t, then read
its query. There is no access to later tokens. The paper describes learning
these projections and rates from inputs; this example only accepts their
supplied values. Rates are restricted to [0,1], including endpoints as
limiting/control cases of the paper's sigmoid parameterization.

## Executed example and counterexamples

Use two experts, two coordinates, key=query=[1,0], target value=[1,1],
uniform write priors and beta=1. The initial matrices are:

```text
expert 0: [[ 2,0], [-2,0]]
expert 1: [[-2,0], [ 2,0]]
```

Their prediction errors are [-1,3] and [3,-1]. Thus the responsibilities are:

```text
expert 0: [0.982013790038, 0.017986209962]
expert 1: [0.017986209962, 0.982013790038]
```

The larger responsibility is exactly `1/(1+exp(-4))`. A single joint expert
assignment based on the sum of the two squared errors would tie, since
both sums are 10. SwiLA instead favors a different expert for each
coordinate. Updated first-column entries are [1.017986209962,−1.946041370114]
for expert 0 and the reverse for expert 1; the unused second column stays zero.

Keeping these updated memories fixed:

| Supplied query probabilities, by output coordinate | Output (both coordinates) |
| --- | ---: |
| 90% to the coordinate's better-predicting expert | 0.721583451955 |
| Uniform 50% / 50% | −0.464027580076 |
| Reversed 10% / 90% | −1.649638612106 |

These are deliberately constructed query priors, not evidence that a
trained model learns to route correctly. Write specialization alone does
not guarantee useful retrieval; even the favorable read is not exactly
the target value 1.

Two controls expose limitations:

* **Identical experts can remain identical.** Start both at zero, use
  uniform write priors and equal rates, and repeat the unit-key/value
  input three times. Both states remain equal; readouts are [0.5,0.5],
  [0.75,0.75], then [0.875,0.875]. Query priors cannot distinguish equal
  readouts. Asymmetric write priors can break this symmetry (tested).
* **A bounded rate is not enough for stability.** One expert, initial
  weight 0, scalar key=2, target=1 and beta=1 gives weight 2 and readout 4.
  Squared error rises from 1 to 9. In general the component's current
  residual is multiplied by `1-beta*r*||k||²`; a unit key and beta in
  [0,1] contract this residual, but a large key can overshoot. This
  statement concerns the current pair, not retention or future losses.

## Validation and scope

The **32 tests** include an independent **80-digit Decimal** oracle that
forms Gaussian likelihoods and explicit normalized priors directly, then
evaluates multiple updates and readouts. A separate finite-difference
gradient of the marginal log likelihood checks every update entry.
Analytical tests cover the logistic responsibility, the single-expert
Delta Rule limit, zero rates/keys, exact prediction, symmetry and symmetry
breaking, orthogonal key directions, coordinate/expert permutations,
shift invariance, causal prefixes, chunked streaming, sequence order,
invalid inputs, nonmutation of inputs, CLI and checked output.

Softmax is evaluated with a max shift. This is ordinary binary64
arithmetic; unrepresentable intermediate dot products, squared errors or
updates raise `ValueError` instead of silently emitting nonfinite state.
Finite values can still round or underflow. We do not promise arbitrary
precision or robustness to adversarial scales.

Only the basic ungated recurrence is implemented. Temporal persistence,
state-decay variants, load balancing, learned projections, multiple heads,
backpropagation and optimized parallel/GPU kernels are omitted. With fixed
J,D, recurrent state has J×D² entries and update work does not grow with
history length; the convenience `run()` still stores its returned outputs.
This does not establish runtime or memory improvements over optimized
attention, unlimited capacity, convergence, or language-model gains.

The original code and constructed examples use the repository's
[MIT license](../../LICENSE). The cited paper is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its equations
are attributed above. No upstream implementation, figures, model weights
or datasets are copied or bundled.
