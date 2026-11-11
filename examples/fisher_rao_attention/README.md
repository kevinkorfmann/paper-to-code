# Fisher–Rao attention: average logits, not probabilities

An original, deterministic single-row example for **11 November 2026**.

## Paper and scope

Aythami Bethencourt de León and Darryl D. Holm,
[A Geometric Theory for Nonequilibrium Transformer Dynamics](https://arxiv.org/abs/2611.10691v1),
**preprint v1, submitted 7 November 2026**. It is a theoretical proposal, not
an experimentally validated replacement for trained attention.

The [primary text](https://arxiv.org/html/2611.10691v1) gives the logit lift
in equation (4.4), its convolution solution in (4.6), the free-energy/KL
identity in (2.3), and the driven energy balance in (4.9). We implement only
one unmasked row with externally supplied constant or affine scores. The
paper connects its row flow to earlier entropy-adjusted replicator dynamics;
this example does not claim the underlying filtering idea is new.

## Run

Use **CPython 3.14.5**, the interpreter pinned at the repository root.
From a fresh repository checkout:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/fisher_rao_attention -t . -v
python -m examples.fisher_rao_attention.demo
python -m examples.fisher_rao_attention.demo --check-sample
```

On Windows use `.venv\Scripts\Activate.ps1`. No packages, model weights,
datasets, random sampling, network calls or paid APIs are needed. JSON output
is rounded to nine decimal places and checked against
[sample_output.json](sample_output.json). Tests use unrounded calculations.

## Implemented equations

For a score row `s`, temperature `tau > 0`, mobility `gamma > 0`, logits `u`,
and probabilities `p = softmax(u)`:

```text
u' = gamma * (s(t) - tau*u)
F(p;s) = sum_j p_j * (tau*log(p_j) - s_j)
p'_j = gamma*p_j * [(s_j - tau*log(p_j))
                   - sum_k p_k*(s_k - tau*log(p_k))]
```

Our affine-score specialization takes `s(t)=s0+d*t` over one interval of
length `h`. Direct integration of the linear ODE gives:

```text
kappa = gamma*tau
r = exp(-kappa*h)
u_next = r*u + (1-r)*s0/tau
         + (d/tau)*(h - (1-r)/kappa)
```

This is an exact-in-time solution for the stated forcing, subject to ordinary
binary64 rounding. It is not an Euler update. Constant scores set `d=0`.
We initialize with `log(p0)` and remove common shifts from logits, scores and
slopes, which changes the logit representative but not its probabilities.
The ramp coefficient uses `expm1` and a small-argument series to avoid
subtracting nearly equal numbers.

## Three constructed controls

**1. Filtering before softmax is not filtering after it.** Start at `[0.5,0.5]`
and use scores `[0,log(9)]`, with temperature and mobility both one. The target
is `[0.1,0.9]`. After one logit half-life, `h=log(2)`, the exact probabilities
are **`[0.25,0.75]`**, while halfway averaging the two probability rows gives
**`[0.3,0.7]`**. Equivalently, the exact update is a normalized geometric
blend of the initial and target probabilities. With scalar values `[0,1]`,
these are attention readouts `0.75` and `0.7`.

**2. A moving objective need not decrease.** Keep the same uniform initial
row but supply the smooth score ramp `[0,-t]`. At `t=1`, free energy is
**−0.267458687**, up from **−0.693147181**. The instantaneous balance is
score work **0.409053523** minus dissipation **0.096589099**, giving derivative
**+0.312464423**. This does not contradict descent for a fixed score row.
The KL to the moving target grows from zero to **0.045803**.

The ramp also exposes persistent lag: at `t=5`, the second probability is
**0.017867585**, versus instantaneous softmax **0.006692851**. Their log-odds
differ by **0.993262053**. For this chosen ramp the gap approaches one even
as both second probabilities approach zero. Log-odds lag and absolute
probability error are different quantities.

**3. Naive probability Euler can leave the simplex.** With the fixed scores
from control 1, an explicit probability-coordinate Euler step of size 2
produces **`[−0.598612289,1.598612289]`**. The exact logit step gives
**`[0.130124030,0.869875970]`**. This is a deliberately unsuitable step-size
control, not a claim that all numerical ODE methods fail.

## Tests and limits

Tests compare independent high-precision and probability-coordinate references,
check interval composition, invariances, the energy balance, fixed-target
relaxation, validation and the sample/CLI contract. CI runs this suite and
checks the output together with the existing attention examples.

The API accepts finite numeric vectors of at least two entries. Initial
probabilities must be strictly positive and sum to one within `1e-12`.
There is no mask or token insertion. Time is nonnegative, and temperature
and mobility must be positive. Overflowing intermediate arithmetic is rejected.
Tiny time-rate products that underflow are rejected rather than silently
treated as valid integration steps. Extremely small softmax tails may round
to zero, although their finite log masses remain in the logit state. Thus the
paper's mathematical strict positivity is not a binary64 output guarantee.
Do not reconstruct recurrent state from rounded demo probabilities.

Each step is O(n) work and O(n) state for this one row. A full dense set of
attention rows still takes O(n²) state. This example does not implement learned
Q/K/V projections, the coupled Hamiltonian system, causal autoregression,
backpropagation, training, GPU kernels, or an LLM. It establishes no quality,
retrieval, speed or memory advantage over a trained model.

All code, fixtures, derivations and wording here are original for this
repository and covered by its MIT license. No paper text, figure or external
implementation is bundled. The cited paper separately lists CC BY 4.0.
