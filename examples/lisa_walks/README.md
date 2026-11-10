# LiSA walks: truncation is a modeling choice

Source: Sahar Dastani et al., [LiSA: Linear Stochastic Attention for Long-Range Visual Modeling](https://arxiv.org/abs/2611.04216v2). Preprint first posted **3 November 2026**, revised **4 November 2026**. The supplied-factor calculation follows the geometric global branch in [equations 4, 6 and 8](https://arxiv.org/html/2611.04216v2#S3.SS2). This is an original educational implementation, not upstream code.

## Run

CPython **3.14.5**, standard library only. No model weights, downloads or randomness in the demo.

```sh
python -m unittest discover -s examples/lisa_walks -t . -v
python -m examples.lisa_walks.demo
python -m examples.lisa_walks.demo --check-sample
```

## Mechanism

Let nonnegative, already transformed feature matrices Q and K define a transition S by row-normalizing Q K^T. For positive row masses, S is stochastic. Mix powers S^0 through S^(T-1) with weights proportional to beta^k, for 0 <= beta < 1. The zero-hop term retains a token's own value. We sum the finite geometric powers directly to avoid cancellation near beta=1.

The implementation never forms S in the main computation. Write Qtilde = D^-1 Q, B = K^T Qtilde and Z = K^T V. The h-hop contribution is Qtilde B^(h-1) Z for h >= 1. We propagate Z in feature space; an explicit token-space transition is available only as a diagnostic. For fixed feature/value widths and T this scales linearly in token count, but this pure-Python toy is not a performance benchmark.

## Constructed experiment

Three nodes follow the directed transitions **0 → 1 → 2 → 2**, with values **[0, 0, 1]**. One hop cannot bring the last value to node 0. With beta=1/2 and T=3, the normalized hop weights are **[4/7, 2/7, 1/7]**, so the output is **[1/7, 3/7, 1]**. No dependency is learned here: Q and K explicitly encode this graph.

The finite normalization has a precise interpretation: start with a geometric stopping length L and *condition on L < T*. It is not the same as forcing every walk still active to stop at T-1. Forced stopping gives **[1/2, 1/4, 1/4]** and output **1/4** at node 0. The infinite walk also gives 1/4 at node 0 in this particular absorbing graph, since every length >=2 ends at node 2. These are different definitions, not evidence that either is universally better.

A second control exposes stabilization: adding epsilon=1e-6 to each normalization denominator gives row sums slightly below one. Constant values then return **0.999999428572143**, rather than exactly one, in this example. The default is epsilon=0 with zero row masses rejected. Positive epsilon is supported to demonstrate this departure from exact stochasticity, not to claim a practical model-quality problem.

## Tests and scope

27 tests include an independent exact-rational enumeration of paths for **972 factor/beta/length cases**, 40 seeded multifeature cases (seed 20261110), epsilon controls, permutation and scaling invariance, constant preservation, convex-hull bounds, input rejection and exact CLI replay. Large or ill-conditioned finite inputs can overflow intermediate products and are rejected; this is not arbitrary-precision production numerics.

The operator is bidirectional, not a causal language-model layer. It omits learned projections, the local branch, learned step weights, training, gradients and optimized kernels. No empirical vision/LLM gains or speedups are established. All examples are synthetic; no private data or paper figures/text were copied. The original code is covered by this repository's MIT license.
