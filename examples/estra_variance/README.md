# ESTRA variance compensation: helpful spread, harmful proxies

Reference: Liu et al., [ESTRA: Enabling Extreme Attention Sparsity for Training-Free Video Diffusion Acceleration](https://arxiv.org/html/2611.10448v1), v1, **7 November 2026**, §4 Eqs. 6–8. This original scalar-value tutorial implements only variance-calibrated mean (VCM) compensation for supplied clusters and selected indices.

## Implemented operator

For each cluster, keep its **full-cluster key and value means**, even after selecting some tokens. Let `r` be the remainder size and `m2` its average squared key distance from that fixed center. For head dimension `d` and all supplied queries:

```text
alpha = mean(||q||²)/(2*d²)
delta = min(alpha*m2, cap)
representative_logit = q·mean_key/sqrt(d) + log(r) + delta
```

Empty remainders are omitted. One stable softmax jointly normalizes selected tokens and representatives. Setting the cap to zero gives the matched mean-compensation baseline. The correction uses an isotropic spread proxy, omits remainder mean shift, and shares alpha across the head. It is not exact query-direction variance. This is bidirectional video-attention machinery, not automatically a causal LLM recipe.

## Original diagnostic fixtures

All values and clusters are handwritten, with cap 1. The three cases isolate what the correction can and cannot capture:

| Case | Dense output | Mean compensation | VCM |
|---|---:|---:|---:|
| Symmetric scalar spread | 0.692804 | 0.666667 | 0.693843 |
| Spread perpendicular to query | 0.666667 | 0.666667 | 0.693843 |
| Selection shifts remainder mean | 0.755272 | 0.788058 | 0.813676 |

In the first case, cluster keys are ±0.5 with values 1; an exactly selected zero key has value 0. Exact cluster mass is `2*cosh(0.5)`, versus 2 for the baseline and `2*exp(0.125)` for VCM. The correction reduces absolute output error from 0.026137 to 0.001039.

In the second, keys vary only along a direction orthogonal to the query, so their logits do not vary. Their norm spread nevertheless triggers a positive correction, making the previously exact baseline worse.

In the third, selecting the positive key leaves a negative key behind. The fixed center remains zero. A positive correction cannot account for that negative mean shift and increases the error. These counterexamples are diagnostics, not representative model-quality estimates or a refutation of the paper's empirical findings.

## Reproduce and verify

CPython **3.14.5**, standard library only, from repository root:

```sh
python -m unittest examples.estra_variance.test_demo -v
python -m examples.estra_variance.demo > /tmp/estra.json
diff -u examples/estra_variance/sample_output.json /tmp/estra.json
```

Twelve tests cover independent scalar closed forms, randomized all-selected equality to dense attention, singleton and identical-key exactness, full-cluster values, head-average alpha, selection mean shift, cap behavior, permutations and invalid input. JSON floats are rounded to 12 decimals; tests compare unrounded values.

No clustering, sampled threshold selection, training, model weights, GPU kernel or video generation is implemented. Selected indices are shared by the supplied query batch (one query-cluster fixture). Scalar values are for clarity. No speed, memory or generation-quality claim is made. Inputs are small finite teaching examples, not arbitrary-scale numerical stress tests.
