# Bilateral buckets: early versus recent interference

Reference: Islam and Cong, [Bilateral Log-Linear Attention](https://arxiv.org/html/2611.18974v1), v1, 13 November 2026, §§2.2–3.2.

The paper complements a forward Fenwick hierarchy with origin-anchored buckets. “Reverse” does **not** mean future access. This example implements only the raw reverse recurrence (Eqs. 8–13): token `s` writes `eta_s v_s k_s^T` into level `floor(log2(s+1))`; each query reads the visible states with nonnegative level weights. Writes happen before reads. A write gate is applied once, not repeatedly as temporal decay. State storage is logarithmic at fixed feature dimensions.

## Run

From the repository root, using CPython 3.14.5 (standard library only):

```sh
python -m unittest examples.bilateral_buckets.test_demo -v
python -m examples.bilateral_buckets.demo > /tmp/bilateral.json
diff -u examples/bilateral_buckets/sample_output.json /tmp/bilateral.json
```

## Our synthetic experiment

At position 14, give two tokens identical scalar keys and opposite scalar values. Hand-select the bucket containing the target, setting other read weights to zero. This is an **oracle intervention**, not a learned ability:

- Target 0, distractor 1: forward sum cancels to 0; reverse gives 1.
- Target 14, distractor 13: forward gives 1; reverse cancels to 0.
- Target 4, distractor 5: both cancel to 0.

Thus complementary grouping can isolate either endpoint in this construction but does not eliminate interior interference. Zero level weights are a diagnostic boundary case, not the paper's strictly positive learned parameterization. After 1024 writes the implemented reverse branch has 11 state matrices. The dense, token-by-token test oracle agrees with recurrent reads under randomized features, gates, and query-dependent weights.

## Scope

No model training, forward online recurrence, decay, normalization, branch mixing, GPU kernel, timing or benchmark reproduction. Forward buckets are explicitly enumerated for illustration; that helper does not have logarithmic memory use. Only `ReverseMemory` has logarithmic persistent state. Handwritten inputs and gating cannot establish language-model quality or speed.
