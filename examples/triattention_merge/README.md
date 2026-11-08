# TriAttention: precision groups need one denominator

Source: Hanzhi Zhang et al., [TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference](https://arxiv.org/abs/2611.07256v1), preprint first posted **5 November 2026**. Read [equations 2–3](https://arxiv.org/html/2611.07256v1#S3).

TriAttention assigns tiles to 8-bit, 4-bit or skipped computation. Two distance boundaries describe each head's regions, with first and diagonal blocks protected. Its precision groups contribute to one normalized output. This original CPU example implements the boundary rule and shared-state merge, using supplied logits in common units. It does not implement quantization, calibration, GPU kernels, KV caches or a trained model, and claims no speed or model-quality improvement.

## Run

CPython **3.14.5**, standard library only, no downloads or randomness in the demo:

```sh
python -m unittest discover -s examples/triattention_merge -t . -v
python -m examples.triattention_merge.demo
python -m examples.triattention_merge.demo --check-sample
```

## Constructed experiment

One group has logit log(9), value 0. A second has logit 0, value 10. Their unnormalized weights are 9 and 1. Shared normalization gives **1**; equally averaging their separately normalized outputs gives **5**. Dropping the second group gives **0**. Perturbing its logit by +0.1 instead gives **1.093668704**. The perturbation is an arbitrary supplied score error, **not INT4 emulation** or evidence that quantization is always preferable to skipping.

For each group, store its maximum m, relative exponential mass l and weighted value sum o. To merge, align maxima before adding masses and numerators, then divide once. Groups must be disjoint and scores/values must already share units. The merge is generic online-softmax algebra, not a new invention of this paper.

For query block 6, boundaries (2,4) produce actions [8-bit, skip, skip, 4-bit, 4-bit, 8-bit, 8-bit]. Future blocks are masked; within-diagonal token causality must additionally be handled by a real kernel.

## Validation and limits

22 tests include a separate 60-digit Decimal reference, every contiguous partition of five keys, order/association invariance, extreme scores, invalid inputs and checked CLI output. Randomized test inputs use seed 20261108. Values are scalar to keep the mechanism inspectable. Very large sums may be rejected on overflow; floating-point equality is tested with tolerances. The grouping operation alone does not establish causality of earlier score preprocessing. All data and implementation are original synthetic material under the repository MIT license, with no upstream code or paper figures copied.
