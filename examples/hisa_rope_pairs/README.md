# HiSA: when a key loses its RoPE pairs

This is an original, standard-library Python illustration of the exact score
and causal pair-count rules in §3.2 (Eqs. 3–5) of [Borbely, Salzmann & Verscheure,
*HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs*
(arXiv v1, 19 November 2026)](https://arxiv.org/html/2611.26363v1#S3.SS2).

In a head of even dimension `d`, the score combines `d/2` rotary pairs.
Pair `l` has rotation frequency `base**(-2*l/d)`. At causal distance
`delta = query_pos - key_pos`, HiSA adds the rotated inner product of that
pair only when `delta*frequency < angle_budget`. The denominator remains
`sqrt(d)`, not the square root of the number of retained coordinates.
A key that has no active pairs is **removed** from the softmax. A retained key
with zero score still has positive softmax weight. Infinite budget recovers
ordinary dense causal RoPE attention for these inputs.

## Reproduce

From the repository root, with CPython 3.14.5:

```sh
python -m unittest examples.hisa_rope_pairs.test_demo -v
python -m examples.hisa_rope_pairs.demo > /tmp/hisa-rope-pairs.json
diff -u examples/hisa_rope_pairs/sample_output.json /tmp/hisa-rope-pairs.json
```

The fixture is deterministic, with no RNG, model weights or data downloads.
The tests independently rotate query and key at absolute positions using
complex numbers (270 comparisons, seed `20261122`) and enumerate 180 tiny
causal work sets to check the distance-grouped count. They also cover strict
threshold boundaries, future and expired keys, and replay the saved JSON.

## What the run shows

We construct one four-coordinate query with four keys at distances 0, 2, 6
and 7. At toy base 16 and a 90° budget, the two rotary frequencies are
1 and 0.25 radians per position. At distance 6 the fast pair has rotated 6
radians, or about −0.283 radians modulo one turn; despite returning close to
its initial orientation it stays masked, because HiSA tests the **unwrapped**
angle. The slow pair still keeps that key in the softmax.

| Distance | Active pairs | Logit | Scalar value |
| ---: | --- | ---: | ---: |
| 0 | fast, slow | 1 | 0 |
| 2 | slow | 0 | 10 |
| 6 | slow | 0 | −4 |
| 7 | none | not in softmax | 100 |

For this fixed query and keys the partial-score readout is **1.271649346**
versus **25.713825818** with infinite budget. The retained zero logits
are not masks: their values contribute to the HiSA output. In a separate
fully populated length-eight causal counting grid, the same angle rule
evaluates **50 of 72** possible pair products. The grid counts permissible
feature-pair inner products, not arithmetic instructions, memory traffic,
or runtime.

These are supplied, deliberately simple vectors with no trained projections.
This is not the block-rounded Flash-HiSA kernel, a KV-cache implementation,
a trained-language-model reproduction, or evidence about model quality or
speed. The dense contrast highlights the rule in a toy; it is not a benchmark
against the trained models in the paper.
