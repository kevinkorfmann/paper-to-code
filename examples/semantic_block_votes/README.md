# Semantic-block routing: vote after softmax, not before

Source: Kobayashi, Yamaguchi and Tokui, *Coarse Semantic Blocks for Efficient
Training-Free Sparse Attention*, [3 November 2026 preprint](https://arxiv.org/abs/2611.04313v1),
Section 3, especially Equations 2–4.

For each grouped-query-attention head, the method softmaxes cosine scores over
unit centroids, then sums these probabilities across heads. Dividing by
`(1 + visible_historical_block_size)^beta` penalizes larger blocks. Whole blocks
are accumulated until their token count reaches or exceeds the historical
budget. Recent tokens are included separately. Attention uses original Q/K/V,
not the normalized vectors used for routing. This original component reference
accepts supplied centroids and assignments; it does not implement clustering.

## Run

CPython 3.14.5, standard library only. Deterministic, no RNG or external data.
From repository root:

```sh
python -m unittest examples.semantic_block_votes.test_core -v
python -m examples.semantic_block_votes.demo
python -m examples.semantic_block_votes.demo > /tmp/semantic-output.json
diff -u examples/semantic_block_votes/sample_output.json /tmp/semantic-output.json
```

## Our constructed experiment

Two heads point along the coordinate axes. Three unit centroids point along
the x axis, the y axis, and the 45-degree diagonal. Every cluster has two tokens,
so size penalties are equal. With temperature 0.1, each head assigns about
0.949217 to its own axis, 0.050740 to the diagonal, and 0.000043 to the other
axis. Summing these votes selects an axis cluster (index 0 under our tie rule).

By contrast, adding cosine logits **before** softmax assigns the diagonal
0.969201 probability and selects that cluster. This comparator is deliberately
a different operation, not a proposed replacement or implementation of the
paper. Neither outcome establishes which routing choice is empirically better.
The example isolates the difference between a mixture of per-head probability
rows and a product-like aggregation of their unnormalized scores.

The requested historical budget is one, but the selected block contains two
tokens: indices [0,3]. Our causal/recent control has nine assigned tokens but
query position 4 (zero-based), with position 4 included without selection.
Only indices [0,1,3,4] are returned; visible historical block sizes are [2,1,1].
The final block may exceed the historical budget, and recent tokens do not
count against it. Keeping all blocks recovers dense attention in our control.

## Tests and limits

Twenty-six tests cover independent 80-digit Decimal vote/attention formulas,
head permutations, query scaling for routing but not attention, size penalties,
empty blocks, budget overshoot, recent/future exclusion, all-key recovery,
constant values, invalid inputs, and exact CLI/output replay.

This is one query position and a supplied shared KV partition. Centroids and
labels are fixed, not learned or fitted by spherical k-means. A causal index
filter is **not** proof of causal independence of the routing index: fitting
centroids to an entire prompt can make earlier selections depend on later
prompt keys, as the paper explicitly notes. We test only fixed-index masking.
No RoPE, GPU grouping, prefill pipeline, training, speed benchmark or model
quality experiment is included. No upstream code or weights were copied.
