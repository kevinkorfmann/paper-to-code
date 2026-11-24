# Depth KV replay: fresh queries, older keys and values

Reference: Licakova et al., [Architectures and KV Replay for Depth-Recurrent Language Models](https://arxiv.org/html/2611.19446v1), v1, **14 November 2026**, Eq. 1 and §§3.3–3.4. This original toy implements their deterministic stride policy on supplied synthetic activations, not the trained model.

At recurrence iteration r and logical attention layer l, stride s selects cache group floor(r/s). Iterations divisible by s write the keys and values for their group and layer. Other iterations skip the KV projection and reuse that writer's exact tensors, but use **their own current query**. This example uses (group, layer) tuples in place of flattened slots. One scalar attention head reads a specified visible prefix. The cache is local to one forward trace, not an autoregressive generation cache.

## Diagnostic

Five iterations, one layer, two visible keys −1 and +1, queries [2, −2, 2, −2, 0]. At each iteration the fresh value for key +1 would be [1, 10, 2, 20, 4]; the other value is zero. An oversized suffix token is ignored.

| Stride | Writer used by each iteration | KV projection calls | Output at iteration 1 (zero-based) |
| --- | --- | ---: | ---: |
| 1 | 0, 1, 2, 3, 4 | 5 | 0.179862 |
| 2 | 0, 0, 2, 2, 4 | 3 | 0.017986 |
| 5 | 0, 0, 0, 0, 0 | 1 | 0.017986 |

At iteration 1 the query flips sign even when KV is replayed, so its output does not equal iteration 0's output. Stride 2 agrees with stride 1 again at iteration 2, a writer. Sharing projection parameters across iterations would not by itself make their activations identical. All outputs and writer call indices are in the checked sample.

## Reproduce

CPython **3.14.5**, standard library only, from the repository root:

```sh
python -m unittest examples.depth_kv_replay.test_demo -v
python -m examples.depth_kv_replay.demo > /tmp/depth-kv-replay.json
diff -u examples/depth_kv_replay/sample_output.json /tmp/depth-kv-replay.json
```

Eight tests cover a direct scalar softmax oracle, partial stride groups, layer-specific slots, causal-prefix isolation, fresh queries, skipped reader projections, extreme logits, invalid controls, and cache isolation between forward traces.

## Limits

Queries and potential KV are supplied directly, not produced by evolving hidden states. In a real depth-recurrent network, changing a reader's output can affect later writer activations too. No backpropagation, training, generation cache, GPU kernel, memory benchmark, or model-quality claim. Counts are toy-layer projection calls, not measured runtime or memory savings. The preprint is CC BY 4.0; no paper code or data was copied.
