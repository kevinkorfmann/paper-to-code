# Transformer-FSQ: pool cached codes before decoding values

Szecsenyi et al., [Transformer-FSQ: Discrete Latents for Extreme KV-Cache Compression](https://arxiv.org/html/2611.09598v1), preprint v1 dated **6 November 2026**. This original standard-library example illustrates the forward map in Eq. 2 and the value-side identity in Eq. 6, not the pretrained-model conversion.

## Operator

For each supplied encoder coordinate u, set h = 7.5075 and delta = 0.5. Compute tanh(u + atanh(delta/h))*h − delta, then round to nearest, ties to even. The signed levels −8,…,7 map to unsigned 0,…,15 by adding 8. Each token uses ceil(width/2) bytes; this toy puts the first coordinate in the low nibble and pads an odd last coordinate with a zero high nibble. Decode back to **signed** levels before projections. Width is supplied once, not stored per token.

One query scores reconstructed keys, using ordinary scaled dot-product attention over the supplied visible prefix. For fixed weights a and the same quantized signed codes z, linearity gives:

```text
sum_t a_t * (W_v z_t) = W_v (sum_t a_t z_t)
```

The right side applies the value projection once for this query head, instead of separately to every visible token. The equality does not require identical attention weights across different queries or heads. It does require that the two sides being compared use the same weights.

## Actual output

- Three width-3 token codes pack into six payload bytes (two per token, including odd-width padding). An illustrative FP16 two-key/two-value-coordinate cache instead has 24 payload bytes for those tokens. This compares payload only, not total memory, metadata, decoder parameters, or measured throughput.
- Decode-then-pool and pool-then-decode both print `[-1.280484137526, 4.606237528119]`. They are algebraically equal; floating roundoff can differ before rounding.
- Skipping rounding while keeping the same bounded latents and decoders gives `[-1.191451035016, 4.387169954274]`. The maximum coordinate gap is **0.219067573846**: the exact value-pooling identity does not remove quantization error. The continuous reference recomputes attention weights and is not an original pretrained-model cache.

## Run and limits

CPython **3.14.5**, standard library only, from the repo root:

```sh
python -m unittest examples.fsq_latent_cache.test_demo -v
python -m examples.fsq_latent_cache.demo > /tmp/fsq-output.json
diff -u examples/fsq_latent_cache/sample_output.json /tmp/fsq-output.json
```

Inputs and projections are handwritten. No learned encoder, gradient estimator, key normalization, positional encoding, fused kernels, trained model, or benchmarks. The research paper reconstructs and applies positional operations to keys before scoring; the pooling shortcut demonstrated here is **value-side**, not a claim that key reconstruction can be moved past softmax. The implementation unpacks all supplied codes for clarity and is not a memory-efficient production kernel. No source code or data was copied; equations are attributed to the linked CC BY 4.0 paper.
