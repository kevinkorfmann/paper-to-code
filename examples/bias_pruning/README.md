# Bias pruning: average coverage is not worst-query coverage

Source: Herr, Spadaro and Tartaglione, *Exploiting Biases in Attention for
Transformers*, [3 November 2026 preprint](https://arxiv.org/abs/2611.04657v1)
(the arXiv record reports acceptance at WACV 2027), Sections 3.2–3.3.

The paper adds a per-key bias after dot-product scaling:
`softmax(Q K^T / sqrt(d) + b) V`. It selects a shared set of key columns by
top bias while keeping every query. The bias acts as a positive multiplicative
gate inside normalization, not an output gate. This original implementation
isolates that operation with supplied Q, K, V and b, not learned projections.
No upstream code, model weights or datasets are included.

## Run

CPython 3.14.5, standard library only. No installation or random seed needed.
From repository root:

```sh
python -m unittest examples.bias_pruning.test_core -v
python -m examples.bias_pruning.demo
python -m examples.bias_pruning.demo > /tmp/bias-output.json
diff -u examples/bias_pruning/sample_output.json /tmp/bias-output.json
```

## Original constructed experiment

There are 100 supplied scalar queries: 99 zeros and one -8. Keys are `[1,-1]`,
values `[1,-1]`, and biases `[4,0]`. A budget of one keeps the first key for
all queries. Dense attention here means the **same biased operator** without
pruning, not an unbiased baseline.

- Average retained dense-attention mass is **0.972193714**.
- For the exceptional query it is only **0.000006144**.
- That query's output changes from **-0.999987712 to +1**, nearly the full
  possible output range.
- Keeping both keys recovers the dense result exactly in this experiment.
- Setting both values to +1 makes pruning harmless, despite identical lost
  attention mass.

Thus neither a high mean retention nor lost mass alone determines every
query's output error. The two-key scalar identity is
`dense - kept = (1 - retained_mass) * (discarded_value - kept_value)`.
For values in [-1,1], absolute error is at most twice the lost mass.
These are our diagnostic consequences, not a benchmark from the source paper.
The chosen rare query is deliberately adversarial, not representative data.

## Verification and limits

Twenty-six tests include an independent 80-digit Decimal oracle, a 51-case
scalar grid, multi-dimensional queries/values, correct bias scaling,
multiplicative gating, pruning renormalization, key permutation, constant
values, all-key recovery, deterministic ties, finite-input checks and exact
CLI replay. Normalization uses max-shifted binary64 exponentials.

This is one unmasked head with supplied finite projections, not a complete
self-attention block. There is no training, learned bias calibration, causal
mask, layer-wise budget schedule, GPU kernel or timing benchmark. Keys and
values are already provided, so this code cannot measure projection savings.
The toy demonstrates a limitation of a proxy, not a contradiction of empirical
vision-transformer results or evidence of model-quality/speed gains.
