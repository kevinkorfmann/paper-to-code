# One shared attention route, two layer preferences

[Sun et al., *You Only Index Once*](https://arxiv.org/html/2606.06467v1#S2)
(June 4, 2026 preprint) use one learned token-level top-k index across
cross-decoder layers sharing a key/value cache. Each layer still uses its
own query to attend over the selected tokens (equations 2–4). During
distillation, the teacher target averages dense attention probabilities
over layers and heads (equation 5); the indexer learns to approximate it.

This original toy supplies that teacher target directly, as an oracle for
what a well-matched indexer might rank. With a three-token causal prefix
and two single-head layers, shared top-1 retains dense mass 0.65 and 0.10.
It maximizes the mean retained mass: the mass for a shared subset is the
sum of the mean probabilities of its tokens. It need not maximize the
lowest layer's mass. An exhaustive maximin diagnostic picks a different
token and retains 0.25 and 0.30. It is a contrast, not part of the paper.
Recompute softmax weights over the selected subset, equivalently
renormalizing the selected dense weights. At full budget every method
recovers the dense outputs.

This example does not implement training, the self-decoder, or a GPU
kernel. Computing the dense teacher target here is for illustration
only, not the inference procedure. Retained mass and output change
are different quantities, and the toy predicts neither real model
accuracy nor wall-clock speed.

From the repository root, with CPython 3.14.5 and no third-party dependencies:

```sh
python -m unittest examples.shared_routing_consensus.test_demo -v
python -m examples.shared_routing_consensus.demo --check-sample
```

No random draws. [Output snapshot](sample_output.txt) covers budgets 1, 2 and 3.
