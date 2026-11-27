# Universal Attention: a veto on decay

This original scalar example isolates the decay combiner from equations 3–4 of
[Wertheimer et al., *A Self-Pruning Transformer*](https://arxiv.org/html/2610.09051v1#S4)
(arXiv preprint, October 6, 2026). Each input is a proposed decay fraction. Their
geometric mean is the combined decay, so a single zero vetoes decay at that
step. The logarithm of one minus this fraction is added to the key's previous
bias. Keys do not decay on their own step. Future positions are masked.

Four keys have equal content logits. In each later step, key 0 has gate fractions
(0.5, 0, 0.5), and all others have (0.5, 0.5, 0.5). Only key 0 has value 1.
Its survival over three steps is 1 with the geometric combiner and (2/3)^3 with
an arithmetic-mean control. The readout is its normalized softmax weight, not
its survival itself. See [actual output](sample_output.txt).

This example supplies gate values instead of learning them. Exact zero is a
boundary illustration: a sigmoid with finite logits only approaches it.
There is no trained attention layer, RoPE, eviction, or runtime measurement.
No model-quality or cache-compression result follows from this toy.

From the repository root, using CPython 3.14.5 and only the standard library:

```sh
python -m unittest examples.universal_decay_veto.test_demo -v
python -m examples.universal_decay_veto.demo --check-sample
```

Deterministic input; no RNG or data downloads.
