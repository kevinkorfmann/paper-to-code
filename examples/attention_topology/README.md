# Attention topology: same merge features, different readout

**Paper:** Saeed AlMarri, [Zero-Dimensional Attention Topology for Linguistic Acceptability](https://arxiv.org/abs/2611.22312v1), arXiv preprint, posted **17 November 2026**. Sections 4–5 specify the representation and analyze its limitations. This is an original educational implementation, not the authors' code or a reproduction of their BERT/CoLA experiments.

## Mechanism

For a nonnegative square attention matrix with positive row sums, normalize each row to obtain P. Set D_ii = 0 and D_ij = 1 − (P_ij + P_ji)/2. Vertices are present at threshold zero, and edges appear when D_ij ≤ threshold. The finite zero-dimensional persistence death times are the sorted minimum-spanning-tree edge weights; the one component that persists forever is omitted. A triangle inequality is not required.

We compute the paper's ten summaries: mean, population standard deviation, five linearly interpolated quantiles (10/25/50/75/90%), maximum, sum and entropy of normalized deaths. Zero total gives zero entropy. Kruskal's algorithm is adequate for this tiny complete graph; this implementation is not a scalable attention extractor.

## What the original fixture shows

The paper already establishes both directional information loss under symmetrization and the algebraic token-count identity. Our **nonuniform four-token** fixture uses a symmetric row-stochastic baseline plus a skew-symmetric circulation around a four-cycle, with amplitude 1/16. This preserves row sums and nonnegativity, and cancels in P + Pᵀ. Unlike the paper's uniform three-token illustration, we explicitly compare scalar attention-value products with values [0, 1, 2, 3]:

| Readout | Token 0 | Token 1 | Token 2 | Token 3 |
| --- | --- | --- | --- | --- |
| Baseline AV | 1.125 | 1.250 | 1.625 | 2.000 |
| Circulated AV | 1.000 | 1.375 | 1.750 | 1.875 |

Both have identical D, finite merge values **[0.750, 0.750, 0.875]**, and all ten identical summary features. Thus these features cannot reconstruct arbitrary attention readouts. This is a representation counterexample, not evidence against usefulness for a particular prediction task.

There are n−1 finite deaths. If their sum S is positive, n = 1 + S/mean(deaths); if S=0, the ratio is undefined. The JSON recovers n=4 from the fixture. This identity does not establish that a downstream linear classifier exploits token count. Tests additionally verify uniform attention's analytical length dependence, credited to the paper.

## Reproduce

CPython **3.14.5**, standard library only; from repository root:

```sh
python -m unittest examples.attention_topology.test_demo -v
python -m examples.attention_topology.demo > /tmp/topology.json
diff -u examples/attention_topology/sample_output.json /tmp/topology.json
```

Eleven tests cover the hand-calculated readouts/features, an independent threshold/flood-fill connectivity oracle on 70 seeded random matrices, ties, zero deaths, uniform-attention formulas, vertex permutations, row rescaling, invalid inputs and byte-for-byte CLI replay. Assertions use unrounded results; JSON floats are rounded to 12 decimals. The reference interpreter is checked by repository Actions, and this example is wired into its tests and replay steps.

No text data, tokenizer, padding removal, trained model, persistence library or classifier is included. This tiny toy makes no accuracy, speed, memory or empirical interpretability claims. Finite, ordinary-sized teaching inputs are assumed; extreme floating-point ranges are not a supported numerical benchmark.
