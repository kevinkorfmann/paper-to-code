# Causal index routing: masking future keys is not enough

An original, dependency-free mechanism demo motivated by **AttentionIndex:
Accelerating Learned Sparse Attention with a Causal Inverted-File Index**, by
Jeongmin Brian Park and Wen-mei Hwu. The [primary preprint, v1](https://arxiv.org/abs/2611.02508v1)
was first posted **November 2, 2026, 04:29:41 UTC**. This example was added
November 5, 2026.

## Paper connection and deliberate scope

[Equation 1 and Sections 4.1–4.2](https://arxiv.org/html/2611.02508v1#S4)
motivate two-stage candidate selection and causal index versions:

```text
indexer_score(key) = sum_h weight[h] * max(0, dot(query[h], key))
```

We use supplied clusters with arithmetic-mean representatives. A query reserves
candidate slots for sink and unclustered open-window keys, ranks cluster
representatives, fills the remaining budget, and exactly scores the candidates
before selecting top-k. The snapshot must have been formed from a prefix visible
to the query. This is **not full AttentionIndex**: its clustering, refits,
learned representative correction, model integration and kernels are omitted.
Inverted-file indexing and the native score predate this paper.

## Three constructed controls

1. **Exact refinement cannot rescue a missing candidate.** For one query head
   `q=1`, weight `1`, and keys `[9,-9,4,4]`, exhaustive scores are `[9,0,4,4]`.
   Groups `[9,-9]` and `[4,4]` have means `0` and `4`. With budget 2, mean routing
   chooses IDs `[2,3]`, then returns ID 2, missing the exact winner ID 0.
   Supplied regrouping into `[9,4]` and `[-9,4]` recovers ID 0 at the same budget.
   Budget 4 recovers it with either grouping. This is a single synthetic control,
   not evidence that similarity grouping always improves recall.

   Also, `score(mean([9,-9])) = 0`, but `mean(score([9,-9])) = 4.5`.
   A nonlinear indexer cannot generally commute with averaging.

2. **Future metadata can leak through a final key mask.** A visible prefix has
   singleton groups `[2]` and `[1]`, so budget 1 selects ID 0. If an invalid global
   index adds a future `-100` to the first group, its mean becomes `-49`.
   Routing then selects ID 1 even after masking all future token IDs.
   The safe API rejects that newer snapshot. Reusing the original snapshot and
   the original visible prefix returns ID 0 regardless of the later append.

3. **Reservation and truncation are not recall guarantees.** A sink and a newly
   appended key enter the candidate set first, but the sink can still lose the
   final score competition. A one-cluster `[1,10]` example with budget 1 loses
   the best key solely because the last list is truncated in token-ID order.

These examples implement **selection**, not an attention output. Indexer scores
are not attention probabilities and are not a measure of downstream accuracy.

## Reproduce in a clean environment

From the repository root with **CPython 3.14.5** installed:

```sh
python3.14 -m venv --without-pip .venv-causal-index
. .venv-causal-index/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s examples/causal_index_routing -v
python -m examples.causal_index_routing.demo
python -m examples.causal_index_routing.demo --check-sample
```

On Windows activate with `.venv-causal-index\Scripts\Activate.ps1` instead.
No packages, downloads, model weights, credentials, GPU or paid
service are needed. The root `.python-version` pins the interpreter and
`requirements.txt` records the empty dependency set. The committed
[sample output](sample_output.json) is actual deterministic JSON, compared
byte for byte by tests and CI. The demo uses no randomness; randomized test
fixtures use the fixed seed `261102508`. The checker never rewrites the sample.

## API contracts and limitations

- `PrefixIndex` copies keys and computes means. Cluster IDs plus explicit sink
  IDs must partition the indexed prefix without duplicates. Empty clusters are
  rejected. Snapshots use immutable nested tuples.
- `select` receives exactly the visible prefix. It rejects a newer snapshot,
  changed prefix values, dimension errors, and budgets too small for reserved
  keys. Appended visible keys form the open window and are candidates immediately.
- Callers must form **memberships and metadata** causally too. The API cannot
  detect whether externally supplied grouping decisions used future data.
  It does not implement sealing schedules, decode insertion, or index-version
  scheduling. This example starts after a supplied index has been built.
- Integers and floats must be finite, excluding booleans. Signed head weights
  are accepted. ReLU is per head, before weighting and summation; final scores
  may be negative. No softmax or extra dimension scaling is inserted.
- Cluster-score ties preserve supplied cluster order. Key-score ties and
  last-cluster truncation use ascending token ID. These deterministic choices
  are ours, not claims about the authors' implementation.
- This readable implementation validates/copies the visible prefix and sorts
  candidates. It makes no runtime, memory, asymptotic speedup or model-quality
  claim. Ordinary floating point can lose precision or underflow; nonfinite
  intermediate scores are rejected. Ties near floating-point precision can
  change selection.

The 27 tests (22 model and 5 demo/CLI tests) use an independent exact-rational
score/ranking oracle, full-budget equivalence, signed weights, headwise rectification, causal and stale-snapshot
controls, reservation, truncation, immutability, invalid inputs and CLI output.

## Citation and license

Park, Jeongmin Brian, and Wen-mei Hwu. *AttentionIndex: Accelerating Learned
Sparse Attention with a Causal Inverted-File Index*. arXiv:2611.02508v1 (2026).
[DOI](https://doi.org/10.48550/arXiv.2611.02508).

The paper is CC BY 4.0 according to its arXiv license link. This independently
written code and explanation use the repository's MIT license. No upstream code,
paper figures, weights, private data, or unpublished research are redistributed.
Citation credits the idea and does not imply endorsement.
