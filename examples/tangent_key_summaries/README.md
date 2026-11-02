# ZHA tangent key summaries: a lower mass estimate can still rank chunks wrongly

A small original implementation of the reference-query summary and merge
component in **Zyphra Hierarchical Attention (ZHA)**, from Lufkin et al.,
[Subquadratic Attention with a Hierarchical Tree-Structured Indexer](https://arxiv.org/abs/2610.42112v1),
preprint first posted **October 30, 2026**. The formulas are in
[section III-A and its shared-reference merge note](https://arxiv.org/html/2610.42112v1#S3.SS1).

The paper summarizes a chunk using attention-weighted keys and a bias for each
reference query, then takes the maximum of the resulting affine scores. Its
full system learns references and uses a hierarchy for retrieval. Here references
are supplied constants. We implement the mathematical summaries, their merge,
and flat chunk-probability diagnostics only. Tangent bounds and entropy identities
have classical mathematical origins; this example makes no invention claim.

## Run

With **CPython 3.14.5**, from the repository root:

```sh
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/tangent_key_summaries -v
python -m examples.tangent_key_summaries.demo --check-sample
```

Only the standard library is used. No model, dataset, package, API or GPU is
needed. All inputs are constructed; there is **no randomness or seed**.
[model.py](model.py) contains the component, [demo.py](demo.py) the experiments,
and [sample_output.json](sample_output.json) actual executed output rounded to
nine decimal places. CI checks the sample byte for byte.

## What is compressed

For a chunk of keys `k_j` of dimension `d`, define the log unnormalized mass
at query `q`:

```text
F(q) = log sum_j exp(q · k_j / sqrt(d)).
```

At a reference query `r`, calculate:

```text
w_j(r) = softmax_j(r · k_j / sqrt(d))
k_hat(r) = sum_j w_j(r) k_j
b_hat(r) = -sum_j w_j(r) log w_j(r)
t_r(q) = q · k_hat(r) / sqrt(d) + b_hat(r)
F_hat(q) = max_r t_r(q).
```

The entropy expression equals `F(r) - r · k_hat(r)/sqrt(d)` and avoids
subtracting two large nearly equal values. Every tangent satisfies `t_r(q) <=
F(q)` in exact arithmetic, touching at `q=r`. Adding references cannot decrease
the envelope. Floating-point assertions use explicit tolerances.

An independent identity makes the error interpretable:

```text
F(q) - t_r(q) = KL(w(r) || w(q)).
```

This follows by substituting each log-softmax into the KL sum. The bound is tight
when the reference and query induce the same within-chunk distribution. It does
not imply all future queries are well represented by a few supplied references.
A zero reference reduces to the mean key and `log(chunk size)` bias.

## Why child biases matter

For child summaries `(k_i,b_i)` and a shared reference `r`, merge with
`a_i = softmax_i(r · k_i/sqrt(d) + b_i)`. The parent key is `sum_i a_i k_i`,
and its bias is `entropy(a) + sum_i a_i b_i`.

When every descendant uses the same reference and there are no positional
rotations, merging recovers a direct summary of all underlying keys. The tests
check unequal child sizes and multiple levels. This equivalence is deliberately
limited: summaries made at different references generally do not yield that
flat summary. Omitting child biases is also wrong; for all-zero chunks of sizes
2 and 3, the correct merged bias is `log(5)`, whereas dropping child biases gives
`log(2)`.

## Executed examples

For keys `[-2,2]`, the summary at reference 1 has compressed key **1.928055160**
and entropy bias **0.090094768**. With references `[-1,0,1]`:

| Query | Exact log mass | Tangent envelope | Gap |
| ---: | ---: | ---: | ---: |
| 0 | 0.693147181 | 0.693147181 | 0 |
| 0.5 | 1.126928011 | 1.054122348 | 0.072805663 |
| 1 | 2.018149928 | 2.018149928 | 0 |
| 2 | 4.000335406 | 3.946205088 | 0.054130318 |

The last gap matches independently computed `KL(w(1)||w(2))`. For chunks `[0,1]`
and `[2,3,4]` with reference 0.7, both direct and merged summaries yield key
**3.169254433** and bias **1.237200184**, agreeing within floating-point tolerance.

The chunk-ranking counterexample uses query 1, chunk A `[-6,6]`, and chunk B
`[2,2]`. Every value in A is 1; every value in B is 0:

| References | Estimated probability of A | Exact probability of A | Illustrative top-one choice / output |
| --- | ---: | ---: | --- |
| `[0]` | 0.119202922 | 0.964663365 | B / 0 |
| `[0,1]` | 0.964663365 | 0.964663365 | A / 1 |

Dense attention returns **0.964663365**. Including the query as a reference
repairs the mass estimate by construction; it is not evidence that learned
references will do so on unseen queries. Selecting only A still returns 1,
not the dense result, because it discards B's positive mass.

Although both chunk log scores are lower bounds, the normalized probability
of B with reference zero is **0.880797078**, far above its true **0.035336635**.
Normalization removes any componentwise lower-bound guarantee. The top-one
choice is an illustrative diagnostic, **not ZHA's top-p tree traversal**.

## Verification and limits

The **32 tests** include an independently written 80-digit Decimal oracle,
finite-difference gradients, the KL-gap identity, tangency and lower bounds,
key translations and rotations, duplication/count effects, key/reference
permutations, shared-reference recursive merging, the missing-bias control,
chunk-ranking failure, stable extreme logits, validation and exact CLI output.

A visible-prefix diagnostic removes future keys **before** building summaries.
Appending keys -10000 and 10000 therefore leaves the visible result unchanged.
Partial visible chunks are recomputed. This checks masking in our component;
it does not implement the paper's incomplete-chunk handling or a causal decoder.

There is no learned reference set, full tree indexer, top-p traversal, sink-token
policy, RoPE, GQA, distillation, trained language model or optimized kernel.
Flat diagnostics scan all chunks and compute dense reference masses; no runtime,
cache-traffic, subquadratic-scaling or real-model performance claim follows.
Finite references can underestimate relevant chunks badly, especially away from
their reference distributions. The component does not compress values or evict
KV entries.

The tracked code, constructed inputs and explanatory text are original under
this repository's [MIT license](../../LICENSE). No upstream code, paper figures
or text passages are copied. The cited paper uses the arXiv perpetual
non-exclusive distribution license; that license is not an MIT license for the
paper. The implementation here expresses the mathematical component independently.
