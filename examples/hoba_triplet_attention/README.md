# HOBA triplet attention: marginals can hide dependence

An original, dependency-free demonstration of a **single-head mathematical
kernel**, using only constructed inputs. It is not a trained model or a
reproduction of the full HOBA architecture or its reported results.

## Source and date

Mafizur Rahman and Lijun Qian, **[Modeling Triplet Interactions Beyond Pairwise
Attention](https://aclanthology.org/2026.findings-aacl.101/)**, Findings of the
Association for Computational Linguistics: IJCNLP-AACL **2026**, pp. 2010–2026.
The proceedings are dated **November 2026**; the [primary PDF](https://aclanthology.org/2026.findings-aacl.101.pdf)
lists November 6–10. This is a proceedings date, not a verified first-preprint date.

We implement the supplied-projection equations in §3.2, Eqs. 2–5 (printed
pp. 2012–2013), and a nonoverlapping version of the local block restriction in
§3.2.1. Full HOBA additionally has learned projections, overlapping blocks,
overlap aggregation and a cross-block interaction module. Those are omitted.

## The kernel

For a query `q`, two key branches `a[j]`, `b[k]`, and two value branches
`u[j]`, `v[k]`, use key dimension `d`:

```text
score[j,k] = sum_r q[r] a[j,r] b[k,r] / sqrt(d)
weight[j,k] = exp(score[j,k]) / sum_j',k' exp(score[j',k'])
output[r] = sum_j,k weight[j,k] u[j,r] v[k,r]
```

The softmax is **joint over all ordered pairs**. It includes diagonal pairs
`j=k`; there is no ordering or causal mask. It does not independently normalize
each row. The implementation subtracts the maximum score before exponentiation.
Value coordinates may have a different width from key coordinates, while the
two value branches must match each other.

## An original two-token experiment

Both key branches and both value branches are `[-1, 1]`. For scalar query `q`,
the score matrix is `[[q,-q],[-q,q]]` and the pair-value matrix is
`[[1,-1],[-1,1]]`. Hence the output is exactly `tanh(q)` in real arithmetic.

| Query | Joint weights, first row | Joint weights, second row | Output |
| ---: | --- | --- | ---: |
| +1 | [0.440398539, 0.059601461] | [0.059601461, 0.440398539] | +0.761594156 |
| -1 | [0.059601461, 0.440398539] | [0.440398539, 0.059601461] | -0.761594156 |
| 0 | [0.25, 0.25] | [0.25, 0.25] | 0 |

**Every row marginal and column marginal is [0.5, 0.5] in all three cases.**
The query changes the dependence between the selected pair indices without
changing either one-axis distribution. Replacing the joint weights by the
product of those marginals makes the expected pair value zero in every case.
This replacement is an explicitly constructed **independence approximation**,
not an implementation of ordinary softmax attention. It is not a proof that
stacked pairwise-attention networks cannot learn the same input/output mapping.

The marginal distributions do not retain enough information to reconstruct
this joint readout. Diagonal-versus-off-diagonal mass is the missing information.

## A boundary counterexample

With three zero queries, identical keys, and values `[1,1,10]` in both branches:

- Disjoint blocks of size two give outputs `[1,1,100]`. The short final block
  has one pair and must not be dropped.
- One full-size block gives `[16,16,16]`, the square of the global value mean.

Local blocks therefore do not reproduce dense triplet attention. A change
outside a query's block cannot affect its output, but a later token **inside**
the same block can. This kernel is bidirectional, not autoregressive.

## Reproduce

Use **CPython 3.14.5**, matching the repository's `.python-version` and CI.
There are no third-party dependencies, weights, datasets, network calls,
GPU requirements, or paid services for this example.

```sh
git clone https://github.com/kevinkorfmann/paper-to-code.git
cd paper-to-code
python3.14 -m venv --without-pip .venv-hoba
. .venv-hoba/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'; assert platform.python_implementation() == 'CPython'"
python -m unittest discover -s examples/hoba_triplet_attention -v
python -m examples.hoba_triplet_attention.demo
python -m examples.hoba_triplet_attention.demo --check-sample
```

On Windows, activate `.venv-hoba\Scripts\Activate.ps1` in PowerShell.
[sample_output.json](sample_output.json) is the actual output. Floats are
rounded to 12 decimal places for display; tests use unrounded values and
explicit tolerances. The sample includes the interpreter version and is checked
byte for byte. Demo inputs are fixed. Randomized tests use seed `20261106`.

## Verification

The 29 tests cover an independently written 80-digit Decimal oracle, 20 seeded
input cases, the `sqrt(d)` scaling, joint normalization, diagonal inclusion,
the exact `tanh` identity, uniform marginals, zero-query/product-of-means
reduction, branch-swap symmetry, independent token permutations, reciprocal
key rescaling, value bilinearity, token duplication, constant and zero values,
extreme scores, rejected overflow and invalid shapes, input immutability,
full and singleton blocks, a short tail, block isolation, a noncausality
counterexample, actual CLI execution and exact checked output.

The existing repository CI also runs these tests and checks the sample.
The oracle uses direct high-precision exponentiation and flattened pair
enumeration rather than calling the implementation's softmax or reducers.

## Limits and provenance

- No training, language modeling, downstream benchmark, optimized kernel or
  speed/memory measurement is performed. This example establishes none of the
  paper's empirical gains.
- The block wrapper deliberately omits overlapping blocks and cross-block
  exchange. Block boundaries change the result, as the counterexample shows.
- The code explicitly materializes pair weights. For sequence length `L` and
  block size `b`, work is `O(L b² (d + d_v))`; returning every query's pair map
  uses `O(L b²)` probability storage. A full block has cubic sequence-length
  work. This is an inspectable CPU reference, not an efficient attention layer.
- Inputs and intermediate arithmetic must be finite. Very large products are
  rejected. Stable softmax does not make all upstream floating-point arithmetic
  safe, and very small probabilities can underflow to zero.
- The implementation, tests, prose and synthetic fixtures are original and
  covered by this repository's [MIT license](../../LICENSE). No upstream code,
  paper figures, pretrained parameters, private projects or real data are
  redistributed. The cited ACL paper remains under its own CC BY 4.0 license.

Try changing the sign or magnitude of `q` in `demo.describe`: which quantities
change even though both one-axis marginals stay exactly uniform?
