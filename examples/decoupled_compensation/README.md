# Decoupled compensation: exact correction can break cancellation

An original, deterministic scalar mechanism demonstration. No training or downloaded data.

## Source and implemented scope

Li et al., [DecoupAttn: Decoupled Sparse Attention for Training-Free Acceleration
of Video Diffusion](https://arxiv.org/abs/2611.05917v1), preprint submitted
November 4, 2026. This implements the **CENTROID** option in Section 4.1,
Equations (4)–(6), with scalar keys and values (head dimension one).

For supplied clusters, compute mean key and mean value using **all** members.
An exact-token set represents the union of selected blocks. Each unselected
member contributes its cluster-centroid weight and mean value. Selected members
contribute their exact exponential weight and value instead. Both groups use
one denominator. Cluster multiplicities exclude selected members, preventing
double counting. This equals predicting every member then replacing selected
predictions with exact contributions. The direct skipped-count implementation
avoids subtracting large nearly equal totals.

This is not the FULL estimator: variance and pooled key–value covariance
corrections, clustering, reassignment, calibration, budget allocation, diffusion,
GPU kernels and real-model evaluations are absent. Supplied clusters and masks
are fixed. The source's empirical residual cancellation is not a universal theorem
that more corrected blocks must monotonically improve output.

## Original constructed controls

With query zero, keys `[0,0]`, values `[0,2]` and one cluster, dense output is 1.
No exact tokens gives 1: the two prediction residuals cancel. Correct token zero
only and output becomes 0.5, because the remaining prediction is still the
all-member mean 1. Correct both and output returns to 1. Recomputing a skipped-only
mean would change the estimator. This counterexample is not a claim that the
paper's calibrated masks fail in practice.

With keys and values `[-1,1]` and query 1, the corresponding outputs are
`0`, `-0.268941421`, and `0.761594156`. A missing covariance cannot generally be
recovered by a centroid alone.

A second control has two value-zero members in one cluster and one value-nine
member in another. Correct multiplicity gives 3; averaging cluster outputs
without their sizes incorrectly gives 4.5.

## Reproduce

From repository root, use CPython **3.14.5** as pinned by `.python-version`.
Standard library only. No packages, model weights, random seeds or network calls.

```sh
python -m unittest discover -s examples/decoupled_compensation -t . -v
python -m examples.decoupled_compensation.demo
python -m examples.decoupled_compensation.demo --check-sample
```

`sample_output.json` is actual captured output. The 21 tests include an independent
80-digit Decimal per-token oracle across 192 query/partition/subset cases,
exact-rational zero-query controls, the residual-error identity, dense and
singleton limits, permutation and affine invariances, cluster counts, overflow,
invalid-input checks and byte-for-byte CLI replay. All arithmetic is float64
in the implementation; finite inputs that overflow score or aggregation are
rejected. The common logit shift protects exponentials, but this is not a
formal arbitrary-magnitude numerical guarantee.

No speedup, memory saving or model-quality gain is measured. The Python routine
explicitly scans inputs and is intended for clarity, not acceleration. All code,
fixtures and prose are original under the repository's MIT license; no source
code, figures or weights are copied from the paper.
