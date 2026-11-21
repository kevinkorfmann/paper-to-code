# XM-Sparse: when can resident tiles hide transfers?

[Sun et al., *XM-Sparse*, 17 November 2026 preprint](https://arxiv.org/abs/2611.22280)
uses resident-first traversal and a bounded tile pipeline to overlap sparse
attention with retrieval of selected KV tiles missing from fast GPU memory.
This original discrete-event example isolates that overlap from GPU execution.

There is one serial transfer engine and one serial attention engine, which can
run concurrently. Every attention tile takes 3 abstract time units and every
remote tile takes 4 units to fetch. A resident tile is immediately ready; a
remote staging slot is occupied from fetch start through the end of attention on
that tile. The baseline fetches all remote tiles before starting any attention
and assumes enough separate staging to hold those fetched tiles. The selected
tile set is the same in the first four rows. The last row is an all-resident
control, not another scheduling policy.

| Schedule | Finish time (abstract units) |
| --- | ---: |
| Fetch everything before attention | 27 |
| Resident first, one slot | 23 |
| Resident first, two slots | 16 |
| Remote first, two slots | 21 |
| All resident | 15 |

With two slots and resident-first order, attention works on two resident tiles
from time 0 to 6 while the first two transfers span times 0–4 and 4–8. The third
fetch cannot start at 8: it must wait until the first remote tile's attention
ends at 9 and releases its slot. It arrives at 13, one unit after attention
finishes reading the preceding tile. This leaves a one-unit gap relative to the
15-unit compute-only control. `sample_output.json` shows every event.

## Reproduce

From the repository root, with CPython 3.14.5 and only the standard library:

```sh
python -m unittest examples.xm_sparse_pipeline.test_demo
python -m examples.xm_sparse_pipeline.demo
python -m examples.xm_sparse_pipeline.demo | diff -u examples/xm_sparse_pipeline/sample_output.json -
```

The five tests include 1,701 deterministic comparisons with a time-stepped
reference. No random seed or external data is needed.

The example does not compute attention values, choose top-k indices, simulate
hardware, model memory contention, or implement the paper's dequantization,
refill, reduction across splits, or separate K-index prefetch. All times are
outputs of the stated model, not GPU measurements or predictions of speedups.
