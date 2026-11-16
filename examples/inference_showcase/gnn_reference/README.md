# Genealogical nearest neighbours depend on the reference panel

This original, deterministic example calls **tskit 1.0.3** on constructed trees. It demonstrates a summary used downstream of genealogy inference, not a genealogy-inference run or an ancestry estimator validated on real data.

## Estimand and input

For a focal sample in each tree, GNN finds its nearest ancestor with another selected reference descendant. It counts the reference descendants in each group there, excluding the focal sample itself, and normalizes their counts. The tree-wise fractions are weighted by genomic span. The reference sets need not contain every sample. These are **conditional reference-neighbour fractions**, not unconditional population ancestry or admixture proportions.

The input is a fixed tree sequence with six sample chromosomes. Node 0 is focal; nodes 1–3 are candidate A references and nodes 4–5 are candidate B references. The group names are annotations supplied to the calculation, not inferred populations. Nodes 0–4 form a polytomy at arbitrary time 1; node 5 joins their ancestor at time 2. The tree spans coordinates [0,100). `sample_output.json` records every edge and node time. No mutations, random numbers, empirical genomes, or private data are used.

```text
             7 (time 2)
            / \
    6 (time 1)  5 (B)
    / | | | \
   0  1 2 3  4
 focal A A A  B
```

The polytomy deliberately makes several reference descendants equally close. It can represent an unresolved relationship, but here it is simply a constructed topology. Resolving it can change the statistic. This is not a random coalescent draw or a claim that arbitrary binary trees have mixed per-tree GNN fractions.

## Actual output

All three calls use **the same tree and focal chromosome**. Unselected leaves remain in the tree.

| Selected A references | Selected B references | GNN A | GNN B |
|---|---|---:|---:|
| 1 | 4,5 | 0.50 | 0.50 |
| 1,2,3 | 4,5 | 0.75 | 0.25 |
| 1,2,3 | 5 | 1.00 | 0.00 |

Adding two existing A chromosomes to the reference panel raises A's fraction from 1/2 to 3/4 without changing the genealogy. The more distant B sample (node 5) does not enter the nearest clade's count. Omitting the closer B sample leaves only A references in that clade, so the statistic is 100% A despite B being present elsewhere in the tree.

A separate two-tree control keeps this topology over [0,75) and pairs the focal sample with B node 4 over [75,100). The first two panels then return A fractions **0.375** and **0.5625**. These are span-weighted averages, not averages assigning equal weight to each tree. Time rescaling and adding a mutation leave the result unchanged when topology and reference sets stay fixed.

## Reproduce

Reuse the repository's existing hash-locked tsinfer environment, which contains the tskit version used here. tsinfer itself is **not called** by this example.

```sh
python3.12 -m venv .venv-gnn
.venv-gnn/bin/python -m pip install --require-hashes -r examples/inference_showcase/tsinfer_genealogy/requirements.lock
.venv-gnn/bin/python -m unittest discover -s examples/inference_showcase/gnn_reference -v
.venv-gnn/bin/python -m examples.inference_showcase.gnn_reference.demo --check-sample
.venv-gnn/bin/python -m examples.inference_showcase.gnn_reference.demo
```

Reference versions: CPython 3.12.11, tskit 1.0.3, NumPy 2.5.4. No seed applies. Twenty tests include an independent exact rational oracle using Python parent-chain intersections, all 180 nonempty two-group reference assignments on each of two topologies, span weighting, self-exclusion, reference permutations, time/coordinate rescaling, mutation independence, invalid inputs, disconnected focal rejection and checked output.

## Assumptions and limitations

This bounded wrapper accepts two nonempty disjoint sets of sample nodes, complete reference labels, and a focal sample connected to another reference in every tree. It explicitly rejects unsupported/disconnected intervals rather than assigning them an ancestry interpretation. Arbitrary internal-node references and missing-topology handling are outside this tutorial.

Interpretation depends on the supplied trees, reference selection and labels, including close relatives and sample balance. Inferred-tree errors and unresolved branches can alter the output. Real analyses need suitable sampling and uncertainty assessment. This example supplies **no confidence interval, inferred demographic history, selection claim, admixture fraction, or real-genome accuracy benchmark**. The lesson is to specify and stress-test the reference panel, not to claim that equal group sizes universally remove bias.

## Sources

- [Official GNN API documentation](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.genealogical_nearest_neighbours), for the statistic and span weighting.
- [Kelleher et al. (2019), *Inferring whole-genome histories in large population datasets*](https://doi.org/10.1038/s41588-019-0483-y), the primary paper introducing this genealogy-based analysis.
- [Pinned tskit 1.0.3 source](https://github.com/tskit-dev/tskit/blob/1.0.3/python/tskit/trees.py), for the exact library version.

Fixtures, derivations and tests are original tutorial work, not copied paper code or a reproduction of an empirical result.
