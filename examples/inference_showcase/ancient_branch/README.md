# Ancient samples: branch divergence is not twice TMRCA

An original deterministic **tskit 1.0.3** example of interpreting a summary downstream of genealogy inference. It constructs dated trees; it does not infer trees, date ancient samples, or estimate a demographic model.

For two connected samples with ages `a` and `b` measured backwards from the same origin, their separating path has length

```text
D_branch = (T_MRCA - a) + (T_MRCA - b)
T_MRCA = (D_branch + a + b) / 2
```

Thus simply dividing branch divergence by two is valid for time-zero samples, not generally for serial samples. With disjoint groups, average over all cross-group pairs and genomic span. The correction uses the sum of the two groups' mean sample ages. All node times must have a common calibrated unit and origin.

## Constructed input and actual output

Three samples have ages `[0,0,6]` generations. All coalesce at generation 10 over `[0,75)` and generation 20 over `[75,100)`. These are arbitrary star trees, not a simulation or empirical inference. Every distinct pair has span-weighted mean TMRCA `0.75*10 + 0.25*20 = 12.5`.

| Comparison | tskit branch divergence | Naive half | Corrected mean TMRCA |
|---|---:|---:|---:|
| Modern sample 0 versus modern 1 | 25 | 12.5 | 12.5 |
| Modern sample 0 versus ancient 2 | 19 | 9.5 | 12.5 |
| Modern 0 versus mixed group [1,2] | 22 | 11 | 12.5 |

The mixed-group per-tree values are `[17,37]`; their unnormalized span integral is 2200 generation-coordinate units. The normalized result is 22 generations, **not** the equally weighted tree mean of 27. With no mutations, site divergence is zero despite positive branch divergence.

## Reproduce

Use the existing hash-locked tsinfer environment. Only tskit is called here, not tsinfer.

```sh
python3.12 -m venv .venv-ancient
.venv-ancient/bin/python -m pip install --require-hashes -r examples/inference_showcase/tsinfer_genealogy/requirements.lock
.venv-ancient/bin/python -m unittest examples.inference_showcase.ancient_branch.test_demo -v
.venv-ancient/bin/python -m examples.inference_showcase.ancient_branch.demo --check-sample
.venv-ancient/bin/python -m examples.inference_showcase.ancient_branch.demo
```

Reference environment: CPython 3.12.11, tskit 1.0.3, NumPy 2.5.4. No random seed or network download is used by the demo. The 13 tests include an independent edge-table parent-chain oracle with exact rational accumulation, 64 grid comparisons, time and coordinate scaling, shifted time origin, mutation independence, group permutations, invalid groups and disconnected-pair rejection.

## Interpretation limits

The correction recovers a summary of the **provided** dated genealogy, not the true history when that genealogy is wrong. Sampling-age uncertainty and branch-time error are not modeled. The wrapper requires nonempty, disjoint sets of sample nodes and a common ancestor for every cross-pair in every interval. It deliberately rejects disconnected genealogies rather than interpreting finite root-to-sample lengths as a finite TMRCA. The exhaustive pair-connectedness check is for tiny fixtures, not scalable production use. Overlapping sample sets, missing trees, confidence intervals, and real-data performance are outside this tutorial.

## Primary sources

- [Official tskit divergence documentation](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.divergence): branch divergence measures the mean separating path length. The age correction above follows from its two branches.
- [Official statistics documentation](https://tskit.dev/tskit/docs/stable/stats.html): branch versus site modes and genomic-span normalization.
- [On ARGs, pedigrees, and genetic relatedness matrices](https://pmc.ncbi.nlm.nih.gov/articles/PMC11908205/): serial-sampling path-distance formula.

Fixtures, tests and explanatory example are original; no source implementation or empirical data was copied.
