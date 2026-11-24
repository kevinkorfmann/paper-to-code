# Equal pairwise sharing does not imply equal three-way ancestry

Four neutral founders produce a cohort of 32 clones. In each treatment, the model
randomly permutes that treatment's four reproductive weights across the
founders independently for every cohort. Each offspring independently draws one parent with probability
proportional to its assigned weight. Both homologues carry the parent's
unique neutral marker. Actual clones are made by SLiM's nonWF
`addCloned` reproduction callback; cohort labels are only used to group
the output. There is no mutation, recombination or heritable fitness
difference among marker states. This is one reproductive step, not a
long-run population-size or coalescent-limit calculation.

| Lottery weights, normalized by their sum (16) | Pair shares one founder | Triple shares one founder |
| --- | ---: | ---: |
| 1, 1, 7, 7 (exact) | 0.390625 | 0.16796875 |
| 1, 3, 3, 9 (exact) | 0.390625 | 0.19140625 |
| 1, 1, 7, 7 (SLiM, 512 cohorts) | 0.387970955 | 0.165033156 |
| 1, 3, 3, 9 (SLiM, 512 cohorts) | 0.389494865 | 0.189658282 |

For parental probabilities pᵢ, two distinct offspring chose the same
parent with probability ∑pᵢ². Three distinct offspring all chose the same
parent with probability ∑pᵢ³. The two lotteries match exactly on the first
quantity but differ on the second by 6/256. For each realized cohort with
founder counts nᵢ and n = 32, the corresponding fractions are
∑ choose(nᵢ,k) / choose(n,k), for k = 2 or 3. Picking the same
offspring twice would be incorrect; the tests enumerate 2,048 examples of
distinct-position pairs/triples. We average the within-cohort measurements,
not a pool of offspring across cohorts, and use between-cohort standard
errors to quantify Monte Carlo variation.

The matched pairwise probability only matches this one pairwise statistic.
It does not assert equal full genealogical distributions or equal effective
population sizes inferred in any other way.

## Reproduce

From the repository root on Linux, use SLiM **4.0.1** on PATH and
**CPython 3.14.5** (standard library only). The demo streams the model
to SLiM through `/dev/stdin`.

```sh
python -m unittest examples.slimulations.pair_triple_ancestry.test_demo -v
python -m examples.slimulations.pair_triple_ancestry.demo
python -m examples.slimulations.pair_triple_ancestry.demo --check-sample
```

Seeds: 20261124 and 20261125. The saved sample includes Monte Carlo standard
errors, and the replay checks exact output. The test suite also checks the
exact equal-pair/different-triple targets and deliberately rejects incomplete
censuses and a mismatched sample.

- [Möhle & Sagitov (2001), exchangeable reproduction and coalescent limits](https://doi.org/10.1214/aop/1015345761)
- [Haller & Messer (2023), SLiM 4](https://doi.org/10.1086/723601)
