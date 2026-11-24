# tsdate: masking differs from assuming zero mutations

How much sequence was exposed to the molecular clock? We give tsdate the same
synthetic two-tip topology and mutation rate in three ways. Twenty deterministic
singletons lie in the first megabase (10 on each tip edge); the second
megabase is designated as known uncallable. Both sampled tips are fixed at
age zero, and the initial root time is an uncalibrated placeholder. The
root time estimated by tsdate is the TMRCA for these two tips on the supplied
tree, not an estimated population split time.

| Treatment | Coordinate length | Exposed span per edge | Root posterior mean |
| --- | ---: | ---: | ---: |
| Keep edges across the uncallable half | 2 Mb | 2 Mb | 50 generations |
| Mask the uncallable half, retain coordinates | 2 Mb | 1 Mb | 100 generations |
| Trim the already masked flank | 1 Mb | 1 Mb | 100 generations |

An edge spanning an interval treats that interval as exposure in this model.
Keeping the edges while providing no variants in the uncallable megabase
makes the input look like twice as much callable sequence. Deleting the edges
there with `delete_intervals(..., simplify=False)` changes exposure without
changing the coordinate length, surviving variant positions, or genotypes.
Trimming the now-empty right flank is a control, not a required step.

Independent likelihood check: if each edge has span L and common age t, each
count is Poisson with mean μLt. The two observed counts of 10 give
t̂ = 20/[μ(L+L)]. Directly maximizing the Poisson likelihood on the exact
integer grid gives 50 or 100 here. The `variational_gamma` posterior means
happen to match those values in this fixture. A posterior mean is not an MLE
in general. Default rescaling is disabled deliberately to isolate the clock.

This is a single topology with fixed sample times, exact mutation assignments,
and constant specified μ = 10⁻⁷ per base per generation. We know which flank
is uncallable by construction; a variant-free interval alone is **not** evidence
that it should be masked. Removing truly callable invariant sequence would
also discard real clock information. A real analysis needs an externally
justified callability mask, and no real-world dating accuracy, substitution
model validation, or uncertainty calibration is claimed.

## Reproduce

Use CPython **3.12.11**, tsdate **0.2.7**, tskit **1.0.3** and NumPy
**2.5.4**, reusing the repository's hash-locked tsdate environment. No seed or
external data are needed.

```sh
python -m pip install --require-hashes -r examples/inference_showcase/tsdate_clock/requirements.lock
python -m unittest examples.inference_showcase.tsdate_gap.test_demo -v
python -m examples.inference_showcase.tsdate_gap.demo
python -m examples.inference_showcase.tsdate_gap.demo --check-sample
```

The replay checks values using relative and absolute tolerances of 10⁻⁷,
and checks non-floats exactly. Tests verify the exact mutation counts and
genotypes, separate nominal length from dating exposure, and compare against
the independently evaluated likelihood.

- [Tsdate methods and preprocessing API](https://tskit.dev/tsdate/docs/stable/python-api.html#preprocessing-tree-sequences)
- [Pope et al. (2026), variational dating](https://pmc.ncbi.nlm.nih.gov/articles/PMC12871112/)
