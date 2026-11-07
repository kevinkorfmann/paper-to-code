# Nucleotide diversity needs callable invariant bases

This original educational example uses **scikit-allel 1.3.13** to estimate
nucleotide diversity (`pi`) from three constructed allele-count records.
Identical variant records give pi **0.016666667** with 100 callable bases,
or **0.083333333** with 20. The variant file alone does not identify the
number of bases over which diversity should be estimated.

## Estimand, inputs and assumptions

Pi is the mean proportion of differing bases between distinct sampled
chromosomes, averaged over the declared callable region. At each recorded site,
we sum the fraction of discordant chromosome pairs, then divide by callable
base count. Unrecorded callable bases must genuinely be invariant, not missing
variants. No demographic history, selection coefficient, mutation rate or
confidence interval is inferred.

The fixture is four complete chromosomes with allele counts `[3,1]`, `[2,2]`,
`[1,3]` at 1-based positions 10, 15 and 20 of a 100-base contig. All other
callable positions are declared invariant. Their pairwise-difference contributions
are 1/2, 2/3, 1/2, totaling 5/3. No randomness or seed is used.

The full and 20-base masks represent **alternative callability assumptions**,
not two measurements of a known population truth. If only 20 bases were truly
callable, declaring all 100 treats 80 unknown positions as invariant and dilutes
pi fivefold. If all 100 were callable, restricting to 20 changes the target
region; it is not automatically a correction. A real mask must come from quality
and callability evidence independently of the observed variant list.

## Checked output

| Input interpretation | Callable bases | Retained records | Pi |
| --- | ---: | ---: | ---: |
| Whole contig callable | 100 | 3 | 1/60 |
| Bases 1–20 callable | 20 | 3 | 1/12 |
| Incorrect variant-only mask | 3 | 3 | 5/9 |
| Bases 1–20 except position 15 | 19 | 2 | 1/19 |
| No callable bases | 0 | 0 | undefined (`null`) |

The variant-only value is diversity conditional on these three variable sites,
not per-base diversity of the contig. Dropping position 15 correctly removes
both its 2/3 contribution and its base from the denominator. The library warns
when inaccessible variant records are masked; this expected warning is retained
on stderr. `sequence_diversity` without explicit boundaries instead uses the
11-base span from first to last record and returns 5/33. Always specify the
intended region and its callability.

The wrapper requires sorted unique positions, nonnegative integral counts,
a boolean base mask, at least two equally sampled chromosomes at each record,
and at least one record. Empty record arrays are explicitly rejected because
the pinned tool's mask path rejects them; a monomorphic record is supported.
Multiallelic counts are supported. Partial or genotype-dependent missingness,
varying chromosome counts, erroneous variants and uncertainty in the mask are
not modeled. Phase is not needed for this per-site statistic. The tiny fixture
is a bookkeeping demonstration, not a biological benchmark.

## Reproduce

Run from the repository root. Reuse the existing isolated, hash-locked
scikit-allel environment (CPython 3.12.11, NumPy 2.3.5):

```sh
python3.12 -m venv .venv-callability
.venv-callability/bin/python -m pip install --require-hashes -r examples/inference_showcase/scikit_allel_fst/requirements.lock
.venv-callability/bin/python -m examples.inference_showcase.diversity_callability.demo
.venv-callability/bin/python -m examples.inference_showcase.diversity_callability.demo --check-sample
.venv-callability/bin/python -m unittest discover -s examples/inference_showcase/diversity_callability -v
```

The 20 tests cover exact rational pair enumeration, all **264 combinations** of
small biallelic count tables and three-base masks (including an invariant base),
allele reversal, monomorphic padding, multiallelic sites, coordinate boundaries,
invalid inputs, expected warnings, null output and exact CLI replay. The oracle
is deliberately limited to 2–32 chromosomes. The main estimator calls the real
tool; the independent oracle explicitly enumerates unordered pairs. JSON rounds
to 12 decimal places and never emits NaN.

## Sources and licensing

- [Official diversity API documentation](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.windowed_diversity).
- [Version-pinned tool implementation](https://github.com/cggh/scikit-allel/blob/v1.3.13/allel/stats/diversity.py).

The stable docs currently identify themselves as 1.3.3; the behavior above was
also checked against the installed 1.3.13 implementation. All example code,
inputs and tests are original and use the repository's MIT license. No source
code was copied from the dependency, no private data or credentials are used,
and no real-genome inference is claimed.
