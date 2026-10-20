# Inference showcase 001: tsinfer genealogy from phased variants

This example uses the real **tsinfer** inference pipeline to reconstruct a
genetic genealogy from synthetic phased alleles. It then asks two different
questions: did the output preserve the input variants, and how much of the
simulated ancestry did it recover?

The checked result preserves **296/296 allele calls**, yet recovers only
**3 of 6 nontrivial rooted clades** in the covered region. Encoding all observed
genotypes does not establish that a reconstructed genealogy is complete or true.

## Tool and primary sources

- Kelleher et al. (2019), *Inferring whole-genome histories in large population
  datasets*, **Nature Genetics 51**, 1330–1338.
  [doi:10.1038/s41588-019-0483-y](https://doi.org/10.1038/s41588-019-0483-y);
  [authors' citation page](https://tskit.dev/tsinfer/docs/stable/CITATION.html).
- [tsinfer usage and interpretation](https://tskit.dev/tsinfer/docs/stable/usage.html):
  phased variants, ancestral states, noninference sites, unresolved ancestry,
  and uncalibrated times.
- [msprime documentation](https://tskit.dev/msprime/docs/stable/intro.html):
  the simulator used only to create synthetic inputs and withheld truth.
- [bio2zarr](https://github.com/sgkit-dev/bio2zarr): format conversion.

The glue code, tests, and explanations here are original. Inference itself is
performed by the installed tsinfer package, not by a reimplementation or a
descriptive-statistics proxy.

## Isolated reference environment

Use **CPython 3.12.11**, separately from the repository's root 3.14.5 environment.
From the repository root:

```sh
python3.12 -m venv .venv-inference
.venv-inference/bin/python -c "import platform; assert platform.python_version() == '3.12.11'"
.venv-inference/bin/python -m pip install --require-hashes \
  -r examples/inference_showcase/tsinfer_genealogy/requirements.lock
.venv-inference/bin/python -m examples.inference_showcase.tsinfer_genealogy.demo
.venv-inference/bin/python -m unittest discover \
  -s examples/inference_showcase/tsinfer_genealogy -v
```

On Windows, use `.venv-inference\Scripts\python.exe` for the environment's Python.
The complete dependency set is version-pinned and SHA-256 hashed in
[requirements.lock](requirements.lock). Direct dependencies are tsinfer **0.5.1**,
msprime **1.4.4**, tskit **1.0.3**, and bio2zarr **0.1.6**.
bio2zarr 0.1.6 is intentional: it supports Zarr 2, which tsinfer 0.5.1 requires;
bio2zarr 0.2+ requires Zarr 3. No system Python changes are needed.

Installation downloads public packages from PyPI. The actual demonstration
requires no network, real genetic data, GPU, model download, or paid service.
All tool licenses remain their respective authors' licenses.

To retain the generated inputs and output trees:

```sh
.venv-inference/bin/python -m examples.inference_showcase.tsinfer_genealogy.demo \
  --output-dir results/tsinfer-run
```

The output directory must be new. It contains `input_variants.csv`, `input.vcz`,
`source.trees`, `inferred.trees`, and `summary.json`. `results/` is git-ignored.
Running without this option uses a temporary directory and prints only JSON.

## Synthetic experiment

The simulator creates four diploid individuals (eight phased genomes) under a
constant-size coalescent model with N=10,000, a 100,000-base region, and no
recombination. Binary mutations occur at rate 5×10^-9 per base per generation
on a discrete genome. Ancestry seed: **104729**. Mutation seed: **104743**.
These values are illustrative; no seeds were searched for a favorable outcome.

The seeded result has one true local tree and 37 variant sites. The
[checked input CSV](sample_input.csv) contains every allele used, with synthetic
haplotype labels 0–7. Binary mutations on a discrete genome permit recurrent
mutation in general; the example does not silently assume an infinite-sites
mutation model.

The workflow is:

```text
msprime synthetic genealogy + mutations
       │
       ├── withheld source tree ───────────────────────┐
       │                                             │
       └── phased variant arrays + ancestral states   │
                       │                              │
                  tsinfer.infer                       │
                       │                              │
             inferred tree sequence ── comparison ───┘
```

The inference function accepts only the variant-store path and genomic length.
It does not receive true edges, mutation ages, node ages, N, or true branch
lengths. Tests verify that the VariantData site/individual time arrays remain
unknown. Ancestral alleles are supplied from the synthetic experiment; that
information would need to be estimated or externally supplied for real data.

Inference runs sequentially (`num_threads=0`) with tsinfer's default
`recombination_rate=None` matching behavior. That setting is **not** an estimate
of zero recombination and is not derived from the true simulation parameter.
No optimization hyperparameters or demography are fitted here.

## Observed result and interpretation

See [sample_summary.json](sample_summary.json) for the actual checked result
and runtime package versions.

| Check | Observed |
| --- | --- |
| Input | 8 phased genomes, 37 sites |
| Sites used for topology inference | 13 |
| Other sites placed by parsimony | 24 |
| Preserved allele calls | 296 / 296 |
| True local trees | 1 |
| Inferred intervals | 1 inferred genealogy plus 2 unknown flanks |
| Genomic span used for topology scoring | 97,020 / 100,000 bases |
| Nontrivial true / inferred / shared clades there | 6 / 3 / 3 |
| Clade precision / recall on scored span | 1.0 / 0.5 |
| Inferred internal time units | **uncalibrated** |

The scored interval is `[1089, 98109)`. The 2,980 bases in the flanks have no
inferred single-root genealogy and are explicitly excluded; reporting their
coverage prevents a perfect score on a tiny covered portion from being
mistaken for whole-region recovery.

For each interval with a single root in both trees, we compare unique sets of
descendant sample IDs, excluding singletons and the all-sample root clade.
Internal node numbering and branch lengths are ignored. Shared, inferred,
and true clade counts are weighted by interval length before computing
precision and recall. This is a simple rooted-clade metric defined by this
example, not a general benchmark score or a calibrated posterior probability.
The source and inferred sample order must refer to the same genomes.

The recovered clades are compatible with the truth in this one realization,
but three true internal clades are unresolved. The other 24 sites are retained
by parsimony; their preservation should not be counted as independent evidence
for the inferred topology. Perfect genotype reconstruction is part of
tsinfer's representation behavior, not validation of the inferred genealogy.

## Tests and limits

The 10 tests execute the pinned tools and check all allele calls and positions,
the committed CSV/summary, deterministic repeat inference, withheld truth times,
and the uncalibrated time label. They validate the scoring code against the
truth itself and an unresolved star tree, detect changed alleles, reject
mismatched sample counts and lengths, and protect an existing output directory.
The repository's Ubuntu CI repeats the complete inference.

This does **not** infer population size, migration, selection, calibrated dates,
or demographic history. Dating would require an additional method and its own
assumptions. Phasing errors, missing data, ancestral-state errors, realistic
mutation spectra, and recombination are not evaluated. There are no uncertainty
intervals, repeated simulation benchmark, or real-data accuracy claims.
The example explains how to run a tool and inspect a limitation, rather than
establishing its general performance.
