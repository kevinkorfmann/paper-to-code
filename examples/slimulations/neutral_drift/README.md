# slimulations 001: drift removes variation faster in small populations

Run the same neutral marker in independent populations of **20** and **100**
diploid individuals. All start at allele frequency 0.5. With no mutation,
migration, or selection, random inheritance alone changes allele frequencies
and eventually fixes or loses the marker.

This is an actual **SLiM/Eidos simulation**, not a Python replacement for SLiM.
The short [Eidos script](neutral_drift.slim) runs the population dynamics.
The [Python wrapper](demo.py) validates output and compares replicate averages
with Wright–Fisher theory.

## Reference environment and sources

- **SLiM 4.0.1**, with bundled Eidos 3.0.1; this is a deliberate pin to a
  verified installed release, not a claim that it is the newest SLiM.
- **CPython 3.14.5** for the standard-library wrapper and tests.
- [Official SLiM 4.0.1 release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1);
  source commit `09fcf604a06307cf75098dfa0a2171a9b7852216`.
- [Version-matched SLiM manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf):
  sections 6.3.1 and 24.1 discuss incidental selfing and independent parent
  choices; the `initializeSLiMOptions` reference documents
  `preventIncidentalSelfing`.
- [SLiM authors' project and citation page](https://messerlab.org/slim/).

The script is original. No upstream recipes or simulator source files are
redistributed. The repository's MIT license covers this example; SLiM itself
has its own GPL license.

## Run

From the repository root, with SLiM 4.0.1 on PATH and the root Python environment:

```sh
slim -v
python -m examples.slimulations.neutral_drift.demo
python -m unittest discover -s examples/slimulations/neutral_drift -v
```

The wrapper can use a specific executable:

```sh
python -m examples.slimulations.neutral_drift.demo --slim /path/to/slim
```

Or run the Eidos script directly:

```sh
slim -s 20261020 -d N=20 -d REPLICATES=256 -d GENERATIONS=80 \
  examples/slimulations/neutral_drift/neutral_drift.slim
```

Lines beginning `DATA,` contain
`generations_elapsed,replicate_id,diploid_N,marker_copy_count`. Other output
is SLiM's normal startup text. `INITIAL_COPIES` defaults to N (frequency 0.5);
it can be set to any integer from 0 through 2N for boundary experiments.

If SLiM 4.0.1 is unavailable, the
[CI build recipe](../../../.github/workflows/tests.yml) downloads its pinned
official source, verifies SHA-256, and builds only the CLI with CMake and a C++
compiler in a temporary directory. It does not install or replace system tools.
The build passes `-include cstdint` to supply the standard integer-type header
that this older source expects transitively; current GCC requires it explicitly.
This compatibility flag does not alter the simulation algorithm.
The wrapper intentionally rejects other SLiM versions instead of silently
treating their random output or APIs as interchangeable.

## Model details

There are 256 replicate populations at each N. The seeds are **20261020**
(N=20) and **20261021** (N=100). Replicates are isolated subpopulations with
no migration; all share the simulator's seeded pseudorandom stream.

The genome consists of one neutral locus. Mutation and recombination rates are
zero. `m1.convertToSubstitution = F` keeps a fixed marker available for counting.
All individuals are diploid hermaphrodites with equal fitness.

The two parents are chosen independently: **incidental selfing is permitted**,
as in the gene-copy Wright–Fisher model. The script does not enforce
outcrossing or add a prescribed selfing fraction.

SLiM's tick 1 includes a reproduction stage. We introduce the marker in
`1 late()`, after that stage, and only then record **t=0**. Thus every replicate
starts with exactly N marker copies out of 2N. Tick 81 records t=80, after
exactly 80 subsequent reproduction steps.

## Theory and what is measured

Let p be the marker's frequency in all 2N copies, and define gene diversity
`H = 2p(1-p)`. This is the probability that two independent draws with replacement
from those copies have different alleles. It is **not** a count of
heterozygous individuals, and no sample-size correction is applied.

Under the specified reproduction model:

```text
X_(t+1) | p_t ~ Binomial(2N, p_t)
E[p_t] = p_0
E[H_t] = H_0 (1 - 1/(2N))^t.
```

The last identity follows by inserting the binomial mean and variance into
`E[2p_(t+1)(1-p_(t+1)) | p_t]`. The Python expectation is independent of
SLiM's realized random draws.

These are ensemble expectations. A single population's H can increase between
generations; the average allele frequency in a finite batch need not remain
exactly 0.5. Fixation and loss are absorbing because mutation is disabled.

## Checked result

The committed [sample_summary.json](sample_summary.json) is the actual run:

| Diploid N | Mean H at t=80 | Theoretical E[H₈₀] | Monte Carlo SE of mean H | Fixed / lost / segregating |
| ---: | ---: | ---: | ---: | --- |
| 20 | 0.074487305 | 0.065968903 | 0.009640767 | 95 / 106 / 55 |
| 100 | 0.340025000 | 0.334823910 | 0.009957965 | 7 / 8 / 241 |

Both started at H=0.5. The observed difference is consistent with much faster
neutral loss of diversity in the smaller population. This one batch is not an
estimate of diversity loss in a particular species.

The 11 tests run the real pinned engine and check initial timing, absorbing
boundaries, zero duration, seed repeatability, direct Eidos validation, CLI
behavior, and the committed summary. They compare both H and p against
distribution-free Hoeffding tolerances. H lies in [0,0.5] and p in [0,1].
A union bound over all 20 mean comparisons has nominal failure probability
at most 10^-6 under independent random sampling. This is a conservative
diagnostic, not a test of the simulator's pseudorandom generator. The reported
standard errors are ordinary between-replicate sample SEs, not those bounds.

Fixed seeds plus a pinned engine make the checked run repeatable. Exact random
streams are not promised for arbitrary simulator releases or compiler/platform
changes; CI also verifies this reference output on Ubuntu.

## Limits

This deliberately excludes selection, new mutation, demography, spatial
structure, migration, linkage, and separate sexes. There is no inference from
real data. The census N equals the ideal-model diploid size here; it must not
automatically be identified with effective population size in other models.
