# Recurrent mutation can maintain a deleterious allele

This original one-locus SLiM experiment combines **purifying selection, reversible
mutation and finite-population sampling**. Starting with no A clones, recurrent
B-to-A mutation supplies A while its lower reproductive weight removes it.
Stronger selection lowers its frequency but does not keep it absent.

## Run

CPython **3.14.5**, standard library only; SLiM **4.0.1 / Eidos 3.0.1**.
The repository workflow builds the checksum-pinned official SLiM release.
With `slim` on PATH, from repository root:

```sh
python -m unittest discover -s examples/slimulations/mutation_selection -t . -v
python -m examples.slimulations.mutation_selection.demo
python -m examples.slimulations.mutation_selection.demo --check-sample
```

Use `--slim /path/to/slim` to select the executable. `run()` exposes population
size, initial count, generation count, fitness, two mutation probabilities,
replicates and seed. The finite oracle is capped at N=200 and 80 generations.
No network is needed after Python and SLiM are installed.

## Model and mathematical control

Each of 512 independent populations contains N=100 clonal individuals for 40
nonoverlapping generations. SLiM represents clones as homozygous diploids,
but both homologues switch state together. There are **N independent sampling
units, not 2N gametes**. This is a haploid-like two-type model, not diploid
recessivity, dominance or a nucleotide mutation mechanism.

A parents have relative reproductive weight w, B parents weight 1. Each child
chooses a parent by those weights, clones it, then gets exactly one opportunity
for state change: B-to-A with probability u, A-to-B with probability v. Mutation
is applied **after selection**. Marker identities merely encode A, not ancestry.
There is no sexual reproduction, recombination, migration or density change.
All replicates, including monomorphic ones, contribute to results.

For p=K/N, the selected parental A probability is

`p_sel = p*w / (p*w + 1-p)`.

The post-mutation offspring probability is

`q = p_sel*(1-v) + (1-p_sel)*u`.

Thus `K_next | K ~ Binomial(N,q)`. The oracle propagates the complete N+1-state
transition distribution, evaluated in double precision. Its mean is the finite
model prediction. Iterating q at the mean is an infinite-population reference,
not generally the finite mean, because selection is nonlinear.

With w=0.8, u=0.02 and v=0.01 the deterministic fixed point is about **0.095764**.
It is found by bisection and checked by substitution. The familiar u/s formula
is exact for this life cycle's interior fixed point when v=0 and u<s=1-w;
it is not the fixed point with backward mutation. A finite population does not
sit at a single fixed frequency. Positive mutation in both directions makes
neither monomorphic state absorbing. The 40-generation simulation is a transient
comparison, **not a claim that every treatment has reached stationarity**.

## Actual checked output

All conditions have u=0.02, v=0.01 unless stated otherwise. Values at generation
40 from [`sample_summary.json`](sample_summary.json):

| Condition | Initial A count | Mean p(A) | Monte Carlo SE | Exact finite mean |
|---|---:|---:|---:|---:|
| w=0.8 | 0 | 0.095176 | 0.002111 | 0.097960 |
| No forward mutation (u=0), w=0.8 | 0 | 0 | 0 | 0 |
| Neutral, w=1 | 0 | 0.479238 | 0.007771 | 0.469525 |
| Stronger selection, w=0.5 | 0 | 0.040781 | 0.001015 | 0.039858 |
| w=0.8, initially all A | 100 | 0.108027 | 0.002589 | 0.108058 |

Seeds are **202611100–202611104** in the order declared in `TREATMENTS`. At the
final observation, A was absent in 1 of 512 selected populations and 13 of 512
strong-selection populations. Such absence is temporary in principle because
forward mutation can reintroduce A. It does not imply that reintroduction was
observed after the final time point.

## Tests and interpretation

Nineteen focused tests include independent exact-rational enumeration of parent
identities and mutation outcomes, a two-generation rational propagation, neutral
closed-form means, boundary behavior, fixed-point substitution, deterministic
flip/no-mutation SLiM controls and seed replay. The demo checks every simulated
generation against a broad six-standard-error finite-oracle tolerance. That is
a regression check, not a calibrated scientific hypothesis test.

The probabilities are deliberately large for a short demonstration. They are
not estimates for any organism. The clonal, single-site model cannot describe
linked deleterious load, Muller's ratchet, diploid recessive disease burden or
real genomic mutation-selection balance. It does not reproduce the diffusion
analysis in the reference below.

## Sources

- Charlesworth & Jain (2014), [Purifying Selection, Drift, and Reversible Mutation
  with Arbitrarily High Mutation Rates](https://doi.org/10.1534/genetics.114.167973),
  *Genetics* 198:1587–1602. Primary conceptual context for the interaction of these
  forces; our discrete finite-state calculation and code are original.
- Haller & Messer (2023), [SLiM 4: Multispecies Eco-Evolutionary
  Modeling](https://doi.org/10.1086/723601). Simulator reference, not a claim that
  this pinned version is the newest release.

Code is original under the repository license. No private research or natural
population data are used.
