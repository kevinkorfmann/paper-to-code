# slimulations: many breakpoints do not mean all gametes are recombinant

An original **SLiM 4.0.1** one-generation neutral-marker inheritance experiment, with a **CPython 3.14.5** standard-library wrapper. Expected breakpoint count, probability of any breakpoint, and two-marker recombination fraction are different quantities.

## Exact prediction

SLiM's per-adjacent-base-gap rate `r` is a crossover probability. With `d` gaps, no gene conversion or interference, the count `K` is Binomial(d,r):

```text
E[K]     = d*r
P(K > 0) = 1 - (1-r)^d
P(K odd) = (1 - (1-2*r)^d)/2
```

Only an odd number of ancestry switches changes endpoint phase. The last quantity is therefore the recombination fraction between the markers. This is the exact finite-lattice calculation, not the Poisson/Haldane approximation. As `r` becomes small with `d*r = m`, it approaches `(1-exp(-2*m))/2`.

## Actual simulation

20,000 diploid Wright–Fisher individuals all start `AB/ab`, with shared neutral mutations at positions 0 and `d`. One generation of uniform parental sampling produces 40,000 gametes. A passive `recombination()` callback counts the unmodified breakpoints; endpoint marker dosage independently identifies recombinant `Ab`/`aB` gametes. **Odd-breakpoint and recombinant-marker counts match exactly in every treatment.** These are gametic ancestry switches, not tetrad-wide crossover counts.

No mutation, selection, gene conversion, migration or spatial structure is modeled. Hermaphroditic parental sampling permits incidental selfing. Parent diplotypes are identical, so parental identity cannot change gamete probabilities. Tick 1 creates the founders and installs markers; tick 2 supplies the only measured reproductive generation. Each treatment has its own recorded seed, 202611190 through 202611194.

For 100 gaps at `r=0.1`, expected breakpoints per gamete are **10**, while expected recombinant fraction is approximately **0.5**. The actual run produced 399,218 breakpoints across 40,000 gametes (mean **9.98045**). Every gamete had at least one breakpoint, but only **20,264 / 40,000 = 0.5066** had recombinant endpoint markers. An even number of switches restores parental endpoint phase.

Controls include `r=0`, a single gap (all three quantities coincide), and sparse/intermediate rates across 100 gaps. Full integer counts, theory and measured proportions are in `sample_output.json`. Each measured mean/proportion is checked against its binomial Monte Carlo standard error using a seven-standard-error tolerance. This tolerance is a regression guard, not an inference confidence interval or a claim of zero sampling noise.

## Reproduce

Use the repository's checksum-pinned SLiM 4.0.1 build described in the main workflow, put `slim` on PATH, then run from the repository root:

```sh
python -m unittest examples.slimulations.crossover_parity.test_demo -v
python -m examples.slimulations.crossover_parity.demo --check-sample
python -m examples.slimulations.crossover_parity.demo
```

No Python packages beyond the standard library are needed. `--slim /path/to/slim` selects another binary, but the wrapper checks version 4.0.1. The 13 tests cover an independent exhaustive enumeration of Bernoulli gap patterns, zero/one-gap controls, saturation, parser validation, independent seeds and exact seeded replay.

## Limits and sources

The deliberately high rates are an abstract teaching map, not realistic per-base rates or a fit to any organism. This is a single-generation meiosis demonstration, not population LD decay, a map estimator, a recombination-interference model, or a prediction for natural chromosomes.

- [Pinned official SLiM engine source](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/chromosome.cpp#L432): conversion from user rate to breakpoint sampling and duplicate handling.
- [Official SLiM manual](https://www.benhaller.com/slim/SLiM_Manual.pdf), `initializeRecombinationRate()` and `recombination()` documentation.

All model code, tests, derivations and fixtures are original synthetic tutorial work. No organism-specific or private data is used.
