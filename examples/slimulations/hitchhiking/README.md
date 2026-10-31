# A neutral allele can hitchhike with a beneficial neighbor

This original two-locus SLiM example shows a neutral marker changing frequency
because of its association with a selected allele. Recombination and reversed
association provide controls; no selection provides a drift-only comparison.

The classical primary source is Maynard Smith & Haigh (1974),
[The hitch-hiking effect of a favourable gene](https://doi.org/10.1017/S0016672300014634)
([publisher's paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/918291A3B62BD50E1AE5C1F22165EF1B/S0016672300014634a.pdf/the-hitch-hiking-effect-of-a-favourable-gene.pdf)).
This demonstrates the mechanism with a deliberately different constructed
diploid model; it does not reproduce the full paper or estimate the genomic
impact of recurrent sweeps.

## Run

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**. Python needs only
its standard library. From the repository root:

```sh
slim -v
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/slimulations/hitchhiking -v
python -m examples.slimulations.hitchhiking.demo --check-sample
```

The runner rejects another SLiM version and checks both engine versions from
Eidos `version()`. Use `--slim /path/to/slim` for a particular executable.
The existing [CI build recipe](../../../.github/workflows/tests.yml) downloads
official source commit `09fcf604a06307cf75098dfa0a2171a9b7852216` (v4.0.1),
verifies archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`,
and builds the CLI without SLiMgui. This is a reproducibility pin, not a claim
that 4.0.1 is the latest SLiM release.

Four treatments use seeds **202610310–202610313**, in the order below.
Each has **256 independent replicate populations**, **N=400 diploid
individuals**, and **30 generations** after initialization. The checked
[sample_summary.json](sample_summary.json) is actual executed output;
`--check-sample` compares it byte for byte. Derived numbers are rounded to
nine decimal places; raw simulation outputs are integer haplotype counts.

A smaller exploratory run, which does not match the default sample:

```sh
python -m examples.slimulations.hitchhiking.demo --n 100 --replicates 32 --generations 10
```

N must be a multiple of 100 for the exact founder construction.

## Model and initialization

At the first locus, A has native SLiM fitnesses **aa=1, Aa=1.5, AA=2**
(`selectionCoeff=1`, `dominanceCoeff=0.5`). The second locus's B mutation
has selection coefficient **zero**, regardless of its frequency. For the
no-selection control, all genotypes have fitness 1. The script audits actual
cached parental fitness after installing the founders.

Haplotype order is `ab, aB, Ab, AB`. The usual founders have frequencies
`[0.5,0.4,0,0.1]`: p(A)=0.1, p(B)=0.5, and linkage disequilibrium D=0.05.
The reversed-association founders are `[0.4,0.5,0.1,0]`, with the same allele
frequencies and D=−0.05. Every replicate starts with exact random-pairing
diplotype counts: `N*x_i^2` for identical haplotypes and `2*N*x_i*x_j`
for distinct ones. All ten actual founder counts are audited, not merely
the two marginal allele frequencies.

The two symbolic loci are adjacent SLiM positions 0 and 1. There is one
possible breakpoint, so rate `c` is directly the recombination fraction.
Treatments use c=0 (complete linkage) or c=0.5 (independent assortment at
meiosis). These are illustrative linkage controls, not empirical rates
between real adjacent bases. Parents are sampled independently with fitness
weights, allowing incidental selfing. There is no new mutation, migration,
sex distinction, demographic change, or gene conversion. Fixed mutations
remain explicit so their alleles can still be counted.

## Executed results

At generation 30, across all 256 replicates per treatment:

| Treatment | Mean neutral p(B) | Monte Carlo SE of mean p(B) | Mean gene diversity at B |
| --- | ---: | ---: | ---: |
| Selected A, c=0 | 0.999589844 | 0.000078039 | 0.000816870 |
| Selected A, c=0.5 | 0.553041992 | 0.005999189 | 0.476018054 |
| No selection, c=0 | 0.497236328 | 0.006347396 | 0.479437109 |
| Selected A, reversed association, c=0 | 0.000415039 | 0.000087896 | 0.000825793 |

Gene diversity means the within-population `H_B=2p(B)(1−p(B))`, averaged
across replicates. It is not observed individual heterozygosity or diversity
calculated after pooling the populations. The corresponding gene-diversity
SEs are 0.000155238, 0.002020772, 0.001628466, and 0.000174578.

All replicates contribute, with no filtering for fixation or survival.
In the linked selected treatment, A had fixed in **205/256** populations
by generation 30; fixation was not imposed. Starting with 80 copies of A
deliberately avoids modeling the establishment of a new single-copy mutation.
The strong illustrative selection coefficient and these results are not
empirical estimates or a real-genome benchmark.

Recombination limits the later hitchhiking effect, but does not undo an
allele-frequency change already caused by selecting parents. At generation
one the exact expected p(B) is **23/44 = 0.522727273** for both recombination
treatments. Observed means are **0.522275391** (SE 0.001128706) and
**0.523168945** (SE 0.001153499). Reversed association gives expectation
**21/44**; no selection gives **1/2**. Thus neutral does not mean that an
allele's frequency is unaffected by selection elsewhere.

## Independent mathematics and tests

For the random-pairing deterministic recurrence, let `x` be the four
haplotype frequencies, `p=x_Ab+x_AB`, `D=x_AB*x_ab−x_Ab*x_aB`, and s be
1 or 0 for the selected or neutral treatment. Mean parental fitness is
`W=1+s*p`. Before the recombination adjustment, haplotype i contributes

```text
y_i = x_i * [1 + (s/2)*(indicator_of_A_on_i + p)] / W.
```

With `delta=c*(1+s/2)*D/W`, the gamete vector is
`q=[y_ab−delta, y_aB+delta, y_Ab+delta, y_AB−delta]`.
The tests independently enumerate all 16 ordered diploids, weight their
fitnesses, and sum all four nonrecombinant/recombinant meiosis paths using
exact rational arithmetic. This agrees with the compact recurrence.

For the audited founders, offspring are 2N independent gametes with these
probabilities. If their B probability is qB:

```text
E[pB] = qB
Var(pB) = qB*(1−qB)/(2N)
E[H_B] = 2*qB*(1−qB)*(1−1/(2N))
E[D_sample] = D_gamete*(1−1/(2N)).
```

An independent enumeration of all 4^4 gamete draws for N=2 verifies the
finite-sampling moments. The neutral control additionally has the exact
multi-generation expectation `E[H_B(t)]=0.5*(1−1/(2N))^t`.

**The later selected deterministic curves are infinite-population
benchmarks, not exact finite-population ensemble means.** Finite parental
diplotype frequencies fluctuate; selection and drift cannot generally be
represented by inserting mean haplotype frequencies into a nonlinear
recurrence. The report keeps these curves separate from measured means.

The **24 tests** cover exact founders, independent meiosis and finite-sampling
enumeration, recombination and sign controls, actual fitness/version audits,
replicate coverage, first-generation mean/variance/diversity/LD checks,
neutral diversity through time, absence of impossible haplotypes without
recombination, fixation implications, invalid input/output, deterministic
seeds, CLI behavior, and checked output. Five-SE bounds are numerical
simulation sanity checks, not confidence intervals for biological parameters.

## Sources and licenses

Simulator: Haller & Messer (2023),
[SLiM 4: Multispecies Eco-Evolutionary Modeling](https://doi.org/10.1086/723601).
The [official version-matched manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf)
documents fitness in section 1.5.4, independent parent sampling under
`initializeSLiMOptions()`, and the breakpoint probability under
`initializeRecombinationRate()` (printed pages 623–624).

The Eidos model, Python code, tests, output, and prose here are original and
MIT licensed. SLiM is a separately installed GPLv3 program; its source and
license remain with its [official project](https://messerlab.org/slim/).
No upstream recipes, paper figures/text, or real-world/private genetic data
are redistributed.
