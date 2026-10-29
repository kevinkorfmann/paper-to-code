# Transmission distortion needs heterozygous parents

An allele can increase without its carriers having a parental fitness advantage:
heterozygotes may transmit it more often than the Mendelian probability of 1/2.
This original, synthetic SLiM example isolates **transmission distortion**, a
generic population-genetic representation of meiotic drive. It is not a model
of a particular molecular mechanism or organism.

The classical source is Sandler and Novitski, **“Meiotic Drive as an Evolutionary
Force” (1957)**, *The American Naturalist* 91:105–110
([primary paper](https://doi.org/10.1086/281969)). Its abstract describes
heterozygotes producing unequal frequencies of the two gamete types and the
consequences for allele frequencies. The experiment and derivations below are
independently constructed, not a reproduction of that paper.

## Run and reproduce

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**. From the repository root:

```sh
slim -v
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/slimulations/transmission_distortion -v
python -m examples.slimulations.transmission_distortion.demo --check-sample
```

The final command compares the actual report byte for byte with
[sample_summary.json](sample_summary.json). No third-party Python package is
needed. An exploratory run with fewer replicates is:

```sh
python -m examples.slimulations.transmission_distortion.demo --replicates 32 --seed 123
```

The original Eidos model can also run directly:

```sh
slim -s 202610291 -d N=200 -d AA=50 -d HET=100 -d TAU=0.75 \
  -d REPLICATES=512 -d GENERATIONS=2 \
  examples/slimulations/transmission_distortion/transmission.slim
```

Raw output includes engine versions, minimum/maximum founder fitness, and
`DATA,generation,replicate,N,AA,Aa` rows. Both versions and the fitness audit
are verified by the wrapper. `GENERATIONS` may be 0, 1, or 2.

This is the repository's existing pinned environment, not a claim that SLiM
4.0.1 is the newest release. The [official release](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1)
includes the manual. The existing [CI](../../../.github/workflows/tests.yml)
builds official source commit `09fcf604a06307cf75098dfa0a2171a9b7852216` from
an archive checked against SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
There are no new dependencies, scheduled workflows, paid services, or datasets.

## The deliberately simple model

Each replicate has **N=200 diploid individuals** at one synthetic locus.
The alleles are a and A. All genotype fitnesses are exactly 1. The WF model
samples parents independently with replacement, allowing incidental selfing.
Population size is fixed, and there is no new mutation, recombination, or
migration. Each transmission is independent, including when the same parent
is chosen twice.

An aa parent always transmits a; an AA parent always transmits A; an Aa parent
transmits A with probability **τ**. The same τ applies to **both parental
roles**, with no separate sexes. τ=0.5 is the Mendelian control, τ=0.75 favors
A, and τ=0.25 reverses the bias. Transmission bias is a selective advantage at
inheritance, so equal parental fitness does **not** make the whole process
neutral.

The `modifyChild()` callback replaces each heterozygous parent's single-locus
transmission with an independent Bernoulli(τ) draw. It adds or removes the
*existing parental allele* on the corresponding child genome as needed, never
rejects a child, and never reweights parent choice. These edits implement an
inheritance probability; they are not a biological mutation or gene-conversion
process. Homozygous transmissions are untouched. Founders are installed after
tick 1 reproduction, which defines elapsed generation zero.

Five treatments all start at p(A)=0.5:

| Founder pairing | aa | Aa | AA | τ |
| --- | ---: | ---: | ---: | ---: |
| Exact Hardy–Weinberg counts | 50 | 100 | 50 | 0.5, 0.75, or 0.25 |
| All heterozygotes | 0 | 200 | 0 | 0.75 |
| Only homozygotes | 100 | 0 | 100 | 0.75 |

## Exact two-generation predictions

For fixed parental counts, write H=n_Aa/N. The transmitted A probability is

```text
q = (n_AA + τ*n_Aa)/N
q - p = (τ - 1/2)*H.
```

Thus bias acts only through heterozygotes. Offspring A-copy count in generation
1 is exactly `Binomial(2N,q)`, so `E[p1]=q` and `Var(p1)=q(1-q)/(2N)`.
Genotype probabilities are `(aa,Aa,AA)=((1-q)²,2q(1-q),q²)`.

A generation-1 individual transmits A with probability Y, where Y is 0, τ,
or 1 for aa, Aa, or AA. The random gamete frequency for generation 2 is the
sample mean Q of N independent Y values. Define

```text
m = E[Q]   = q² + 2τ*q(1-q)
v = Var(Q) = [q² + 2τ²*q(1-q) - m²]/N.

E[p2]   = m
Var(p2) = v + [m(1-m)-v]/(2N)
E[f_AA,2] = m² + v
E[f_Aa,2] = 2[m(1-m)-v].
```

These are **exact finite-population expectations**, conditional on the fixed
founders and model assumptions. The second-generation heterozygote fraction
has a drift correction; simply using `2m(1-m)` is not exact. The Python
oracle uses rational arithmetic. Tests separately enumerate every possible
first-generation genotype count and second-generation allele count for a
small N, without calling these prediction formulas.

These two-generation results do not imply that a deterministic frequency
recursion remains the exact finite-population ensemble mean indefinitely.

## Actual executed results

**512 replicates per treatment**, seeds **202610290–202610294**, in table order.
All replicates are included. The following are actual pinned-version outputs:

| Founders, τ | Exact E[p1] | Observed p1 mean ± Monte Carlo SE | Exact E[p2] | Observed p2 mean ± Monte Carlo SE |
| --- | ---: | ---: | ---: | ---: |
| HWE, 0.5 | 0.500000000 | 0.500664062 ± 0.001024831 | 0.500000000 | 0.499765625 ± 0.001488938 |
| HWE, 0.75 | 0.625000000 | 0.624243164 ± 0.000984750 | 0.742187500 | 0.741621094 ± 0.001396420 |
| HWE, 0.25 | 0.375000000 | 0.376298828 ± 0.001046216 | 0.257812500 | 0.258608398 ± 0.001449698 |
| All Aa, 0.75 | 0.750000000 | 0.748808594 ± 0.000993560 | 0.843750000 | 0.843027344 ± 0.001126387 |
| Only aa/AA, 0.75 | 0.500000000 | 0.499941406 ± 0.001138333 | 0.625000000 | 0.624326172 ± 0.001596879 |

Identical allele frequency does not imply identical immediate response: all-Aa
founders have the largest first-generation shift; homozygote-only founders
have none in expectation. Random mating then creates heterozygotes, allowing
bias to act in their offspring. The homozygote-only treatment's mean
heterozygote fraction is **0.501367187** at generation 1, near expectation 0.5.

Monte Carlo SE quantifies uncertainty in a replicate mean, not uncertainty in
a real population's parameters. These finite means are not exact expectations.

## Checks and limitations

The **22 tests** cover independent parental/gamete enumeration, complete small-N
multinomial/binomial enumeration for generation 2, both generations' means and
variances, genotype fractions, the finite-population correction, Mendelian and
reversed-bias controls, allele relabeling, and founder-pairing contrasts.
Variance tolerances use independently derived fourth moments, also checked by
full enumeration. Stochastic regression checks use conservative six-SE bounds,
not a biological confidence claim.

Exact engine controls include all-Aa founders with τ=0 or 1 (deterministic loss
or fixation), absorbing boundaries, and identical generation-1 output across
τ values for homozygote-only founders with the same seed. Tests also check
versions, equal founder fitness, all-replicate summaries, seed reproducibility,
invalid wrapper/direct-script inputs, parser guards, and checked CLI output.

No fertility costs, sex-specific transmission, suppressors, linkage, molecular
mechanism, empirical genotype data, long-term fixation probability, or natural
population is modeled. In particular, this is **not an R2d2 model** and does
not estimate any R2d2 transmission rate or fitness effect. Strong, constant τ
values make the toy contrast visible; they are not empirical estimates.

## Sources and licenses

- [Sandler and Novitski (1957)](https://doi.org/10.1086/281969): classical unequal-transmission principle.
- [Haller and Messer (2023), SLiM 4](https://doi.org/10.1086/723601): simulator reference.
- [Official SLiM 4.0.1 manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf): chapter 12 and section 25.5 explain child modification; `initializeSLiMOptions()` documents independent parental sampling and incidental selfing.
- [Pinned SLiM reproduction source](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/population.cpp#L1446-L1498): parent-to-child-genome mapping.

The new model, analysis, tests, and prose are original and use the repository's
[MIT license](../../../LICENSE). The small integer/version helpers are reused
from this repository's neutral-drift example. SLiM is an externally installed
GPL-3.0 program and retains its own license. No upstream model implementation,
paper figures, private research, real genotype data, or credentials are included.
