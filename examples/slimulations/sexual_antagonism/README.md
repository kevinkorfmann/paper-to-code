# Sexually antagonistic selection can hide behind a stable allele frequency

An original, synthetic, one-generation SLiM experiment, added 2026-11-04.
This isolates **sex-specific reproductive selection**, rather than changing the
sex ratio, seeding different allele frequencies in the sexes, or assigning
heterozygote advantage. No private research or empirical data are used.

## Experiment and result

Each of 512 independent populations has 200 diploids, 100 of each sex. Both
sexes start with exactly 25 aa, 50 Aa, and 25 AA individuals at one autosomal
locus. We compare three fitness-weighted random-mating treatments:

| Treatment | Male fitness aa,Aa,AA | Female fitness aa,Aa,AA | Expected offspring p(A) | Expected heterozygosity |
|---|---|---|---:|---:|
| Neutral | 1,1,1 | 1,1,1 | 0.5 | 0.5 |
| Antagonistic | 9,3,1 | 1,3,9 | 0.5 | 0.625 |
| Concordant | 1,3,9 | 1,3,9 | 0.75 | 0.375 |

The checked SLiM output gives antagonistic p(A) = **0.5009961** and
heterozygosity **0.6237891**, compared with neutral **0.4997754** and concordant
**0.3740332** heterozygosity. All nine simulated genotype frequencies fall
within six binomial standard errors of the exact conditional probabilities.
The bounds are regression diagnostics, not confidence that this model describes
any organism, and are not adjusted hypothesis tests.

## Independent derivation

For founder genotype fractions (1/4,1/2,1/4) and fitnesses (w0,w1,w2),
the selected gamete frequency is q = (w1+w2)/(w0+2*w1+w2).
Thus antagonistic male and female gametes have qM=1/4 and qF=3/4 even
though the unweighted parental allele frequencies are both 1/2.
Independent maternal and paternal draws give offspring probabilities

- P(aa) = (1-qM)(1-qF)
- P(Aa) = qM(1-qF)+(1-qM)qF
- P(AA) = qM*qF.

At p=(qM+qF)/2, expected heterozygosity exceeds 2p(1-p) by
(qM-qF)^2/2, here 1/8. This compares with HWE at the **expected** offspring
allele frequency, not the finite-sample plug-in estimate. The gamete pools,
not unequal census sizes or opposite founder allele frequencies, create the
excess. Offspring counts have a multinomial distribution conditional on the
fixed founders. Neutral and concordant controls remove the difference between
gamete pools. Heterozygotes have intermediate reproductive fitness within each
sex; no within-sex overdominance is imposed.

## Assumptions and limits

- Sexual Wright–Fisher reproduction, one autosomal biallelic locus, Mendelian
  transmission, fixed 1:1 sex ratio and non-overlapping generations.
- Fitness is relative parental reproductive success within each sex, not
  individual survival, demographic growth, or lifetime fitness.
- No migration, new mutation, recombination, mate preference or environmental
  variation. Each replicate's allele is shared across its founders.
- Founder genotype counts are deterministic, not sampled from an equilibrium.
- Output is the next generation before its own reproductive selection. This
  one-generation result does not demonstrate long-term polymorphism, stable
  equilibrium, adaptation, or a general fitness benefit of sexual antagonism.

## Reproduce

Reference environment: **CPython 3.14.5 and SLiM 4.0.1**, Python standard
library only. 4.0.1 is deliberately the repository's reproducibility pin, not
the latest release (the official site listed 5.2 when checked on 2026-11-04).
The existing CI builds official commit
`09fcf604a06307cf75098dfa0a2171a9b7852216`, verifying archive SHA256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
Put that `slim` executable on PATH, then run from the repository root:

```sh
python -m unittest discover -s examples/slimulations/sexual_antagonism -v
python -m examples.slimulations.sexual_antagonism.demo
python -m examples.slimulations.sexual_antagonism.demo --check-sample
slim -s 202611041 -d MODE=1 examples/slimulations/sexual_antagonism/model.slim
```

Twelve tests cover the exact probabilities, general excess identity, fitness
scale invariance and allele relabeling, invalid inputs, malformed/missing and
duplicate rows, actual SLiM integration, deterministic replay, a second seed
at N=8, and the stored output. The Eidos script checks every founder sex's
realized fitness and emits founder counts for Python to validate. The Python
wrapper verifies the executable version and refuses missing observations.
There is no Python fallback pretending to run SLiM. Seeds and measured runtime
versions are recorded in `sample_summary.json`. Exact replay is a check for
the pinned runtime; random streams need not match other SLiM versions.

## Sources and attribution

- [Official SLiM site and version history](https://messerlab.org/slim/).
- [Pinned SLiM source](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216), for the WF sexual reproduction and fitness callback implementation.
- [Connallon, Cox & Calsbeek (2010), Fitness consequences of sex-specific selection](https://doi.org/10.1111/j.1558-5646.2009.00934.x), primary theoretical context on sex-specific and antagonistic selection. This demo is not a reproduction of that paper or its quantitative conclusions.

The elementary one-generation derivation above is supplied in full; the code
and controls are original educational work within this repository's MIT license.
