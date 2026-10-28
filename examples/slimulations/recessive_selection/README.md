# Recessive deleterious alleles can be weakly selected when rare

A harmful allele can have a large effect in homozygotes and little effect on
its own frequency when most copies are in heterozygotes. This original SLiM
experiment isolates that mechanism in **one generation**, comparing recessive,
additive, dominant, and neutral effects at the same locus.

The classical source is J. B. S. Haldane, **“A Mathematical Theory of Natural
and Artificial Selection—I” (1924)**, *Transactions of the Cambridge
Philosophical Society* 23:19–41, reprinted in 1990
([primary paper and abstract](https://doi.org/10.1007/BF02459574)). Its abstract
explicitly identifies weak selection on rare autosomal recessive characters.
This is an independently designed illustration of that principle, not a
reproduction of the full paper.

## Run

Use **SLiM 4.0.1 / Eidos 3.0.1** and **CPython 3.14.5**. The Python wrapper
uses only the standard library. From the repository root:

```sh
slim -v
python3.14 -m venv --without-pip .venv
. .venv/bin/activate
python -c "import platform; assert platform.python_version() == '3.14.5'"
python -m unittest discover -s examples/slimulations/recessive_selection -v
python -m examples.slimulations.recessive_selection.demo --check-sample
```

For a small exploratory run with a different seed:

```sh
python -m examples.slimulations.recessive_selection.demo --replicates 32 --seed 123
```

The original Eidos script can also be run directly:

```sh
slim -s 202610281 -d MODE=1 -d N=400 -d AA=1 -d HET=38 \
  -d REPLICATES=512 examples/slimulations/recessive_selection/selection.slim
```

`MODE=0,1,2,3` denotes neutral, recessive, additive, and dominant effects.
Raw output includes audited founder fitnesses and genotype counts for every
replicate at elapsed generations 0 and 1. `--check-sample` compares the default
JSON output byte for byte with [sample_summary.json](sample_summary.json).

The version is pinned to the repository's existing simulator environment,
not advertised as the latest SLiM. Official [4.0.1 downloads and manual](https://github.com/MesserLab/SLiM/releases/tag/v4.0.1)
remain available. The existing [CI job](../../../.github/workflows/tests.yml)
builds official commit `09fcf604a06307cf75098dfa0a2171a9b7852216`, verifying the
source archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
No new dependency, paid service, or scheduled workflow is introduced.

## Model and analytical prediction

Each replicate contains **N=400 diploid individuals** at one synthetic locus.
A is the deleterious allele. There is no mutation, migration, or recombination.
Fitness affects parental sampling in SLiM's Wright–Fisher model; population
size remains fixed. The first and second parents are sampled independently,
so incidental selfing is allowed. Alleles segregate by Mendelian inheritance.

| Treatment | aa fitness | Aa fitness | AA fitness |
| --- | ---: | ---: | ---: |
| Neutral | 1 | 1 | 1 |
| Recessive | 1 | 1 | 0.2 |
| Additive | 1 | 0.6 | 0.2 |
| Dominant | 1 | 0.2 | 0.2 |

SLiM uses `1+s` for AA and `1+h*s` for Aa. Selected treatments use `s=-0.8`
and `h=0,0.5,1`; the neutral control uses `s=0`. The script recalculates and
prints the engine's actual genotype fitnesses, which the wrapper verifies.
“Additive” here means additive in relative fitness, not log fitness.

The eight main treatments combine these four fitness schemes with starting
frequencies **0.05 and 0.5**, using exact Hardy–Weinberg founder counts. At
p=0.05, the counts are `(aa,Aa,AA)=(361,38,1)`: 38 of the 40 A copies are in
heterozygotes. At p=0.5, the counts are `(100,200,100)`. Founders are installed
after tick 1 reproduction; that point defines elapsed time zero. The next
reproduction produces the measured offspring.

For arbitrary parental counts, let their genotype fitnesses be
`(w_aa,w_Aa,w_AA)`. The exact probability of transmitting A is

```text
q = (n_AA*w_AA + n_Aa*w_Aa/2)
    / (n_aa*w_aa + n_Aa*w_Aa + n_AA*w_AA).
```

Conditional on these founders, offspring A-copy count is `Binomial(2N,q)`:
`E[p_next]=q`, `Var(p_next)=q(1-q)/(2N)`. Offspring genotype probabilities
are `q²` for AA and `2q(1-q)` for Aa. The code computes these expectations
using exact rational arithmetic and separately enumerates parental genotypes
and Mendelian gametes in tests.

For Hardy–Weinberg founders with a fully recessive cost `c=0.8`,

```text
q = (p - c*p²)/(1 - c*p²)
q - p = -c*p²*(1-p)/(1-c*p²).
```

The frequency change is quadratic in p near zero. These expressions are
exact for this first generation. Iterating a deterministic curve is generally
not the exact mean trajectory of a finite selected population.

## Executed results

There are **512 replicates per treatment**, with seeds **202610280–202610288**
in the order recorded in the JSON. All replicates are included.

| Starting p=0.05, HWE founders | Exact expected p after one generation | Observed mean | Monte Carlo SE |
| --- | ---: | ---: | ---: |
| Neutral | 0.050000000 | 0.050283203 | 0.000332533 |
| Recessive | 0.048096192 | 0.048452148 | 0.000330454 |
| Additive | 0.030208333 | 0.030405273 | 0.000272438 |
| Dominant | 0.010845987 | 0.010866699 | 0.000174776 |

The recessive allele's theoretical relative decline is about **3.81%** at
p=0.05, compared with **25%** at p=0.5. In the latter case the observed mean is
**0.376264648**, Monte Carlo SE **0.000752530**, versus expectation **0.375**.
These are finite-replicate results, not claims of exact agreement.

A ninth control keeps p=0.05 but puts **all 40 A copies into heterozygotes**:
`(aa,Aa,AA)=(360,40,0)`. All parents then have equal fitness in the recessive
model, so the exact first-generation expectation remains 0.05. The observed
mean is **0.049951172**, SE **0.000337130**. Random mating creates AA offspring
(mean fraction **0.002529297**), which can be selected against in subsequent
generations; this control does not imply permanent protection.

Even in the recessive HWE treatment, allele frequency **increased in 189/512
replicates**. Selection sets an expectation, not the direction of every finite
realization. The reported SE measures Monte Carlo uncertainty in the replicate
mean; it is not uncertainty about a real population's parameters.

## Tests and limitations

The **22 tests** check actual SLiM execution, founder counts and engine
fitnesses, exact parental/gamete enumeration, exact small-binomial moments,
the recessive change formula, conditional mean and variance, offspring
genotype proportions, rare/common contrasts, neutral and all-heterozygote
controls, absorbing boundaries, seed reproducibility, input/output validation,
and checked CLI output. Stochastic regression checks use six theoretical
standard errors as a tolerance; this is not a biological confidence claim.

One locus, fixed fitness, fixed population size, hermaphroditic random mating,
and constructed founders deliberately remove other causes. No inbreeding
program, pedigree, mutation-selection balance, environmental dependence,
linkage effect, long-term extinction probability, or human genetic trait is
modeled. The strong cost makes the mechanism visible; it is not an empirical
estimate. The neutral shared utilities are reused from this repository; the
new SLiM model, analysis, tests, and documentation are original.

## Sources and licenses

- [Haldane (1924), reprinted 1990](https://doi.org/10.1007/BF02459574): classical selection and dominance principle.
- [Haller and Messer (2023), SLiM 4](https://doi.org/10.1086/723601): simulator reference.
- [Official SLiM 4.0.1 manual](https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM_Manual.pdf): sections 1.1 and 1.5, and `initializeSLiMOptions()` document WF parental sampling, fitness, and incidental selfing.
- [Pinned mutation fitness definitions](https://github.com/MesserLab/SLiM/blob/09fcf604a06307cf75098dfa0a2171a9b7852216/core/mutation.h): the version-matched `1+s` and `1+h*s` convention.

The example uses the repository's MIT license. SLiM is an external GPL-3.0
program and retains its own license. No upstream implementation, paper figure,
private research, or real genotype data is copied into this example.
