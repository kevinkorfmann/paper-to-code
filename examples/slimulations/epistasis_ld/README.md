# Epistasis can create LD without shifting allele frequencies

An original, synthetic SLiM experiment, added 2026-11-05. Two loci begin in
linkage equilibrium. Fitness depends only on their interaction. After one
reproductive generation, allele frequencies still have expectation 1/2, while
haplotypes become associated, even with free recombination.

This is distinct from the repository's neutral LD-decay and hitchhiking examples:
we **create** LD from balanced, independent founders using two-locus selection.
No empirical data, private research, or upstream code are used.

## Experiment and actual output

Each of 512 independent populations contains 256 diploids. The 16 ordered pairs
of haplotypes ab, aB, Ab, AB are represented exactly 16 times each. Both alleles
start at frequency 1/2, each haplotype at 1/4, with D = 0.

For derived-allele dosages nA and nB (each 0, 1, or 2), parental reproductive
fitness is

```text
w(nA,nB) = 1 + s (nA - 1)(nB - 1).
```

At s = 0.8, matching double homozygotes aabb and AABB have fitness 1.8,
opposing double homozygotes aaBB and AAbb have fitness 0.2, and genotypes
heterozygous at either locus have fitness 1. The mutation types have zero
single-locus coefficients, but these loci are **not neutral**: they participate
in epistatic selection. Fitness is imposed through a SLiM `fitnessEffect()`
callback, then recomputed and checked for every founder.

| Treatment | Mean sample D after reproduction | Exact finite-sample expectation | Exact SE of the mean |
|---|---:|---:|---:|
| s = +0.8, r = 0.5 | 0.04925588 | 0.04990234 | 0.00047799 |
| s = +0.8, r = 0 | 0.05007824 | 0.04990234 | 0.00047799 |
| s = -0.8, r = 0.5 | -0.05070405 | -0.04990234 | 0.00047799 |
| s = 0, r = 0.5 | 0.00066219 | 0 | 0.00048780 |

In the first treatment, measured p(A) = **0.50056839** and p(B) =
**0.50041199**. Its haplotype frequencies ab, aB, Ab, AB are
**0.29883575, 0.20059586, 0.20075226, 0.29981613**, near their exact values
**0.3, 0.2, 0.2, 0.3**. Realized populations fluctuate; the allele frequencies
are not forced to remain 1/2 in offspring.

All 16 treatment-by-haplotype frequencies and all four mean LD values are within
six exact standard errors of their conditional expectations. These broad bounds
are reproducibility diagnostics, not independent hypothesis tests or evidence
that this model describes a particular organism. Haplotype counts are correlated.
Full precision, seeds and measured versions are in [sample_summary.json](sample_summary.json).

## Exact derivation and sampling controls

The founders give equal probability to all ordered haplotype pairs, and mean
fitness is exactly 1. Weight each founder pair by w and enumerate its meiosis:
choose either homolog with probability 1/2, transmitting that haplotype with
probability 1-r or exchanging the second locus with probability r. The result is

```text
q(ab) = q(AB) = 1/4 + s/16
q(aB) = q(Ab) = 1/4 - s/16
p(A) = p(B) = 1/2
D = q(ab)q(AB) - q(aB)q(Ab) = s/16.
```

Why is this independent of r? Only double heterozygotes change their transmitted
haplotypes under recombination. Coupling AB/ab and repulsion Ab/aB founders
occur equally often and all have fitness 1, so the changes cancel. This is a
property of these balanced founders and this **first generation**, not a general
claim that recombination cannot reduce epistatic LD.

For M = 2N independent offspring gametes, counts are Multinomial(M,q).
The reported statistic is the average of each population's plug-in estimate,
not LD computed after pooling populations:

```text
D_hat = f(ab)f(AB) - f(aB)f(Ab) = f(AB) - p_hat(A)p_hat(B)
E[D_hat] = (1 - 1/M)D.
```

Thus with M = 512 and s = 0.8, the correct mean target is 0.04990234375,
not exactly 0.05. The exact variance for the balanced probabilities above is

```text
Var(D_hat) = (M-1)[M - 16(M-2)D^2] / (16 M^3).
SE(mean D_hat over R independent replicates) = sqrt(Var(D_hat)/R).
```

`demo.py` computes general multinomial first and second moments using
falling-factorial moments and rational arithmetic. Tests independently enumerate
all ordered gamete outcomes in tiny samples, verify the balanced closed form,
and enumerate founder meioses using exact rational probabilities. This checks
both the biological expectation and the finite-sample statistic without fitting
the expectation to the simulation output.

## Assumptions and limits

- Diploid Wright–Fisher generations, constant census, hermaphroditic random
  mating with independent fitness-weighted parental draws. Incidental selfing
  is allowed. No sex-specific effects or mate choice.
- Two biallelic loci at positions 0 and 1. The single intervening breakpoint has
  probability r, so r = 0.5 represents free recombination, despite adjacent
  coordinates in the toy genome. This is not a physical distance model.
- No new mutation, migration, other loci, environmental variation or demographic
  change. Each population has its own shared marker mutations.
- Founder diplotypes are balanced deterministic inputs, not a claim of natural
  equilibrium. Selection acts through relative reproductive success, not
  survival or population growth. Output is the next generation before its own
  reproductive selection.
- This is not a long-term equilibrium experiment, an inference method, or a
  reproduction of the cited paper's mutation–selection–drift models. LD has many
  possible causes; observing it alone would not identify epistasis.

## Reproduce

Reference runtime: **SLiM 4.0.1 and CPython 3.14.5**, Python standard library
only. SLiM 4.0.1 is the repository's reproducibility pin, not a claim to be the
latest release. Existing CI builds official source commit
`09fcf604a06307cf75098dfa0a2171a9b7852216`, checking archive SHA256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
Put the pinned `slim` executable on PATH and run from the repository root:

```sh
slim -v
python -m unittest discover -s examples/slimulations/epistasis_ld -v
python -m examples.slimulations.epistasis_ld.demo
python -m examples.slimulations.epistasis_ld.demo --check-sample
slim -s 202611050 -d S=0.8 -d R=0.5 examples/slimulations/epistasis_ld/model.slim
```

The 24 tests cover independent exact meiosis and sampling enumerations,
recombination and sign controls, finite-sample bias and variance, invalid inputs,
malformed/duplicate/missing rows, runtime failure handling, genuine SLiM runs,
a second seed at N=16, intermediate recombination, zero-fitness boundaries,
deterministic replay and checked CLI output. The Eidos script verifies all 16
founder diplotype counts and all founder fitnesses. The wrapper checks the
executable version and requires every population's output. It has no Python
simulation fallback. Exact random-stream replay is specific to the pinned runtime.

## Sources and attribution

- [Ragsdale (2022), *Local fitness and epistatic effects lead to distinct patterns of linkage disequilibrium in protein-coding genes*](https://doi.org/10.1093/genetics/iyac097), Genetics 221(4), iyac097. Primary context for epistasis and signed LD. Its sign conventions and fitness models differ from this original demonstration.
- [Haller & Messer (2023), *SLiM 4: Multispecies Eco-Evolutionary Modeling*](https://doi.org/10.1086/723601), simulation framework.
- [Official SLiM manual](https://www.benhaller.com/slim/SLiM_Manual.pdf) and [pinned source](https://github.com/MesserLab/SLiM/tree/09fcf604a06307cf75098dfa0a2171a9b7852216), WF parental sampling, fitness callbacks and recombination mechanics. The live manual may describe a newer release; executable behavior here is tested against 4.0.1.

The equations, implementation and tests here are original educational work under
the repository's MIT license. Citations establish context, not author endorsement.
