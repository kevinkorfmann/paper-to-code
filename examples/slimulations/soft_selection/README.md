# Local fitness normalization changes the direction of selection

An original synthetic SLiM experiment, added 2026-11-06. The same opposing
fitnesses in two habitats can give different global allele-frequency changes
depending on how much each habitat contributes to the mating pool. A rare
allele increases in expectation under locally normalized reproduction, but has
no expected change under the unscaled globally normalized control.

This is a **Levene-inspired, one-generation reproductive-selection** example,
not a reproduction of Levene's demographic life cycle, a proof of permanent
polymorphism, or an estimate for a natural population. It differs from the
repository's sexual-antagonism example: both parental roles draw from the same
mixed habitat pool, and the treatment changes habitat contributions rather
than requiring one parent of each sex.

## Experiment and actual output

Each of 512 independent populations has 200 diploid founders, with 100 assigned
to each habitat. At initial p(A)=0.2, **each** habitat has exactly 64 aa, 32 Aa,
and 4 AA founders. At p=0.8 these counts are reversed; at p=0.5 they are 25,
50,25. These are exact Hardy–Weinberg proportions, not random initial samples.

| Habitat | aa | Aa | AA |
|---|---:|---:|---:|
| 0 | 1 | 2 | 3 |
| 1 | 3 | 2 | 1 |

The table gives relative reproductive weights, not survival probabilities.
There is no within-habitat heterozygote advantage. Two parental roles are sampled
independently with replacement from the whole population. Each parent supplies
a Mendelian gamete. SLiM performs actual fitness-weighted Wright–Fisher
reproduction, not an analytical draw substituted for the simulation.

- **Soft treatment:** divide each founder's raw weight by its habitat's realized
  mean raw weight. Since habitats have equal numbers, their **expected**
  contributions to the global pool are equal.
- **Hard treatment:** use raw weights. A habitat contributes in proportion to
  the sum of its founders' raw weights.
- **Neutral control:** all weights are 1.

| Treatment | Initial p | Mean offspring p | Exact expected p | Exact SE of mean |
|---|---:|---:|---:|---:|
| Soft | 0.2 | 0.22643555 | 0.22637363 | 0.00092473 |
| Hard | 0.2 | 0.19793457 | 0.20000000 | 0.00088388 |
| Soft | 0.8 | 0.77225098 | 0.77362637 | 0.00092473 |
| Hard | 0.8 | 0.80195313 | 0.80000000 | 0.00088388 |
| Soft | 0.5 | 0.49928223 | 0.50000000 | 0.00110485 |
| Neutral | 0.2 | 0.20050293 | 0.20000000 | 0.00088388 |
| Soft, habitat 0 weights doubled | 0.2 | 0.22665039 | 0.22637363 | 0.00092473 |
| Hard, habitat 0 weights doubled | 0.2 | 0.23099609 | 0.22962963 | 0.00092939 |

At p=0.2, habitat 0 has lower mean raw fitness, 1.4 versus 2.6. It supplies
35% of expected parental draws under hard selection, but 50% under soft
selection. The observed proportions are **0.34983398** and **0.50110352**.
Local normalization prevents the A-favoring habitat from contributing less
simply because most of its founders currently carry a.

Multiplying every raw weight in habitat 0 by two leaves its within-habitat
relative fitnesses unchanged. It cancels under soft normalization, but changes
hard selection by increasing that habitat's contribution. A same-seed test
confirms identical simulated rows for the two soft scales, not just similar
averages. The table uses different seeds for its eight treatments.

All eight frequency means, eight habitat-contribution means and 24 genotype
frequencies lie within six exact standard errors of their conditional
expectations. Every replicate counts. These fixed-seed checks diagnose obvious
implementation errors; they do not establish universal biological claims.

## Derivation and finite-sampling prediction

With HWE founders at frequency p, the two habitat mean fitnesses and selected
gamete frequencies are

```text
w0 = 1 + 2p                 w1 = 3 - 2p
q0 = p(2+p)/(1+2p)          q1 = p(2-p)/(3-2p)
q_soft = (q0 + q1)/2
       = p + p(1-p)(1-2p)/[(1+2p)(3-2p)]
q_hard = (w0*q0 + w1*q1)/(w0+w1) = p.
```

For p=1/5, q_soft=103/455. The soft expected change is positive below 1/2
and negative above 1/2 for these weights and HWE founders. This restoring
one-generation tendency is not a simulated long-term equilibrium result.
With habitat 0 multiplied by c, its hard-selection contribution is
c*w0/(c*w0+w1), and q_hard=(c*w0*q0+w1*q1)/(c*w0+w1).
For c=2, q_hard=p(6+p)/(5+2p), giving 31/135 at p=1/5.

Conditional on the **fixed** founders and this independent reproductive
sampling, each of 2N offspring gene copies is an independent Bernoulli(q)
draw. Thus X~Binomial(2N,q), E[p_hat]=q, Var(p_hat)=q(1-q)/(2N), and the
standard error of a mean over R independent populations is
sqrt[q(1-q)/(2NR)]. Offspring genotype probabilities are (1-q)^2,
2q(1-q),q^2. The tests independently enumerate founders, their homologs and
ordered gamete pairs using exact rational arithmetic.

Soft normalization makes habitat 0's parental-draw probability 1/2, not its
realized number of draws exactly N. That number is Binomial(2N,1/2). It is
recorded for every replicate. Neither realized allele frequency nor realized
habitat contribution is forcibly set to its expected value.

## Assumptions and limits

- One biallelic autosomal locus, fixed total census size, nonoverlapping
  generations and independent random mating. Incidental selfing is allowed.
- Founder habitat labels determine reproductive weight only. All offspring
  are produced in a common pool. This represents global mixing; there is no
  separate migration event or persistent spatial structure in this experiment.
- No new mutation or recombination. The mutation type's own coefficient is
  zero, but the locus is selected through `fitnessEffect()`, not neutral.
- Local regulation is represented by deterministic weight normalization.
  We do not sample local survival, impose exact realized habitat quotas,
  model resource dynamics, or reassign offspring to habitats for later generations.
- Initial habitat sizes and genotype counts are identical. The displayed
  formulas use HWE founders; allele frequency alone would not determine the
  same result for arbitrary genotype counts.
- Equal expected p in the unscaled hard control is a symmetry of these weights,
  not a general property of hard selection. The scaled control breaks it.
- Parameters are illustrative. No empirical genomes, private research,
  source-paper code, figures or datasets are used.

## Reproduce

Reference environment: **SLiM 4.0.1 / Eidos 3.0.1 and CPython 3.14.5**.
SLiM 4.0.1 is the repository's reproducibility pin, not a latest-version claim.
Python uses only its standard library. SLiM is GPL-3.0; this original example
uses the repository MIT license. No GPU, paid API or new service is needed.

From the repository root, with `slim` and `python3.14` on PATH:

```sh
slim -v
python3.14 -c "import platform; assert platform.python_version() == '3.14.5'"
python3.14 -m unittest discover -s examples/slimulations/soft_selection -v
python3.14 -m examples.slimulations.soft_selection.demo --check-sample
```

The eight default seeds are **202611060–202611067** in the order in
`TREATMENTS` in [demo.py](demo.py). [sample_summary.json](sample_summary.json)
is actual checked output, including seeds, full-precision numerical results,
observed and exact standard errors, genotype frequencies and source-parent
fractions. Exact replay deliberately includes runtime metadata, so use the
pinned Python version.

For a small custom run:

```sh
slim -s 1234 -d REPLICATES=8 -d MODE=1 -d START=1 -d SCALE=2.0 \
  examples/slimulations/soft_selection/model.slim
```

`MODE=0/1/2` means neutral/soft/hard; `START=1/2/3` means p=0.2/0.5/0.8.
`N` is total diploid census and must be a multiple of 200, between 200 and 1000.
`SCALE` is habitat 0's raw weight multiplier, in [0.01,10]. `REPLICATES` is 1–2048.
The script checks each habitat's founder counts and recomputed fitness, then
emits `INIT`, `WEIGHT` and `DATA` rows. The Python parser checks completeness,
uniqueness, counts, and expected normalized habitat weights.

The existing [CI workflow](../../../.github/workflows/tests.yml) builds official
SLiM source commit `09fcf604a06307cf75098dfa0a2171a9b7852216`, with archive SHA-256
`e2448f4aaca909ffdfb33690f0d1dea2a102c2ef1b71373d72661c31df29b481`.
Its build commands are also a reproducible installation recipe. The existing
public-repository runner job runs these 26 tests and exact sample replay.
No additional runner, schedule, credential or permission is introduced.

## Tests and sources

The **26 tests** cover exact rational founder/homolog enumeration, independent
Mendelian genotype enumeration, rare/common direction, allele-exchange symmetry,
neutral and p=1/2 controls, habitat scaling, parser corruption and missing rows,
Python and SLiM input validation, all eight actual simulations, same-seed
reproducibility, different seeds, exact soft scaling invariance, actual habitat
contribution variation, larger census size, 1,024-replicate binomial variance,
and CLI sample replay. All pass locally in the reference environment.

- [Levene (1953), *Genetic Equilibrium When More Than One Ecological Niche is
  Available*](https://doi.org/10.1086/281792), American Naturalist 87:331–333.
  Historical source for local selection followed by habitat mixing.
- [Novak (2011), *The number of equilibria in the diallelic Levene model with
  multiple demes*](https://doi.org/10.1016/j.tpb.2010.12.002), especially its
  [model introduction and equations 1–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC3120059/).
  Context for the locally normalized recurrence; this example does not
  reproduce that paper's equilibrium-counting results.
- [Haller & Messer (2023), *SLiM 4: Multispecies Eco-Evolutionary
  Modeling*](https://doi.org/10.1086/723601) and the
  [official SLiM project](https://messerlab.org/slim/).

These sources motivate the mechanism and simulator. The implementation,
finite-sampling checks and numerical example here are original.
