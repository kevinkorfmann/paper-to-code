# slimulations: source mixing creates signed LD

An original one-generation **SLiM 4.0.1** simulation. Two sources each have zero haplotype linkage disequilibrium (LD), but their mixed offspring population has nonzero LD. This demonstrates generation of an association, not the subsequent recombination decay covered by the earlier LD example.

## Model and estimand

Each of 1,024 independent replicates contains two source demes and a target, with 40 hermaphroditic diploids each. Neutral markers A and B occupy adjacent abstract positions. Source A is fixed AB and source B fixed ab. The phase-reversed control instead uses Ab and aB. Thus both sources have **D = 0** (they are monomorphic), while their allele frequencies differ. There is no mutation, recombination, selection, gene conversion or empirical data.

At tick 1, founders are created and markers installed. At tick 2 all target offspring choose one of the sources, with probability m for source A and 1−m for source B. **SLiM WF migration chooses a source per offspring and both parents come from that source.** It is not independent migration of 2N gametes or an exact quota of adult migrants. Sources remain fixed; every source and target count is checked. Random mating occurs within the chosen source, allowing incidental selfing. The run ends after this pulse.

For haplotypes ordered AB, Ab, aB, ab, the plug-in statistic is

```text
D_hat = x_AB - p_A*p_B.
```

The infinite-mixture identity with zero source LD is `D = m(1-m) Δp_A Δp_B`. Here it is +m(1−m) for coupling, −m(1−m) for reversed phase. In a finite target, the number K of source-A offspring is Binomial(N,m). Each contributes two identical haplotypes, so

```text
D_hat = ±(K/N)(1-K/N)
E[D_hat] = ±(1-1/N)m(1-m).
```

The factor is **1−1/N, not 1−1/(2N)**, because both chromosomes share the offspring's source. This is a finite-sampling property of the statistic, not weaker biological mixing.

## Actual seeded result

| Treatment | Mean measured D | Exact expected plug-in mean |
|---|---:|---:|
| m=0.25, coupling | 0.181992798 | 0.1828125 |
| m=0.50, coupling | 0.243615723 | 0.24375 |
| m=0.50, reversed phase | −0.243794556 | −0.24375 |
| Either source alone | 0 | 0 |

Seeds 202611200–202611204, complete settings and an illustrative target haplotype-count row are recorded in `sample_output.json`. Ensemble means are checked within seven exact Monte Carlo standard errors, computed by summing the finite binomial distribution. This is a regression tolerance, not an inferential confidence interval.

## Reproduce

CPython **3.14.5**, standard library only. Use the checksum-pinned SLiM 4.0.1 build in the repository workflow and put `slim` on PATH:

```sh
python -m unittest examples.slimulations.admixture_ld.test_demo -v
python -m examples.slimulations.admixture_ld.demo --check-sample
python -m examples.slimulations.admixture_ld.demo
```

The CLI accepts `--slim /path/to/slim` and checks its version. Twelve tests include an independent exhaustive source-assignment oracle, finite-size correction, phase reversal, source validation, output parsing, endpoint controls and multiple engine seeds.

## Limits and sources

Fixed sources and one pulse isolate the mechanism; this does not estimate an admixture date or ancestry fraction, simulate recombination decay, or imply that observed LD uniquely identifies admixture. No species-specific claims or real genomes are involved. Ordinary signed D, not r² or a standardized LD measure, is used; monomorphic source r² would be undefined.

- Loh et al. (2013), [Inferring Admixture Histories of Human Populations Using Linkage Disequilibrium](https://doi.org/10.1534/genetics.112.147330), initial mixture-LD identity. This toy does not implement ALDER or its dating procedure.
- [SLiM 4.0.1 manual](https://gensoft.pasteur.fr/docs/SLiM/4.0.1/SLiM_Manual.pdf), §§1.6 and 22.2.1, WF source-population semantics.

All model code, tests and fixtures are original tutorial work.
