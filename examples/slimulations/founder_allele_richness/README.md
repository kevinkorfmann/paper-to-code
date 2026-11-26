# Founder sampling can erase alleles faster than it erases gene diversity

One neutral SLiM Wright–Fisher reproduction step creates 2,048 cohorts of
five diploids. Both parents are drawn from the same 50-diploid source.
There are 12 founder haplotype states: 70 unmarked copies, 20 copies of
one marker, and ten distinct singletons. No selection, recombination
or new mutation. Markers label a single nonrecombining haplotype;
this is not a cull of existing diploid adults.

With m=10 sampled gene copies and source frequencies p_i, expected
allele count is sum_i [1-(1-p_i)^m]. Founder gene diversity is
1-sum_i (n_i/m)^2 (the chance two draws **with replacement** differ);
its expectation is (1-1/m)(1-sum_i p_i^2). An estimator for the source
based on sampling distinct copies would use a different denominator.
The example models only one founding generation, not long-run recovery.

## Reproduce

SLiM 4.0.1 and CPython 3.14.5; seed 20261126. With slim on PATH:

```sh
python3 demo.py --test
python3 demo.py > actual.txt
diff -u expected.txt actual.txt
```

[Nei, Maruyama & Chakraborty (1975)](https://doi.org/10.1111/j.1558-5646.1975.tb00807.x)
and [SLiM 4](https://doi.org/10.1086/723601).
