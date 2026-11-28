# Same mean gene diversity, different odds of losing polymorphism

Begin with 20 diploids and a single neutral marker at frequency 1/2.
Send independent populations through census sizes 2 → 6 → 20 or
6 → 2 → 20, one Wright–Fisher reproduction step at each arrow
(including the step from the founding census to the first size).
There is no ongoing mutation, recombination, migration, or selection.
Both treatments have the same duration and end at the same size.

Given frequency p, the next number of marked copies is Binomial(2N, p).
The updated expected gene diversity E[2p'(1−p')] is therefore the current
gene diversity times 1−1/(2N). Multiplication makes the order irrelevant
for this particular mean: both expectations are 0.33515625. Gene diversity
here is computed from the allele frequency, not observed diploid
heterozygosity.

By contrast, the full transition distribution depends on order.
For example, the six-diploid step first allows some populations to
wander toward an extreme before the two-diploid step samples four
copies. Loss and fixation are absorbing. Propagating the complete
binomial count distributions gives final monomorphism probabilities
14.31% and 19.00%, respectively. Across 4,096 simulated populations per
path, these were 15.09% and 18.38%. This is an example of order dependence,
not a general rule for the direction or size of the difference.

Original synthetic illustration. Background:
[Nei, Maruyama & Chakraborty (1975)](https://doi.org/10.1111/j.1558-5646.1975.tb00807.x)
on bottlenecks and variation, and
[Haller & Messer (2023)](https://doi.org/10.1086/723601) for SLiM.

Run from this directory with SLiM 4.0.1 on PATH and Python 3.14.5:

```sh
python3 -m unittest -v
python3 demo.py --check-sample
```

[Saved results](sample_output.json) include seeds and separate counts for
loss and fixation. On another SLiM build, use the theory tests rather than
expecting an identical random stream.
