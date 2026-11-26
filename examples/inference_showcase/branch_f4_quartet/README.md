# A minority quartet split can determine the sign of branch f4

Two synthetic, dated quartets share four contemporaneous haploid samples:
one each from A, B, C, and D. AC|BD covers 70 bp with internal path length
2 generations; AD|BC covers 30 bp with internal path length 10 generations.
tskit's branch f4(A,B;C,D) summarizes an ordered allele-sharing contrast
on the supplied genealogy, in generations. On these singleton quartets
it is the signed internal length, averaged by genomic span:
(70 × 2 − 30 × 10)/100 = −1.6. Matching the internal lengths at 2, without
changing either split or its span, instead gives +0.8.

There are no mutations: site-mode f4 is zero on these particular inputs.
Under a common mutation rate and the infinite-sites Poisson mutation
model, expected site f4 equals the mutation rate times branch f4. Neither
a realized site value nor the branch value directly counts trees or
estimates an admixture fraction. The fixture neither reconstructs
genealogies nor controls empirical inference error.

## Reproduce

CPython 3.12.11, tskit 1.0.4, NumPy 2.5.5. Deterministic; no seed.

```sh
python -m pip install tskit==1.0.4 numpy==2.5.5
python demo.py --test
python demo.py > actual.txt
diff -u expected.txt actual.txt
```

Six tests include a distance-based oracle across 15 length/order
combinations, local windows, and a zero matching-split control.

Sources: [tskit f4 API](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.f4)
and [Ralph, Thornton & Kelleher (2020)](https://doi.org/10.1534/genetics.120.303253).
