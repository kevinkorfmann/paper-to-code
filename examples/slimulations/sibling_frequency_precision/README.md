# Full siblings inflate allele-frequency sampling variance

A neutral biallelic locus has population frequency 0.5. Each replicate draws
12 independent outbred diploids, with the two copies in each drawn independently.
Ten founders supply the unrelated sample. The other two produce ten full siblings
through ordinary Mendelian crosses in SLiM. There is no ongoing mutation,
selection or recombination. This is a one-locus sampling comparison, not an
estimate of population effective size or a general model of a biobank.

Both sample frequencies have expectation 0.5. For the unrelated sample the
variance is 0.25/20 = 0.0125. Conditional on the parents, the ten siblings are
independent. Their conditional mean is the frequency among the four parental
copies, with variance 0.25/4 = 0.0625 across families. The mean conditional
sampling variance is 0.00625, giving 0.06875 in total. A naive binomial model
treating the 20 observed copies as independent misses that family term.

Original simulation. [Boehnke & Lange (2001) on allele frequencies from
sibships](https://pubmed.ncbi.nlm.nih.gov/11255240/). Run in this directory
with CPython 3.14.5 and SLiM 4.0.1 (seed 20261127, 2,048 replicates):

```sh
python -m unittest -v
python demo.py --check-sample
```

See [checked results](sample_output.json). A build with a different RNG stream
can be assessed using the theory checks rather than exact byte-for-byte replay.
