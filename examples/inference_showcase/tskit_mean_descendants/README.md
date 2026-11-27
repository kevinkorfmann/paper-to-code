# tskit: where a node's mean descendant count is defined

A focal ancestral node has two group A descendants on [0, 20), one
group B descendant on [20, 50), and no sampled descendants on [50, 100).
All three samples form one tree at every position. There are no sites
or mutations: this statistic uses the edges of the supplied genealogy.

[`TreeSequence.mean_descendants`](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.mean_descendants)
reports mean **counts** conditional on the focal node being ancestral to
at least one sample (here the groups cover all samples). Its denominator
is 50, giving (40/50, 30/50) = (0.80, 0.60). Including the inactive
half of the genome as zeros gives (0.40, 0.30), a different estimand.
These are not proportions of ancestry and need not sum to one. On
inferred trees they inherit errors in topology and breakpoints, and
counts depend on which and how many samples are represented.

This original fixture is deterministic, with no random seed. Tested
with CPython 3.12.11, tskit 1.0.4 and NumPy 2.5.5:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python example.py
```

Five tests check local descendant sets, both denominators, group order,
and that the branch-based calculations require no sites. See [actual
output](output.txt).
