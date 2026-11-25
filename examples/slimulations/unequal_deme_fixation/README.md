# Same neutral frequency, different fixation odds

Each Wright–Fisher replicate has two demes with 5 and 20 diploids. One
neutral marker starts at ten of 50 gene copies (20% pooled), all either in
the small deme or in the large one. Each destination deme obtains 20% of
its parental contribution from the other deme. These are equal *fractions*
of parentage, not equal numbers of migrants. There are no recurrent mutations
or selection. We census both demes, not a sample.

Conditional expected frequencies in the next generation are
`(0.8*p_small + 0.2*p_large, 0.2*p_small + 0.8*p_large)`. Their average
is unchanged in expectation. This bounded neutral martingale ends at one
if the marker fixes and zero if it is lost, so its starting value gives
fixation probability. It is 1/2 when initial frequencies are (1,0) and 1/8
when they are (0,1/4). The pooled frequency weights demes 1:4 and is not
the conserved quantity. Equal fractions make the two demes have equal
total reproductive value, so a copy in the small deme has more weight.

| Initial location | Fixed / 384 | Observed | Exact |
| --- | ---: | ---: | ---: |
| Small deme | 188 | 0.489583 | 0.500000 |
| Large deme | 55 | 0.143229 | 0.125000 |

All replicates absorbed within 400 reproduction steps. These probabilities
are model-specific, not a prediction when migration is specified as equal
numbers of migrants, or in the presence of selection.

## Reproduce
Tested using the native SLiM 4.0.1 binary and CPython 3.14.5 on macOS arm64.
No dependencies beyond those tools. Seeds 20261125 and 20261126.

```sh
python verify.py --test
python verify.py | diff -u result.txt -
```

References: [Whitlock, 2003, fixation in subdivided populations](https://doi.org/10.1093/genetics/164.2.767) and [Haller and Messer, SLiM 4](https://doi.org/10.1086/723601).
