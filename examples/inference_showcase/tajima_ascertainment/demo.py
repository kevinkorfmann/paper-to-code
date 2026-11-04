"""Tajima's D is a spectrum contrast, not a uniquely identified demographic cause."""
import argparse
from fractions import Fraction as F
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import allel
import numpy as np

SAMPLE = Path(__file__).with_name('sample_output.json')


def validate(ac):
    a = np.asarray(ac)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a):
        raise ValueError('require a nonempty (sites, 2) biallelic count matrix')
    if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a > 1000000):
        raise ValueError('require integer allele counts in [0, 1000000]')
    a = a.astype(np.int64)
    totals = a.sum(axis=1)
    if totals[0] < 4 or not np.all(totals == totals[0]):
        raise ValueError('require at least four chromosomes and equal called counts at every site')
    return a


def reference(ac):
    """Independent rational arithmetic up to the final square root."""
    a = validate(ac).tolist()
    n = sum(a[0])
    s = sum(x > 0 and y > 0 for x, y in a)
    pi = sum((F(2*x*y, n*(n-1)) for x, y in a), F(0))
    a1 = sum((F(1, i) for i in range(1, n)), F(0))
    a2 = sum((F(1, i*i) for i in range(1, n)), F(0))
    b1 = F(n+1, 3*(n-1))
    b2 = F(2*(n*n+n+3), 9*n*(n-1))
    e1 = (b1-1/a1)/a1
    e2 = (b2-F(n+2, n)/a1+a2/a1**2)/(a1**2+a2)
    theta = F(s)/a1
    variance = e1*s + e2*s*(s-1)
    d = float(pi-theta)/math.sqrt(float(variance)) if s >= 3 else None
    return {'segregating_sites': s, 'pairwise_differences': float(pi),
            'watterson_theta': float(theta), 'reference_D': d}


def estimate(ac):
    a = validate(ac)
    result = reference(a)
    d = float(allel.tajima_d(a, min_sites=3))
    result['scikit_allel_D'] = d if math.isfinite(d) else None
    return result


def fixtures():
    # Completely constructed spectra, not draws from a population model.
    mixed = [[19, 1]]*80 + [[10, 10]]*20
    return {
        'all_sites': mixed,
        'minor_count_at_least_2': [row for row in mixed if min(row) >= 2],
        'singletons_only': [[19, 1]]*100,
        'intermediate_only': [[10, 10]]*100,
        'alleles_reversed': [row[::-1] for row in mixed],
        'monomorphic_padding': mixed + [[20, 0]]*100 + [[0, 20]]*100,
        'two_segregating_sites': [[19, 1], [10, 10], [20, 0]],
        'no_segregating_sites': [[20, 0]]*10,
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def run():
    return rounded({
        'versions': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ('scikit-allel', 'numpy')}},
        'randomness': 'none; deterministic constructed allele counts',
        'chromosomes': 20,
        'scenarios': {name: estimate(ac) for name, ac in fixtures().items()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.check_sample and text != SAMPLE.read_text():
        raise SystemExit('sample output mismatch')
    print(text, end='')


if __name__ == '__main__':
    main()
