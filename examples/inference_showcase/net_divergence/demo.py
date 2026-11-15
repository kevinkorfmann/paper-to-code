"""Cross-population pair differences are not a split-time estimate."""
import argparse
from fractions import Fraction
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import platform
import allel
import numpy as np

HERE = Path(__file__).resolve().parent


def validate(a, b):
    arrays = []
    for x in (a, b):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
            raise ValueError('require nonempty sites by two aligned alleles')
        if x.dtype.kind not in 'iu' or np.any(x < 0) or np.any(x > 1000):
            raise ValueError('require integer allele counts in [0,1000]')
        if np.any(x.sum(axis=1) < 2) or len(x) > 1000:
            raise ValueError('require at least two chromosomes per site and at most 1000 sites')
        arrays.append(x.astype(np.int64, copy=True))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('aligned site counts required')
    return arrays


def estimate(a, b):
    a, b = validate(a, b)
    between = allel.mean_pairwise_difference_between(a, b)
    within_a = allel.mean_pairwise_difference(a)
    within_b = allel.mean_pairwise_difference(b)
    return dict(dxy=between, pi_a=within_a, pi_b=within_b,
                net=between - (within_a + within_b)/2)


def count_oracle(a, b):
    a, b = validate(a, b)
    out = {k: [] for k in ('dxy', 'pi_a', 'pi_b', 'net')}
    for x, y in zip(a.tolist(), b.tolist()):
        n, m = sum(x), sum(y)
        d = Fraction(x[0]*y[1] + x[1]*y[0], n*m)
        u = Fraction(2*x[0]*x[1], n*(n-1))
        v = Fraction(2*y[0]*y[1], m*(m-1))
        for k, value in zip(out, (d, u, v, d-(u+v)/2)):
            out[k].append(value)
    return out


def pair_oracle(x, y):
    """Independent enumeration from explicit chromosomes, for small test panels."""
    x, y = tuple(x), tuple(y)
    if min(len(x), len(y)) < 2 or max(len(x), len(y)) > 30:
        raise ValueError('enumeration requires 2..30 chromosomes')
    if any(v not in (0, 1) for v in x+y):
        raise ValueError('binary chromosomes required')
    d = Fraction(sum(a != b for a, b in product(x, y)), len(x)*len(y))
    u = Fraction(sum(a != b for a, b in combinations(x, 2)), comb(len(x), 2))
    v = Fraction(sum(a != b for a, b in combinations(y, 2)), comb(len(y), 2))
    return dict(dxy=d, pi_a=u, pi_b=v, net=d-(u+v)/2)


def binomial_expectation(n, m, p, q):
    """Exact independent chromosome-sampling expectation; no random draws."""
    if not (2 <= n <= 20 and 2 <= m <= 20 and 0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('small valid sample sizes and frequencies required')
    p, q = Fraction(p), Fraction(q)
    result = {k: Fraction(0) for k in ('dxy', 'pi_a', 'pi_b', 'net')}
    for i, j in product(range(n+1), range(m+1)):
        w = comb(n, i)*p**i*(1-p)**(n-i)*comb(m, j)*q**j*(1-q)**(m-j)
        values = count_oracle([[n-i, i]], [[m-j, j]])
        for k in result:
            result[k] += w*values[k][0]
    return result


def summary():
    panels = {'identical_counts': ([[5,5]], [[5,5]]),
              'fixed_difference': ([[10,0]], [[0,10]]),
              'shared_invariant': ([[10,0]], [[10,0]]),
              'unequal_counts': ([[8,2]], [[2,8]])}
    results = {name: {k: round(float(v[0]), 9) for k,v in estimate(a,b).items()}
               for name,(a,b) in panels.items()}
    expectations = {}
    for name,p,q in [('same_frequency', Fraction(1,2), Fraction(1,2)),
                     ('different_frequency', Fraction(1,5), Fraction(4,5))]:
        expectations[name] = {k: str(v) for k,v in binomial_expectation(10,10,p,q).items()}
    return {'python':platform.python_version(), 'scikit_allel':allel.__version__,
            'numpy':np.__version__, 'randomness':'none; exact constructed counts',
            'per_site_panels':results, 'independent_binomial_expectations':expectations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    actual = summary()
    if args.check_sample:
        if actual != json.loads((HERE/'sample_summary.json').read_text()):
            raise SystemExit('sample mismatch')
        print('sample matches')
    else:
        print(json.dumps(actual, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
