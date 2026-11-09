"""Rogers-Huff genotype correlation under population pooling."""
import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import platform

import allel
import numpy as np

HERE = Path(__file__).resolve().parent


def validate_table(table):
    x = np.asarray(table)
    if x.shape != (3, 3) or x.dtype.kind not in 'iu':
        raise ValueError('require a 3 by 3 integer genotype-count table')
    if np.any(x < 0) or np.any(x > 1000) or int(x.sum()) < 2:
        raise ValueError('require counts 0..1000 and at least two diploids')
    return np.array(x, dtype=np.int64, copy=True)


def genotypes(table):
    """Rows index locus X dosage; columns index locus Y dosage."""
    x = validate_table(table)
    pairs = [(i, j) for i, j in itertools.product(range(3), repeat=2)
             for _ in range(int(x[i, j]))]
    return np.asarray(pairs, dtype=np.int8).T.copy()


def estimate(table):
    x = validate_table(table)
    gn = genotypes(x)
    r = float(allel.rogers_huff_r(gn))
    return {'n_diploids': int(x.sum()), 'joint_dosage_counts': x.tolist(),
            'alt_frequencies': (gn.mean(axis=1)/2).tolist(),
            'r': r if math.isfinite(r) else None,
            'r_squared': r*r if math.isfinite(r) else None}


def exact_pair_oracle(table):
    """Use weighted differences between distinct genotype classes, not centered sums."""
    x = validate_table(table)
    c = a = b = 0
    cells = list(itertools.product(range(3), repeat=2))
    for (i,j), (k,l) in itertools.combinations(cells, 2):
        w = int(x[i,j])*int(x[k,l])
        c += w*(i-k)*(j-l)
        a += w*(i-k)**2
        b += w*(j-l)**2
    if not a or not b:
        return {'sign': None, 'r_squared': None}
    return {'sign': (c>0)-(c<0), 'r_squared': Fraction(c*c, a*b)}


def fixtures():
    # Exact HWE marginals (16, 8, 1)/25 at p=.2, reversed at p=.8.
    lo = np.array([16,8,1], dtype=np.int64)
    hi = lo[::-1]
    a, b = np.outer(lo,lo), np.outer(hi,hi)
    return {'group_A': a, 'group_B': b, 'pooled': a+b,
            'pooled_reverse_Y': (a+b)[:,::-1],
            'no_frequency_difference': 2*a,
            'constant_X': [[10,20,10],[0,0,0],[0,0,0]],
            'perfect_association': np.diag([4,2,4])}


def run():
    result = {}
    for name, table in fixtures().items():
        o = exact_pair_oracle(table)
        result[name] = {**estimate(table), 'exact_sign': o['sign'],
                       'exact_r_squared': str(o['r_squared']) if o['r_squared'] is not None else None}
    # Law of total covariance for equally weighted population labels.
    within_var, between_var = Fraction(8,25), Fraction(9,25)
    return {'versions': {'python': platform.python_version(), 'scikit_allel': allel.__version__,
                         'numpy': np.__version__},
            'randomness': 'none', 'cases': result,
            'population_moments': {'within_dosage_covariance': '0',
                                   'within_dosage_variance': str(within_var),
                                   'between_group_dosage_mean_covariance': str(between_var),
                                   'pooled_dosage_variance': str(within_var+between_var),
                                   'pooled_r': str(between_var/(within_var+between_var)),
                                   'pooled_r_squared': str((between_var/(within_var+between_var))**2)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    result = run()
    if args.check_sample:
        if result != json.loads((HERE/'sample_output.json').read_text()):
            raise SystemExit('sample output differs')
        print('sample output matches')
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
