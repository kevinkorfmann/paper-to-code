"""Three-population f3: target drift and a finite-sample correction."""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path
import platform
import allel
import numpy as np

HERE = Path(__file__).resolve().parent


def validate(counts):
    x = np.asarray(counts)
    if x.ndim != 3 or x.shape[1:] != (3, 2) or len(x) == 0:
        raise ValueError('require nonempty (sites, populations C/A/B, alleles REF/ALT)')
    if x.dtype.kind not in 'iu' or np.any(x < 0) or np.any(x > 1_000_000):
        raise ValueError('counts must be integers from zero to one million')
    x = np.array(x, dtype=np.int64, copy=True)
    if np.any(x.sum(axis=2) < 2):
        raise ValueError('require at least two chromosomes per population at every site')
    return x


def estimate(counts):
    x = validate(counts)
    t, h = allel.patterson_f3(x[:, 0], x[:, 1], x[:, 2])
    c, a, b = (x[:, i, 1] / x[:, i].sum(axis=1) for i in range(3))
    denominator = float(h.sum())
    return {'sites': len(x), 'f3': float(t.mean()),
            'naive_product_mean': float(((c-a)*(c-b)).mean()),
            'target_heterozygosity_mean': float(h.mean()),
            'f3_star': float(t.sum()/denominator) if denominator > 0 else None}


def ordered_pair_oracle(site):
    """Independent unbiased U-statistic: E[(C_i-A_j)(C_k-B_l)], i != k."""
    x = validate([site])[0]
    if np.any(x.sum(axis=1) > 8):
        raise ValueError('enumeration oracle is limited to eight chromosomes per population')
    vectors = [[0]*int(row[0])+[1]*int(row[1]) for row in x]
    c, a, b = vectors
    total = sum((ci-aj)*(ck-bl) for i,ci in enumerate(c) for k,ck in enumerate(c)
                if i != k for aj in a for bl in b)
    return Fraction(total, len(c)*(len(c)-1)*len(a)*len(b))


def fixture(target_alt):
    """Two sites with identical sources, using 20 chromosomes per population."""
    return [[[20-c,c],[16,4],[4,16]] for c in target_alt]


def run():
    cases = {name: {'counts_C_A_B': fixture(k), **estimate(fixture(k))}
             for name,k in [('intermediate',[10,10]), ('moderate_dispersion',[4,16]),
                            ('fixed_endpoints',[0,20])]}
    # Exact population-level calculation, not a fitted demographic history.
    p, a, b = Fraction(1,2), Fraction(1,5), Fraction(4,5)
    theory = {name: {'target_variance': str(d*d),
                    'population_f3': str((p-a)*(p-b)+d*d)}
              for name,d in [('intermediate',Fraction(0)),
                             ('moderate_dispersion',Fraction(3,10)),
                             ('fixed_endpoints',Fraction(1,2))]}
    return {'versions': {'python': platform.python_version(), 'scikit_allel': allel.__version__,
                         'numpy': np.__version__}, 'seed': None, 'cases': cases,
            'exact_population_identity': theory}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    output=json.dumps(run(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.check_sample:
        if output != (HERE/'sample_output.json').read_text():
            raise SystemExit('sample output differs')
        print('sample output matches')
    else: print(output,end='')

if __name__ == '__main__': main()
