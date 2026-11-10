"""Extended haplotype homozygosity and distance units, using scikit-allel."""
import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import platform

import allel
import numpy as np

HERE = Path(__file__).resolve().parent


def validate_haplotypes(h):
    x = np.asarray(h)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2:
        raise ValueError('require markers by chromosomes, at least one marker and two chromosomes')
    if x.dtype.kind not in 'iu' or np.any((x != 0) & (x != 1)):
        raise ValueError('require complete biallelic integer haplotypes, with known phase')
    if x.size > 100000:
        raise ValueError('educational demo limited to 100000 entries')
    return np.array(x, dtype=np.int8, copy=True)


def estimate(h):
    """Fraction of distinct chromosome pairs matching from first through each marker."""
    return np.asarray(allel.ehh_decay(validate_haplotypes(h), truncate=False))


def exact_pair_oracle(h):
    x = validate_haplotypes(h)
    n = x.shape[1]
    return [Fraction(sum(all(x[k, i] == x[k, j] for k in range(end+1))
                         for i, j in itertools.combinations(range(n), 2)), n*(n-1)//2)
            for end in range(x.shape[0])]


def prefix_count_oracle(h):
    """Independent frequency-count formulation, without pair enumeration."""
    x = validate_haplotypes(h)
    n = x.shape[1]
    return [Fraction(sum(c*(c-1) for c in Counter(map(tuple, x[:end+1].T)).values()), n*(n-1))
            for end in range(x.shape[0])]


def finite_area(ehh, positions):
    """Trapezoidal area over supplied markers only; no extrapolation or gap correction."""
    y = np.asarray(ehh, dtype=float)
    x = np.asarray(positions)
    if y.ndim != 1 or len(y) < 1 or x.shape != y.shape or x.dtype.kind not in 'iuf':
        raise ValueError('require equal nonempty one-dimensional curves and positions')
    x = x.astype(float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any((y<0)|(y>1)):
        raise ValueError('finite positions and EHH in [0,1] required')
    if np.any(np.diff(x) <= 0):
        raise ValueError('positions must be strictly increasing')
    area = float(np.trapezoid(y, x))
    if not np.isfinite(area):
        raise ValueError('area must be representable')
    return area


def panel():
    # Four markers, eight chromosomes, all carrying the same core allele at marker 0.
    # Prefix partitions: [8], [4,4], [2,2,2,2], [1,1,1,1,1,1,1,1].
    return np.array([[0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,1,1,1],
                     [0,0,1,1,0,0,1,1],
                     [0,1,0,1,0,1,0,1]], dtype=np.int8)


def run():
    h = panel()
    y = estimate(h)
    cases = {}
    for name, pos, unit in [('compact_bp', [0,100,200,300], 'bp'),
                             ('stretched_bp', [0,1000,2000,3000], 'bp'),
                             ('compact_kb', [0,.1,.2,.3], 'kb'),
                             ('marker_index', [0,1,2,3], 'marker intervals')]:
        cases[name] = {'positions': pos, 'unit': unit, 'ehh': y.tolist(),
                       'finite_trapezoid_area': finite_area(y, pos)}
    # Removing a distinguishing marker increases sharing at the remaining endpoints.
    sparse = h[[0,1,3]]
    return {'versions': {'python': platform.python_version(), 'scikit_allel': allel.__version__,
                         'numpy': np.__version__}, 'randomness': 'none',
            'haplotypes_markers_by_chromosomes': h.tolist(),
            'exact_ehh': [str(v) for v in exact_pair_oracle(h)], 'cases': cases,
            'dropped_marker_control': {'retained_rows': [0,1,3], 'ehh': estimate(sparse).tolist(),
                                       'exact_ehh': [str(v) for v in exact_pair_oracle(sparse)]},
            'truncated_window_control': {'ehh': estimate(h[:3]).tolist(),
                                          'terminal_ehh': float(estimate(h[:3])[-1]),
                                          'finite_area_bp': finite_area(estimate(h[:3]), [0,100,200])},
            'copy_count_control': {'duplicated_each_chromosome_ehh': estimate(np.repeat(h,2,axis=1)).tolist()}}


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
