"""Expected down-projection is not one random smaller sample."""
import argparse
from fractions import Fraction
from math import comb
import importlib.metadata
import json
from pathlib import Path
import platform

import dadi
import numpy as np

SAMPLE = Path(__file__).with_name('sample_output.json')


def validate(counts, n, m):
    if any(type(x) is not int for x in (n, m)) or not 2 <= m <= n <= 100:
        raise ValueError('require integer chromosome sizes 2 <= m <= n <= 100')
    if not isinstance(counts, (list, tuple)) or len(counts) != n+1:
        raise ValueError('one count for each derived allele count 0..n required')
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('site counts must be nonnegative integers')


def exact_projection(counts, n, m):
    """Hypergeometric probabilities with exact integer combinations."""
    validate(counts, n, m)
    return [sum((Fraction(c * comb(k, j) * comb(n-k, m-j), comb(n, m))
                 for k, c in enumerate(counts) if j <= k and m-j <= n-k), Fraction())
            for j in range(m+1)]


def project(counts, n=20, m=4):
    validate(counts, n, m)
    # Keep invariant mass explicitly for conservation checks. Mask it only
    # when reporting segregating sites / diversity, as in ordinary SFS fitting.
    raw = dadi.Spectrum(counts, mask_corners=False)
    projected = raw.project([m])
    expected = np.array([float(x) for x in exact_projection(counts, n, m)])
    np.testing.assert_allclose(projected.data, expected, rtol=1e-12, atol=1e-12)
    return raw, projected


def statistics(fs):
    masked = dadi.Spectrum(fs.data, mask_corners=True)
    return {'expected_segregating_sites': round(float(masked.S()), 12),
            'pairwise_difference_total': round(float(masked.pi()), 12)}


def panel(k, sites=100, n=20):
    counts = [0]*(n+1); counts[k] = sites
    return counts


def run():
    counts = panel(1, 80)
    counts[10] = 20
    fixtures = {'mixed': counts, 'singletons_only': panel(1),
                'intermediate_only': panel(10)}
    out = {'versions': {'python': platform.python_version(),
                       **{p: importlib.metadata.version(p) for p in ('dadi','numpy')}},
           'seed': None, 'chromosomes_before':20, 'chromosomes_after':4, 'panels':{}}
    for name, data in fixtures.items():
        original, projected = project(data)
        out['panels'][name] = {'input_sfs': data,
            'projected_expected_sfs_including_corners': [round(float(v),12) for v in projected.data],
            'before': statistics(original), 'after':statistics(projected)}
    # Deliberate different operation: ascertain only projected polymorphic sites
    # and inflate their expected spectrum back to 100 retained variants.
    p = project(counts)[1]
    conditional = dadi.Spectrum(p.data)
    conditional *= 100 / conditional.S()
    out['condition_on_polymorphism_then_renormalize_to_100_sites'] = statistics(conditional)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args();result=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if result != SAMPLE.read_text(): raise SystemExit('sample output mismatch')
        print('sample output matches')
    else: print(result,end='')

if __name__ == '__main__': main()
