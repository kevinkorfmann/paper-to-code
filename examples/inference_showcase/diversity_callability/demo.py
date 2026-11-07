"""Nucleotide diversity needs callable invariant bases, not just variant records."""
import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import platform
import allel
import numpy as np

SAMPLE = Path(__file__).with_name('sample_output.json')


def estimate(pos, ac, accessible):
    """One inclusive whole-contig window; complete, equal-size allele samples only."""
    p, a, mask = np.asarray(pos), np.asarray(ac), np.asarray(accessible)
    if mask.ndim != 1 or mask.size == 0 or mask.dtype.kind != 'b':
        raise ValueError('accessible must be a nonempty boolean base mask')
    if p.ndim != 1 or p.size == 0 or p.dtype.kind not in 'iu':
        raise ValueError('positions must be a nonempty integer vector')
    if np.any(p < 1) or np.any(p > mask.size) or np.any(np.diff(p.astype(np.int64)) <= 0):
        raise ValueError('positions must be unique, increasing, and within contig')
    if a.ndim != 2 or a.shape[0] != p.size or a.shape[1] < 2:
        raise ValueError('allele counts must align with positions and have >=2 columns')
    if a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError('allele counts must be nonnegative integers')
    # Bound before conversion/summing to avoid integer overflow on malformed input.
    if np.any(a > 1000000):
        raise ValueError('educational input limit is one million per allele')
    totals = a.astype(np.int64).sum(axis=1)
    if p.size and (np.any(totals < 2) or np.any(totals != totals[0])):
        raise ValueError('require >=2 complete chromosomes with constant sample size')
    pi, windows, bases, records = allel.windowed_diversity(
        p.astype(np.int64), a.astype(np.int64), windows=[[1, mask.size]],
        is_accessible=mask)
    value = float(pi[0])
    return {'pi': round(value, 12) if np.isfinite(value) else None,
            'callable_bases': int(bases[0]), 'retained_records': int(records[0]),
            'window_1based_inclusive': windows[0].tolist()}


def pair_oracle(pos, ac, accessible):
    """Tiny exact oracle by explicit chromosome-pair enumeration, no tool formulas."""
    if sum(accessible) == 0:
        return None
    total = Fraction(0)
    for position, counts in zip(pos, ac):
        if not accessible[position - 1]:
            continue
        if not 2 <= sum(counts) <= 32:
            raise ValueError('oracle supports only 2..32 chromosomes')
        chromosomes = [allele for allele, count in enumerate(counts) for _ in range(count)]
        pairs = list(combinations(chromosomes, 2))
        total += Fraction(sum(x != y for x, y in pairs), len(pairs))
    return total / sum(accessible)


def run():
    pos = [10, 15, 20]
    ac = [[3, 1], [2, 2], [1, 3]]
    full = np.ones(100, dtype=bool)
    twenty = np.arange(100) < 20
    snps = np.zeros(100, dtype=bool)
    snps[np.array(pos) - 1] = True
    dropped = twenty.copy()
    dropped[14] = False
    panels = {'all_100_bases': full, 'only_20_callable_bases': twenty,
              'incorrect_variant_only_mask': snps,
              'exclude_middle_variant_and_its_base': dropped,
              'no_callable_bases': np.zeros(100, dtype=bool)}
    results = {}
    for name, mask in panels.items():
        result = estimate(pos, ac, mask)
        exact = pair_oracle(pos, ac, mask.tolist())
        result['exact_pair_oracle'] = str(exact) if exact is not None else None
        results[name] = result
    return {'versions': {'python': platform.python_version(), 'scikit_allel': allel.__version__,
                         'numpy': np.__version__},
            'seed': None, 'positions_1based': pos, 'allele_counts': ac,
            'cases': results,
            'default_variant_span_pi': round(float(allel.sequence_diversity(pos, ac)), 12),
            'default_variant_span_bases': 11}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    output = json.dumps(run(), indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.check_sample:
        if output != SAMPLE.read_text():
            raise SystemExit('sample output differs')
        print('sample output matches')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
