"""Identical unphased genotypes need not determine Garud's H statistics."""
import argparse
from collections import Counter
from fractions import Fraction
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform

import allel
import numpy as np

SAMPLE = Path(__file__).with_name('sample_output.json')
NAMES = ('H1', 'H12', 'H123', 'H2_H1')


def validate(h):
    """Require complete binary arrays; correct phase remains a caller assumption."""
    a = np.asarray(h)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        raise ValueError('require (variants, haplotypes) with >=1 variant and >=2 haplotypes')
    if a.dtype.kind not in 'iu' or np.any((a != 0) & (a != 1)):
        raise ValueError('require integer 0/1 alleles; missing calls are not supported')
    return a.astype(np.int8, copy=True)


def unphased_genotypes(h):
    """Adjacent columns are the two chromosomes of one diploid."""
    a = validate(h)
    if a.shape[1] % 2:
        raise ValueError('diploid conversion requires an even number of haplotypes')
    return np.sort(a.reshape(a.shape[0], -1, 2), axis=2)


def phase_panel(pair_counts):
    """Copies of 000/111, 001/110, 010/101, 011/100, in that order."""
    counts = np.asarray(pair_counts)
    if counts.shape != (4,) or counts.dtype.kind not in 'iu':
        raise ValueError('require four nonnegative integer pair counts')
    if any(int(c) < 0 or int(c) > 10000 for c in counts) or sum(map(int, counts)) < 1:
        raise ValueError('require pair counts in [0, 10000] and at least one diploid')
    columns = []
    for first, count in zip(itertools.product((0,), (0, 1), (0, 1)), counts):
        second = tuple(1 - x for x in first)
        for _ in range(int(count)):
            columns.extend((first, second))
    return np.array(columns, dtype=np.int8).T


def pair_reference(h):
    """Exact ordered-pair enumeration, including draws of a chromosome twice.

    Independently merge the most frequent sequence labels, then count matching
    ordered draws. No tool frequency/counting routine is used.
    """
    a = validate(h)
    if a.shape[1] > 256:
        raise ValueError('exact pair oracle is limited to 256 chromosomes for this toy example')
    columns = [tuple(column) for column in a.T.tolist()]
    counts = Counter(columns)
    ranked = sorted(counts, key=lambda label: (-counts[label], label))
    rank = {label: i for i, label in enumerate(ranked)}
    draws = list(itertools.product(columns, repeat=2))

    def matching_after_merge(k):
        def group(label):
            return 0 if rank[label] < k else rank[label] - k + 1
        return Fraction(sum(group(a) == group(b) for a, b in draws), len(draws))

    h1 = matching_after_merge(1)
    h2 = Fraction(sum(a == b and a != ranked[0] for a, b in draws), len(draws))
    return dict(zip(NAMES, (h1, matching_after_merge(2), matching_after_merge(3), h2 / h1)))


def estimate(h):
    a = validate(h)
    exact = pair_reference(a)
    counts = Counter(''.join(map(str, col)) for col in a.T.tolist())
    return {
        'haplotype_counts': dict(sorted(counts.items())),
        'scikit_allel': dict(zip(NAMES, map(float, allel.garud_h(a)))),
        'exact_pair_reference': {k: str(v) for k, v in exact.items()},
    }


def fixtures():
    return {
        'two_haplotypes': phase_panel([20, 0, 0, 0]),
        'four_haplotypes': phase_panel([10, 10, 0, 0]),
        'eight_haplotypes': phase_panel([5, 5, 5, 5]),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def run():
    panels = fixtures()
    first = panels['two_haplotypes']
    same = all(np.array_equal(unphased_genotypes(first), unphased_genotypes(h))
               for h in panels.values())
    # The tool does not apply a finite-sample correction. This diagnostic is
    # a distinct-pair probability, not a correction to H12 or H2/H1.
    n = first.shape[1]
    return rounded({
        'versions': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ('scikit-allel', 'numpy')}},
        'randomness': 'none; deterministic constructed phase assignments',
        'diploids': 20,
        'variants': 3,
        'chromosomes': n,
        'same_unphased_genotypes': same,
        'unphased_to_haplotypes_reshape_equals_two_haplotype_panel': bool(np.array_equal(
            allel.GenotypeArray(unphased_genotypes(first)).to_haplotypes(), first)),
        'genotype_at_every_site_in_every_individual': [0, 1],
        'alt_counts_in_each_panel': {name: h.sum(axis=1).tolist() for name, h in panels.items()},
        'scenarios': {name: estimate(h) for name, h in panels.items()},
        'distinct_pair_haplotype_match_probability': {
            name: str(Fraction(sum(c * (c - 1) for c in Counter(map(tuple, h.T)).values()), n * (n - 1)))
            for name, h in panels.items()},
        'monomorphic_control': estimate(np.zeros((3, 40), dtype=np.int8)),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.check_sample and text != SAMPLE.read_text():
        raise SystemExit('sample output mismatch')
    print(text, end='')


if __name__ == '__main__':
    main()
