"""Joint folding removes ancestral polarity, not population allele alignment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import dadi
import numpy as np

SAMPLE = Path(__file__).with_name('sample_output.json')


def validate(counts, sizes):
    if len(sizes) != 2 or any(type(n) is not int or not 2 <= n <= 100 for n in sizes):
        raise ValueError('require two integer sample sizes in 2..100')
    a = np.asarray(counts)
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in 'iu':
        raise ValueError('counts must be a nonempty integer matrix with two columns')
    if len(a) == 0 or np.any(a < 0) or np.any(a > np.array(sizes)):
        raise ValueError('counts outside complete chromosome samples')
    return a.astype(int)


def spectra(counts, sizes=(10, 10)):
    a = validate(counts, sizes)
    raw = np.zeros(tuple(n + 1 for n in sizes))
    for row in a:
        raw[tuple(row)] += 1
    unfolded = dadi.Spectrum(raw)
    # Incorrect substitute for dadi joint folding: each population chooses
    # its own minor allele. This is a deliberately different summary.
    separate = np.zeros(tuple(n // 2 + 1 for n in sizes), dtype=int)
    for row in a:
        separate[tuple(min(int(k), n-int(k)) for k, n in zip(row, sizes))] += 1
    return unfolded, unfolded.fold(), separate


def oracle(counts, sizes=(10, 10)):
    """Explicit per-site orientation choices; ties split equally, corners omitted."""
    a = validate(counts, sizes)
    out = np.zeros(tuple(n + 1 for n in sizes))
    for row in a:
        forward = tuple(int(k) for k in row)
        reverse = tuple(n-k for n,k in zip(sizes,forward))
        if sum(forward) in (0, sum(sizes)):
            continue
        if sum(forward) < sum(reverse):
            out[forward] += 1
        elif sum(forward) > sum(reverse):
            out[reverse] += 1
        else:
            out[forward] += 0.5
            out[reverse] += 0.5
    return out


def cells(s):
    a = np.ma.filled(s, 0)
    return [{'counts': list(index), 'sites': float(a[index])}
            for index in np.ndindex(a.shape) if a[index] != 0]


def fixtures():
    return {'same_minor_allele': [[1,1]] * 20,
            'opposite_minor_alleles': [[1,9]] * 20,
            'fixed_differences': [[0,10]] * 20,
            'shared_absence': [[0,0]] * 20}


def run():
    output = {'versions': {p:importlib.metadata.version(p) for p in ('dadi','numpy')},
              'sample_sizes': [10,10], 'seed': None, 'panels': {}}
    for name, panel in fixtures().items():
        unfolded, folded, separate = spectra(panel)
        np.testing.assert_array_equal(folded.filled(0), oracle(panel))
        output['panels'][name] = {'dadi_joint_folded':cells(folded),
                                 'independent_minor_counts':cells(separate),
                                 'retained_segregating_sites':float(folded.S())}
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args(); result=json.dumps(run(),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if result != SAMPLE.read_text(): raise SystemExit('sample output mismatch')
        print('sample output matches')
    else: print(result,end='')

if __name__ == '__main__': main()
