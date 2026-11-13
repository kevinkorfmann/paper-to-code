"""Deterministic constructed inputs; run as a module from the repository root."""
import argparse
import json
from pathlib import Path
from .model import compact_step, project, read, update


def results():
    p = [[1.0], [0.0]]
    k1, k2 = [0.6, 0.8], [0.8, -0.6]
    writes = {}
    for name, kwargs in [('uncompensated', {'compensate': False}), ('compensated', {}), ('cap_2', {'cap': 2.0}), ('floor_half', {'epsilon': 0.5})]:
        c, info = compact_step([[0.0]], p, k1, [1.0], 1.0, 1.0, **kwargs)
        writes[name] = {**info, 'same_key_read': read(c, project(p, k1))[0]}
    dense, compact = [[0.0, 0.0]], [[0.0]]
    for k, v in [(k1, [1.0]), (k2, [-1.0])]:
        dense = update(dense, k, v, 1.0, 1.0)
        compact, _ = compact_step(compact, p, k, v, 1.0, 1.0)
    c0, _ = compact_step([[0.0]], p, [0.0, 1.0], [1.0], 1.0, 1.0)
    return {'environment': 'CPython 3.14.5; standard library; deterministic, no seed',
            'single_write': writes,
            'orthogonal_key_collision': {'dense_reads': [read(dense, k)[0] for k in (k1, k2)],
                                        'compact_reads': [read(compact, project(p, k))[0] for k in (k1, k2)]},
            'discarded_key_read': read(c0, project(p, [0.0, 1.0]))[0]}


def output():
    def rounded(x):
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(results()), indent=2, sort_keys=True) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = output()
    if args.check_sample:
        if text != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample mismatch')
        print('sample output verified')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
