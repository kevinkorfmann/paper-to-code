"""Run: python -m examples.decoupled_compensation.demo [--check-sample]."""
import argparse
import json
from pathlib import Path
from .core import estimate, dense


def results():
    keys, values, labels = [0., 0.], [0., 2.], [0, 0]
    cancellation = [estimate(0., keys, values, labels, s).output for s in [(), (0,), (0, 1)]]
    # Cluster sizes matter: two zero-valued tokens versus one value-nine token.
    multiplicity = estimate(0., [0., 0., 0.], [0., 0., 9.], [0, 0, 1]).output
    # Equal centroid keys conceal a covariance between keys and values.
    spread = [estimate(1., [-1., 1.], [-1., 1.], [0, 0], s).output
              for s in [(), (0,), (0, 1)]]
    return {'environment': 'CPython 3.14.5; standard library; deterministic; no seed',
            'cancellation': {'dense': dense(0., keys, values).output,
                             'exact_token_counts': [0, 1, 2],
                             'outputs': cancellation,
                             'absolute_errors': [abs(x-1.) for x in cancellation]},
            'cluster_multiplicity': {'correct': multiplicity, 'equal_cluster_average_wrong': 4.5},
            'key_value_spread': {'exact_token_counts': [0, 1, 2],
                                 'outputs': [round(x, 9) for x in spread]},
            'scope': 'CENTROID only; supplied scalar inputs/clusters; no calibration, FULL corrections, GPU or model claims'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    content = json.dumps(results(), indent=2, sort_keys=True) + '\n'
    if args.check_sample:
        assert content == Path(__file__).with_name('sample_output.json').read_text()
        print('Sample output matches.')
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
