"""Deterministic address reorganization and a retention counterexample."""

import argparse
import json
import math
from pathlib import Path

from .model import dot, factorized_step, read, step, unit

SAMPLE = Path(__file__).with_name('sample_output.json')


def rounded(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [rounded(x) for x in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def report():
    initial = [[1., 2.], [0., 0.], [1., 3.]]
    k, v, alpha, mask = [0., 1., 0.], [1., 0.], [1., 1., 1.], [.5, .5]
    baseline = step(initial, k, v, alpha, mask, .5, 0.)
    dual = step(initial, k, v, alpha, mask, .5, .5)
    factored = factorized_step(initial, k, v, alpha, mask, .5, .5)
    target = k
    before = [row[0] for row in baseline]
    after = [row[0] for row in dual]
    old_read = read(initial, [1., 0., 0.])
    baseline_read = read(baseline, [1., 0., 0.])
    dual_read = read(dual, [1., 0., 0.])
    sq = lambda a,b: math.fsum((x-y)**2 for x,y in zip(a,b))
    oblique = [[1., 0.], [0., 2.]]
    ov, om = [3., 4.], [.8, .2]
    vh = unit(ov)
    selector = unit([m*x for m,x in zip(om,vh)])
    tau = dot(ov, selector)
    correction = step(oblique, [1., 0.], ov, [1., 1.], om, 0., 1.)
    response_before = [dot(row,selector) for row in oblique]
    response_after = [dot(row,selector) for row in correction]
    return rounded({
        'example': 'Dual DeltaNet value-side reorganization', 'python': '3.14.5',
        'randomness': 'none; constructed projections and gates',
        'uniform_selector': {'initial_state': initial, 'key_only_state': baseline,
            'dual_state': dual, 'factorized_state': factored,
            'target_address': target, 'address_before': before, 'address_after': after,
            'address_squared_error_before': sq(before,target), 'address_squared_error_after': sq(after,target),
            'orthogonal_value_channel_unchanged': [row[1] for row in dual]},
        'retention_counterexample': {'desired_old_address_read': old_read, 'key_only_read': baseline_read,
            'dual_read': dual_read, 'key_only_squared_error': sq(baseline_read,old_read),
            'dual_squared_error': sq(dual_read,old_read)},
        'nonuniform_selector': {'normalized_value': vh, 'selector': selector, 'gamma': 1.,
            'target_address': [tau,0.], 'address_before': response_before, 'address_after': response_after,
            'residual_multiplier': 1.-dot(vh,selector),
            'address_squared_error_after': sq(response_after,[tau,0.])}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(report(), indent=2, allow_nan=False)+'\n'
    if args.check_sample and text != SAMPLE.read_text():
        parser.error('Output differs from checked sample')
    print(text, end='')


if __name__ == '__main__':
    main()
