"""Deterministic two-source examples. Run as a module from the repository root."""
import argparse
import json
import math
from pathlib import Path
from .model import advance, diagnostics, logits_from_probabilities, probabilities


def _round(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [_round(x) for x in value]
    if isinstance(value, dict):
        return {k: _round(v) for k,v in value.items()}
    return value


def result():
    start = logits_from_probabilities([0.5, 0.5])
    scores = [0.0, math.log(9.0)]
    half = math.log(2.0)
    u = advance(start, scores, half)
    fixed = []
    for n in [0, 1, 2, 4, 8]:
        state = advance(start, scores, n*half)
        info = diagnostics(state, scores)
        fixed.append({"half_lives": n, "probability_source_1": info['probabilities'][1],
                      "kl_to_target": info['kl_to_target']})
    ramp = advance(start, [0.,0.], 1., slope=[0.,-1.])
    balance = diagnostics(ramp, [0.,-1.], slope=[0.,-1.])
    late = advance(start, [0.,0.], 5., slope=[0.,-1.])
    # Explicit Euler in probability coordinates is a deliberately bad control.
    vel = diagnostics(start, scores)['probability_derivative']
    euler = [0.5+2*x for x in vel]
    return _round({
        "environment": "CPython 3.14.5; standard library; deterministic",
        "scope": "One unmasked row, supplied constant or affine scores; no training",
        "half_life": half,
        "constant_scores": scores,
        "target_probabilities": probabilities(scores),
        "exact_half_life_probabilities": probabilities(u),
        "arithmetic_probability_average": [0.3,0.7],
        "fixed_score_relaxation": fixed,
        "ramp_scores": "[0,-t], temperature=mobility=1",
        "ramp_at_t1": balance,
        "initial_free_energy": diagnostics(start,[0.,0.])['free_energy'],
        "ramp_at_t5": {"probabilities": probabilities(late),
                       "instantaneous_probabilities": probabilities([0.,-5.]),
                       "log_odds_lag": (late[1]-late[0])-(-5.)},
        "probability_euler_dt2": euler,
        "exact_constant_score_dt2": probabilities(advance(start,scores,2.)),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(result(), indent=2, allow_nan=False)+'\n'
    if args.check_sample:
        if text != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample output mismatch')
        print('Sample output matches.')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
