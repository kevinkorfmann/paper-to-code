"""Deterministic supplied-logit controls; no quantizer or model weights."""
import argparse
import json
import math
from pathlib import Path
from .model import action, partial, merge, output


def results():
    high = partial([math.log(9)], [0])
    low = partial([0], [10])
    retained = output(merge([high, low]))
    perturbed = output(merge([high, partial([0.1], [10])]))
    return {
        'environment': 'CPython 3.14.5, standard library only',
        'supplied_logits': [round(math.log(9), 9), 0],
        'values': [0, 10],
        'shared_normalization': round(retained, 9),
        'wrong_equal_average_of_group_outputs': (output(high) + output(low)) / 2,
        'skip_second_group': output(high),
        'retained_with_supplied_logit_error_0.1': round(perturbed, 9),
        'absolute_error_perturb': round(abs(perturbed - retained), 9),
        'absolute_error_skip': round(abs(retained - output(high)), 9),
        'query_block_6_policy_w8_2_w4_4': [action(6, j, 2, 4) for j in range(8)],
        'note': 'Supplied score perturbation is not a simulation of INT4 quantization.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(results(), indent=2, allow_nan=False) + '\n'
    if args.check_sample:
        if text != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample output mismatch')
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
