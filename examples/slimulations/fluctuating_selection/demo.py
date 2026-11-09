"""Temporal fitness multiplies: a clonal SLiM experiment and finite Markov oracle."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

HERE = Path(__file__).parent
SEED = 202611090
TREATMENTS = {'alternating': (1.8, 0.4), 'reversed': (0.4, 1.8),
              'constant_arithmetic': (1.1, 1.1),
              'constant_geometric': (math.sqrt(0.72), math.sqrt(0.72)),
              'neutral': (1.0, 1.0)}


def validate(n, k, steps, weights):
    _integer(n, 'n', 2, 200)
    _integer(k, 'k', 0, n)
    _integer(steps, 'steps', 1, 40)
    if len(weights) != 2 or any(type(w) not in (int, float) or not math.isfinite(w)
                               or not 0.1 <= w <= 3 for w in weights):
        raise ValueError('need two finite fitnesses in [0.1, 3]')


def selected_frequency(k, n, w):
    return k*w/(k*w+(n-k))


def binomial_row(n, p):
    if not 0 <= p <= 1:
        raise ValueError('invalid probability')
    # Direct PMF is safe for the deliberately small N <= 200.
    return [math.comb(n, j)*p**j*(1-p)**(n-j) for j in range(n+1)]


def oracle(n=100, k=50, steps=12, weights=(1.8, 0.4)):
    """Exact finite-state model evaluated in double precision, all replicates included."""
    validate(n, k, steps, weights)
    matrices = [[binomial_row(n, selected_frequency(i, n, w))
                 for i in range(n+1)] for w in weights]
    prob = [float(i == k) for i in range(n+1)]
    result = []
    for g in range(steps+1):
        mean = math.fsum(i/n*prob[i] for i in range(n+1))
        var = math.fsum((i/n-mean)**2*prob[i] for i in range(n+1))
        result.append({'mean_frequency': mean, 'variance_frequency': var,
                       'loss_probability': prob[0], 'fixation_probability': prob[-1]})
        if g != steps:
            matrix = matrices[g % 2]
            prob = [math.fsum(prob[i]*matrix[i][j] for i in range(n+1))
                    for j in range(n+1)]
            if not math.isclose(math.fsum(prob), 1.0, abs_tol=1e-11):
                raise ArithmeticError('transition mass lost')
    return result


def deterministic(k=50, n=100, steps=12, weights=(1.8, 0.4)):
    validate(n, k, steps, weights)
    p = k/n
    out = [p]
    for g in range(steps):
        w = weights[g % 2]
        p = p*w/(p*w+(1-p))
        out.append(p)
    return out


def parse_output(text, n, k, steps, replicates):
    validate(n, k, steps, (1, 1))
    _integer(replicates, 'replicates', 1, 2048)
    rows = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] != 'DATA':
            continue
        if len(row) != 4:
            raise ValueError('wrong row width')
        g, r, count = map(int, row[1:])
        if not 0 <= g <= steps or not 0 <= r < replicates or not 0 <= count <= n or (g, r) in rows:
            raise ValueError('invalid or duplicate observation')
        if g == 0 and count != k:
            raise ValueError('wrong founder count')
        rows[g, r] = count
    if len(rows) != (steps+1)*replicates:
        raise ValueError('missing observations')
    return [[rows[g, r] for r in range(replicates)] for g in range(steps+1)]


def run(n=100, k=50, steps=12, weights=(1.8, 0.4), replicates=512, seed=SEED, executable='slim'):
    validate(n, k, steps, weights)
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for key, value in [('N', n), ('K', k), ('STEPS', steps), ('REPLICATES', replicates),
                       ('W1', float(weights[0])), ('W2', float(weights[1]))]:
        command += ['-d', f'{key}={value}']
    result = subprocess.run([*command, str(HERE/'model.slim')], capture_output=True,
                            text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, k, steps, replicates)


def experiment(executable='slim'):
    summary = {'runtime': {'python': platform.python_version(), 'slim': '4.0.1',
                            'eidos': '3.0.1'},
               'n_clonal_individuals': 100, 'replicates': 512, 'generations': 12,
               'initial_frequency': 0.5, 'treatments': {}}
    for index, (name, weights) in enumerate(TREATMENTS.items()):
        rows = run(weights=weights, seed=SEED+index, executable=executable)
        exact = oracle(weights=weights)
        det = deterministic(weights=weights)
        trajectory = []
        for g, counts in enumerate(rows):
            ps = [x/100 for x in counts]
            mean = statistics.mean(ps)
            se = statistics.stdev(ps)/math.sqrt(len(ps))
            target = exact[g]
            # Deliberately broad six-SE regression check, not a scientific p-value.
            expected_se = math.sqrt(target['variance_frequency']/len(ps))
            if abs(mean-target['mean_frequency']) > 6*expected_se+1e-10:
                raise AssertionError(f'{name} generation {g} disagrees with finite oracle')
            trajectory.append({'generation': g, 'mean_frequency': round(mean, 9),
                               'monte_carlo_se': round(se, 9),
                               'exact_finite_mean': round(target['mean_frequency'], 9),
                               'deterministic_frequency': round(det[g], 9),
                               'losses': counts.count(0), 'fixations': counts.count(100)})
        summary['treatments'][name] = {'fitnesses_relative_to_B': list(weights),
                                     'seed': SEED+index, 'trajectory': trajectory}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim', default='slim')
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    output = json.dumps(experiment(args.slim), indent=2, sort_keys=True)+'\n'
    if args.check_sample:
        if output != (HERE/'sample_summary.json').read_text():
            raise SystemExit('checked output differs')
        print('sample matches')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
