"""Recurrent mutation opposes purifying selection in a finite clonal population."""
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
SEED = 202611100
# (initial A clones, A reproductive weight, B->A probability, A->B probability)
TREATMENTS = {'selected': (0, .8, .02, .01),
              'no_forward_mutation': (0, .8, 0., .01),
              'neutral': (0, 1., .02, .01),
              'strong_selection': (0, .5, .02, .01),
              'initially_all_A': (100, .8, .02, .01)}


def validate(n, k, steps, w, u, v):
    _integer(n, 'n', 2, 200)
    _integer(k, 'k', 0, n)
    _integer(steps, 'steps', 1, 80)
    for name, value, lo, hi in [('w', w, .1, 1), ('u', u, 0, 1), ('v', v, 0, 1)]:
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'invalid {name}')


def offspring_probability(p, w, u, v):
    """Selection first, followed by exactly one mutation opportunity per clone."""
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid frequency')
    validate(2, 0, 1, w, u, v)
    selected = p*w/(p*w+(1-p))
    # Convex form avoids cancellation at p=1.
    return selected*(1-v)+(1-selected)*u


def binomial_row(n, p):
    _integer(n, 'n', 2, 200)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid probability')
    return [math.comb(n, j)*p**j*(1-p)**(n-j) for j in range(n+1)]


def oracle(n=100, k=0, steps=40, w=.8, u=.02, v=.01):
    """Exact finite-state transition model evaluated in double precision."""
    validate(n, k, steps, w, u, v)
    matrix = [binomial_row(n, offspring_probability(i/n, w, u, v)) for i in range(n+1)]
    prob = [float(i == k) for i in range(n+1)]
    result = []
    for g in range(steps+1):
        mean = math.fsum(i/n*prob[i] for i in range(n+1))
        var = math.fsum((i/n-mean)**2*prob[i] for i in range(n+1))
        result.append({'mean_frequency': mean, 'variance_frequency': var,
                       'absence_probability': prob[0], 'all_A_probability': prob[-1]})
        if g != steps:
            prob = [math.fsum(prob[i]*matrix[i][j] for i in range(n+1)) for j in range(n+1)]
            if not math.isclose(math.fsum(prob), 1., abs_tol=1e-11):
                raise ArithmeticError('transition mass lost')
    return result


def deterministic(k=0, n=100, steps=40, w=.8, u=.02, v=.01):
    validate(n, k, steps, w, u, v)
    ps = [k/n]
    for _ in range(steps):
        ps.append(offspring_probability(ps[-1], w, u, v))
    return ps


def equilibrium(w=.8, u=.02, v=.01):
    """Unique interior deterministic fixed point for 0<u,v and u+v<=1."""
    validate(2, 0, 1, w, u, v)
    if u <= 0 or v <= 0 or u+v > 1:
        raise ValueError('equilibrium helper requires positive u,v with u+v<=1')
    low, high = 0., 1.
    for _ in range(60):
        p = (low+high)/2
        if offspring_probability(p, w, u, v) > p:
            low = p
        else:
            high = p
    return (low+high)/2


def parse_output(text, n, k, steps, replicates):
    validate(n, k, steps, 1, 0, 0)
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


def run(n=100, k=0, steps=40, w=.8, u=.02, v=.01, replicates=512, seed=SEED, executable='slim'):
    validate(n, k, steps, w, u, v)
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for key, value in [('N', n), ('K', k), ('STEPS', steps), ('REPLICATES', replicates),
                       ('W', float(w)), ('U', float(u)), ('V', float(v))]:
        command += ['-d', f'{key}={value}']
    result = subprocess.run([*command, str(HERE/'model.slim')], capture_output=True,
                            text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, k, steps, replicates)


def experiment(executable='slim'):
    summary = {'runtime': {'python': platform.python_version(), 'slim': '4.0.1', 'eidos': '3.0.1'},
               'n_clonal_individuals': 100, 'replicates': 512, 'generations': 40, 'treatments': {}}
    for index, (name, (k, w, u, v)) in enumerate(TREATMENTS.items()):
        args = dict(k=k, w=w, u=u, v=v)
        rows = run(**args, seed=SEED+index, executable=executable)
        exact = oracle(**args)
        det = deterministic(**args)
        trajectory = []
        for g, counts in enumerate(rows):
            ps = [x/100 for x in counts]
            mean = statistics.mean(ps)
            se = statistics.stdev(ps)/math.sqrt(len(ps))
            target = exact[g]
            expected_se = math.sqrt(target['variance_frequency']/len(ps))
            if abs(mean-target['mean_frequency']) > 6*expected_se+1e-10:
                raise AssertionError(f'{name} generation {g} disagrees with finite oracle')
            if g in (0, 1, 10, 20, 40):
                trajectory.append({'generation': g, 'mean_frequency': round(mean, 9),
                                   'monte_carlo_se': round(se, 9),
                                   'exact_finite_mean': round(target['mean_frequency'], 9),
                                   'deterministic_frequency': round(det[g], 9),
                                   'absent_populations': counts.count(0), 'all_A_populations': counts.count(100)})
        summary['treatments'][name] = {'initial_count': k, 'A_fitness_relative_to_B': w,
                                     'u_B_to_A': u, 'v_A_to_B': v, 'seed': SEED+index,
                                     'deterministic_equilibrium': round(equilibrium(w,u,v),9) if u*v else None,
                                     'trajectory': trajectory}
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
