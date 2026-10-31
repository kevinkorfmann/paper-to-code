"""Run actual SLiM and distinguish exact first-generation predictions from later benchmarks."""

import argparse
import csv
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

SCRIPT = Path(__file__).with_name('hitchhiking.slim')
SAMPLE = Path(__file__).with_name('sample_summary.json')
SEED = 202610310
TREATMENTS = [('linked_selected', 1, 0, 0.0), ('free_recombination_selected', 1, 0, 0.5),
              ('linked_neutral', 0, 0, 0.0), ('reversed_linked_selected', 1, 1, 0.0)]


def founder_frequencies(reversed=0):
    _integer(reversed, 'reversed', 0, 1)
    return [Fraction(v, 10) for v in ([4, 5, 1, 0] if reversed else [5, 4, 0, 1])]


def founder_pairs(n, reversed=0):
    _integer(n, 'n', 100, 2000)
    if n % 100: raise ValueError('N must be divisible by 100 for exact founder pairing')
    x = founder_frequencies(reversed)
    pairs = {(i, j): n*x[i]*x[j]*(1 if i == j else 2) for i in range(4) for j in range(i, 4)}
    assert all(count.denominator == 1 for count in pairs.values())
    return {pair: int(count) for pair, count in pairs.items()}


def gamete_step(x, selected, c):
    """Infinite random-pairing recurrence; exact for the constructed founders' first offspring."""
    _integer(selected, 'selected', 0, 1)
    if len(x) != 4 or any(not math.isfinite(v) or v < 0 for v in x) or not math.isclose(sum(x), 1, abs_tol=1e-12):
        raise ValueError('Expected four nonnegative haplotype frequencies summing to one')
    if isinstance(c, bool) or not math.isfinite(c) or not 0 <= c <= .5:
        raise ValueError('Recombination fraction must lie in [0,0.5]')
    s = Fraction(selected); half = Fraction(1, 2)
    p = x[2]+x[3]; wbar = 1+s*p
    y = [v*(1+s*half*((i//2)+p))/wbar for i, v in enumerate(x)]
    d = x[3]*x[0]-x[2]*x[1]
    correction = c*(1+s*half)*d/wbar
    return [y[0]-correction, y[1]+correction, y[2]+correction, y[3]-correction]


def deterministic(reversed, selected, c, generations):
    _integer(generations, 'generations', 0, 100)
    x = list(map(float, founder_frequencies(reversed))); path = [x]
    for _ in range(generations):
        x = list(map(float, gamete_step(x, selected, c))); path.append(x)
    return path


def first_generation_prediction(n, reversed, selected, c):
    founder_pairs(n, reversed)
    q = gamete_step(founder_frequencies(reversed), selected, Fraction(str(c)))
    moments = finite_moments(q, n)
    return {'gamete_haplotype_probabilities': list(map(float, q)),
            'expected_pB_fraction': str(q[1]+q[3]),
            **{key: float(value) for key, value in moments.items()}}


def finite_moments(q, n):
    """Moments of 2N independent gametes, including plug-in diversity/LD bias."""
    _integer(n, 'n', 1, 2000)
    gamete_step(q, 0, 0)  # Validate the probability vector.
    a, b = q[2]+q[3], q[1]+q[3]
    correction = Fraction(2*n-1, 2*n)
    return {'expected_pA': a, 'expected_pB': b, 'pB_variance': b*(1-b)/(2*n),
            'expected_gene_diversity_B': 2*b*(1-b)*correction,
            'expected_sample_LD': (q[3]*q[0]-q[2]*q[1])*correction}


def parse_output(text, n, replicates, generations, selected, reversed):
    records = {}; init = {}; fitness = {}
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        if row[0] == 'DATA':
            if len(row) != 7: raise ValueError('Malformed haplotype output')
            t, rep, *counts = map(int, row[1:])
            if (t, rep) in records or min(counts) < 0 or sum(counts) != 2*n:
                raise ValueError('Invalid or duplicate haplotype counts')
            records[t, rep] = counts
        elif row[0] == 'INIT':
            if len(row) != 5: raise ValueError('Malformed founder audit')
            rep, a, b, count = map(int, row[1:])
            if (rep, a, b) in init: raise ValueError('Duplicate founder audit')
            init[rep, a, b] = count
        elif row[0] == 'FIT':
            if len(row) != 3 or int(row[1]) in fitness: raise ValueError('Invalid fitness audit')
            fitness[int(row[1])] = float(row[2])
    times = sorted({0, 1, generations//3, generations} & set(range(generations+1)))
    if set(records) != {(t, r) for t in times for r in range(replicates)}:
        raise ValueError('Missing or unexpected replicate/checkpoint')
    pairs = founder_pairs(n, reversed)
    if init != {(r, a, b): count for r in range(replicates) for (a, b), count in pairs.items()}:
        raise ValueError('Actual founder diplotypes differ from the exact requested pairing')
    if fitness != {g: 1+selected*g/2 for g in range(3)}:
        raise ValueError('Native SLiM fitness does not match the analytical model')
    if 'Eidos version 3.0.1\n' not in text or 'SLiM version 4.0.1\n' not in text:
        raise ValueError('Unexpected SLiM/Eidos version')
    initial = [int(v*2*n) for v in founder_frequencies(reversed)]
    if any(records[0, r] != initial for r in range(replicates)):
        raise ValueError('Wrong initial haplotype counts')
    return {t: [records[t, r] for r in range(replicates)] for t in times}


def run_counts(*, n=400, replicates=256, generations=30, selected=1, reversed=0, c=0., seed=SEED, executable='slim'):
    founder_pairs(n, reversed); gamete_step(founder_frequencies(reversed), selected, c)
    _integer(replicates, 'replicates', 1, 1024); _integer(generations, 'generations', 0, 100)
    _integer(seed, 'seed', 1, 2**62-1); check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for name, value in [('N', n), ('REPLICATES', replicates), ('GENERATIONS', generations),
                        ('SELECTED', selected), ('REVERSED', reversed), ('C', c)]:
        command += ['-d', f'{name}={value}']
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, timeout=120)
    if result.returncode: raise ValueError('SLiM failed: '+result.stderr[-2000:])
    return parse_output(result.stdout, n, replicates, generations, selected, reversed)


def statistics_for(rows, n):
    a = [(r[2]+r[3])/(2*n) for r in rows]; b = [(r[1]+r[3])/(2*n) for r in rows]
    h = [2*p*(1-p) for p in b]; d = [(r[3]*r[0]-r[2]*r[1])/(2*n)**2 for r in rows]
    def se(values): return statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None
    return {'mean_pA': statistics.mean(a), 'mean_pB': statistics.mean(b), 'pB_monte_carlo_se': se(b),
            'pB_sample_variance': statistics.variance(b) if len(b) > 1 else None,
            'mean_gene_diversity_B': statistics.mean(h), 'gene_diversity_B_monte_carlo_se': se(h),
            'mean_LD': statistics.mean(d), 'selected_A_fixed': sum(p == 1 for p in a),
            'selected_A_lost': sum(p == 0 for p in a), 'neutral_B_fixed': sum(p == 1 for p in b),
            'neutral_B_lost': sum(p == 0 for p in b)}


def build_report(executable='slim', n=400, replicates=256, generations=30, seed=SEED, results=None):
    _integer(seed, 'seed', 1, 2**62-4)
    treatments = []
    for i, (name, selected, reversed, c) in enumerate(TREATMENTS):
        rows = (results[name] if results is not None else run_counts(n=n, replicates=replicates,
                generations=generations, selected=selected, reversed=reversed, c=c, seed=seed+i, executable=executable))
        trajectory = deterministic(reversed, selected, c, generations)
        observed = []
        for t, counts in rows.items():
            x = trajectory[t]; pB = x[1]+x[3]
            item = {'generation': t, **statistics_for(counts, n),
                    'infinite_population_pA': x[2]+x[3], 'infinite_population_pB': pB,
                    'infinite_population_gene_diversity_B': 2*pB*(1-pB)}
            if selected == 0: item['exact_finite_neutral_expected_diversity_B'] = .5*(1-1/(2*n))**t
            observed.append(item)
        treatments.append({'name': name, 'seed': seed+i, 'recombination_fraction': c,
                           'fitness_aa_Aa_AA': [1., 1.+selected*.5, 1.+selected],
                           'initial_haplotype_frequencies_ab_aB_Ab_AB': list(map(float, founder_frequencies(reversed))),
                           'exact_first_generation': first_generation_prediction(n, reversed, selected, c),
                           'checkpoints': observed})
    return {'example': 'Neutral genetic hitchhiking and recombination',
            'slim_version': '4.0.1', 'eidos_version': '3.0.1', 'python_version': platform.python_version(),
            'diploid_population_size': n, 'replicates_per_treatment': replicates,
            'generations_elapsed': generations, 'treatments': treatments,
            'interpretation': 'All replicates are included, without conditioning on fixation. The first-generation expectations are exact for the audited founders; later selected trajectories are infinite-population benchmarks, not exact finite-population means. Gene diversity is 2pB(1-pB), not observed individual heterozygosity.'}


def render(result):
    def rounded(x):
        if isinstance(x, float): return round(x, 9)
        if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(result), indent=2, allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim', default='slim'); parser.add_argument('--n', type=int, default=400)
    parser.add_argument('--replicates', type=int, default=256); parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--seed', type=int, default=SEED); parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    try:
        text = render(build_report(args.slim, args.n, args.replicates, args.generations, args.seed))
        if args.check_sample and text != SAMPLE.read_text(): raise ValueError('Output differs from checked sample')
    except (ValueError, OSError, subprocess.SubprocessError) as error: parser.error(str(error))
    print(text, end='')


if __name__ == '__main__': main()
