"""Underdominance can polarize populations while their mean remains near one-half."""
import argparse
import csv
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

SCRIPT = Path(__file__).with_name('underdominance.slim')
SAMPLE = Path(__file__).with_name('sample_summary.json')
SEED = 202611030
# Genotypes are always (aa, Aa, AA), before parental reproductive selection.
TREATMENTS = {
    'underdominant_below': ((36, 48, 16), .6),
    'underdominant_at_threshold': ((25, 50, 25), .6),
    'underdominant_above': ((16, 48, 36), .6),
    'neutral_at_threshold': ((25, 50, 25), 1.),
    'small_underdominant': ((2, 4, 2), .6),
    'small_neutral': ((2, 4, 2), 1.),
}


def validate(founders, w_het, generations):
    if not isinstance(founders, (tuple, list)) or len(founders) != 3:
        raise ValueError('Founders must contain aa, Aa, AA counts')
    for x in founders:
        _integer(x, 'founder count', 0, 200)
    _integer(sum(founders), 'population size', 2, 200)
    if isinstance(w_het, bool) or not isinstance(w_het, (int, float)) or not math.isfinite(w_het) or not 0 < w_het <= 1:
        raise ValueError('Heterozygote fitness must be finite and in (0,1]')
    _integer(generations, 'generations', 0, 100)


def gamete_frequency(counts, w_het):
    """Exact conditional probability that a fitness-weighted gamete carries A."""
    validate(counts, w_het, 0)
    aa, het, AA = counts
    return (AA + .5*w_het*het)/(aa + w_het*het + AA)


def deterministic_path(p, w_het, generations):
    """Infinite random-mating benchmark, not an exact finite-ensemble mean."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid allele frequency')
    validate((1, 0, 1), w_het, generations)
    path = [p]
    for _ in range(generations):
        p = (p*p + w_het*p*(1-p))/(1-2*(1-w_het)*p*(1-p))
        path.append(p)
    return path


@lru_cache(maxsize=20)
def states(n):
    _integer(n, 'exact-model population size', 2, 20)
    return tuple((a, h, n-a-h) for a in range(n+1) for h in range(n-a+1))


@lru_cache(maxsize=20)
def transition(n, w_het):
    """Exact finite genotype-count transition, evaluated in floating point."""
    configurations = states(n)
    validate((n, 0, 0), w_het, 0)
    matrix = []
    for source in configurations:
        q = gamete_frequency(source, w_het)
        probs = ((1-q)**2, 2*q*(1-q), q*q)
        row = []
        for a, h, d in configurations:
            coefficient = math.comb(n, a)*math.comb(n-a, h)
            row.append(coefficient*probs[0]**a*probs[1]**h*probs[2]**d)
        matrix.append(tuple(row))
    return tuple(matrix)


def exact_path(founders, w_het, generations):
    validate(founders, w_het, generations)
    n = sum(founders)
    configurations = states(n)
    matrix = transition(n, w_het)
    mass = [float(s == tuple(founders)) for s in configurations]
    result = []
    for t in range(generations+1):
        if t:
            mass = [math.fsum(m*row[j] for m, row in zip(mass, matrix)) for j in range(len(mass))]
        frequencies = [(h+2*d)/(2*n) for a, h, d in configurations]
        result.append({'generation': t,
            'mean_p': math.fsum(m*p for m, p in zip(mass, frequencies)),
            'mean_gene_diversity': math.fsum(m*2*p*(1-p) for m, p in zip(mass, frequencies)),
            'second_moment_p': math.fsum(m*p*p for m, p in zip(mass, frequencies)),
            'second_moment_gene_diversity': math.fsum(m*(2*p*(1-p))**2 for m, p in zip(mass, frequencies)),
            'loss_probability': mass[configurations.index((n, 0, 0))],
            'fixation_probability': mass[configurations.index((0, 0, n))],
            'probability_mass': math.fsum(mass)})
    return result


def parse_output(text, founders, w_het, generations, replicates):
    records = {}; fitness = {}
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0] == 'DATA':
            if len(row) != 6:
                raise ValueError('Malformed DATA row')
            t, r, a, h, d = map(int, row[1:])
            if (t, r) in records or not 0 <= t <= generations or not 0 <= r < replicates or min(a, h, d) < 0 or a+h+d != sum(founders):
                raise ValueError('Invalid or duplicate genotype record')
            records[t, r] = (a, h, d)
        elif row[0] == 'FIT':
            if len(row) != 3 or int(row[1]) in fitness:
                raise ValueError('Malformed or duplicate fitness audit')
            fitness[int(row[1])] = float(row[2])
    if set(records) != {(t, r) for t in range(generations+1) for r in range(replicates)}:
        raise ValueError('Missing genotype records')
    expected = {g: (w_het if g == 1 else 1.) for g, count in enumerate(founders) if count}
    if set(fitness) != set(expected) or any(not math.isclose(fitness[g], w, rel_tol=2e-6) for g, w in expected.items()):
        raise ValueError('Wrong fitness audit')
    if 'Eidos version 3.0.1\n' not in text or 'SLiM version 4.0.1\n' not in text:
        raise ValueError('Missing runtime versions')
    for r in range(replicates):
        if records[0, r] != tuple(founders):
            raise ValueError('Wrong founder genotypes')
        for t in range(generations):
            a, h, d = records[t, r]
            if (a == sum(founders) or d == sum(founders)) and records[t+1, r] != records[t, r]:
                raise ValueError('Absorbed state changed')
    return {t: [records[t, r] for r in range(replicates)] for t in range(generations+1)}


def run_counts(founders=(25, 50, 25), w_het=.6, generations=80, replicates=512, seed=SEED, executable='slim'):
    validate(founders, w_het, generations)
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for name, value in [('N', sum(founders)), ('INITIAL_AA', founders[2]), ('INITIAL_HET', founders[1]),
                        ('W_HET', w_het), ('GENERATIONS', generations), ('REPLICATES', replicates)]:
        command += ['-d', f'{name}={value}']
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError('SLiM failed: '+result.stderr[-2000:])
    return parse_output(result.stdout, founders, w_het, generations, replicates)


def summarize(rows):
    n = sum(rows[0]); r = len(rows)
    frequencies = [(h+2*d)/(2*n) for a, h, d in rows]
    diversity = [2*p*(1-p) for p in frequencies]
    def se(values):
        return statistics.stdev(values)/math.sqrt(r) if r > 1 else None
    mean_p = statistics.mean(frequencies)
    return {'mean_p': mean_p, 'mean_p_monte_carlo_se': se(frequencies),
            'mean_gene_diversity': statistics.mean(diversity), 'gene_diversity_monte_carlo_se': se(diversity),
            'gene_diversity_of_pooled_frequency': 2*mean_p*(1-mean_p),
            'mean_observed_heterozygosity': statistics.mean(h/n for a, h, d in rows),
            'mean_absolute_distance_from_half': statistics.mean(abs(p-.5) for p in frequencies),
            'lost_replicates': sum(a == n for a, h, d in rows),
            'fixed_replicates': sum(d == n for a, h, d in rows),
            'segregating_replicates': sum(a != n and d != n for a, h, d in rows)}


def build_report(executable='slim', replicates=512, seed=SEED, results=None):
    _integer(seed, 'seed', 1, 2**62-len(TREATMENTS))
    groups = []
    for i, (name, (founders, w)) in enumerate(TREATMENTS.items()):
        counts = results[name] if results is not None else run_counts(founders, w, replicates=replicates, seed=seed+i, executable=executable)
        n = sum(founders); p0 = (founders[1]+2*founders[2])/(2*n)
        deterministic = deterministic_path(p0, w, 80)
        finite = exact_path(founders, w, 80) if n <= 20 else None
        groups.append({'name': name, 'seed': seed+i, 'diploids': n, 'founders_aa_Aa_AA': list(founders),
            'fitness_aa_Aa_AA': [1., w, 1.], 'initial_p': p0,
            'first_generation_exact_mean_p': gamete_frequency(founders, w),
            'checkpoints': [{'generation': t, 'observed': summarize(counts[t]),
                'infinite_population_p': deterministic[t], 'exact_finite_genotype_model': finite[t] if finite else None}
                for t in (0, 1, 10, 40, 80)]})
    return {'example': 'Underdominance: stable ensemble means can hide unstable populations',
            'slim_version': '4.0.1', 'eidos_version': '3.0.1', 'python_version': platform.python_version(),
            'replicates_per_treatment': replicates, 'treatments': groups,
            'interpretation': 'All replicates including loss and fixation are counted. Deterministic paths are not finite-ensemble predictions. Exact finite genotype transitions are evaluated numerically for N=8. No mutation or migration.'}


def render(report):
    def rounded(x):
        if isinstance(x, float): return round(x, 9)
        if isinstance(x, list): return [rounded(v) for v in x]
        if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
        return x
    return json.dumps(rounded(report), indent=2, allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim', default='slim')
    parser.add_argument('--replicates', type=int, default=512)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    try:
        text = render(build_report(args.slim, args.replicates, args.seed))
        if args.check_sample and text != SAMPLE.read_text(): raise ValueError('Output differs from checked sample')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(text, end='')


if __name__ == '__main__': main()
