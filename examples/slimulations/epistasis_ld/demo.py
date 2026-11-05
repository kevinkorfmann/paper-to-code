"""Epistasis creates LD without moving marginal allele frequencies in expectation."""
import argparse
import csv
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import platform
import subprocess
from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

HERE = Path(__file__).parent
SEED = 202611050
TREATMENTS = {
    'positive_unlinked': (0.8, 0.5),
    'positive_linked': (0.8, 0.0),
    'negative_unlinked': (-0.8, 0.5),
    'neutral_unlinked': (0.0, 0.5),
}


def _real(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return value


def haplotype_probabilities(s, r):
    """Exact one-generation result for the specified uniform phased founders."""
    _real(s, 's', -1, 1)
    _real(r, 'r', 0, 0.5)
    return (0.25+s/16, 0.25-s/16, 0.25-s/16, 0.25+s/16)


def ld_moments(probabilities, copies):
    """Exact multinomial moments of D_hat=f(ab)f(AB)-f(aB)f(Ab).

    Falling-factorial moments account for finite-sample bias. Fraction arithmetic
    avoids subtractive loss of precision in the second moment calculation.
    """
    _integer(copies, 'copies', 2, 1000000)
    if len(probabilities) != 4:
        raise ValueError('Four haplotype probabilities required')
    for p in probabilities:
        _real(p, 'probability', 0, 1)
    if abs(sum(probabilities)-1) > 1e-12:
        raise ValueError('Probabilities must sum to one')
    q = [Fraction(str(p)) for p in probabilities]
    q = [p/sum(q) for p in q]
    a, b, c, d = q
    m = copies
    ld = a*d-b*c
    fall2 = m*(m-1)
    fall3 = fall2*(m-2)
    fall4 = fall3*(m-3)
    mean = Fraction(m-1,m)*ld
    second = (fall4*ld*ld+fall3*(a*d*(a+d)+b*c*(b+c))+fall2*(a*d+b*c))/m**4
    return {'population_ld': float(ld), 'mean_sample_ld': float(mean),
            'variance_sample_ld': float(second-mean*mean)}


def parse_output(text, n, replicates):
    initial, data = {}, {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT', 'DATA'):
            continue
        kind = row[0]
        if len(row) != (18 if kind == 'INIT' else 6):
            raise ValueError('Malformed simulation row')
        rep, *counts = map(int, row[1:])
        target = initial if kind == 'INIT' else data
        if rep in target or any(x < 0 for x in counts):
            raise ValueError('Duplicate row or negative count')
        if kind == 'INIT' and counts != [n//16]*16:
            raise ValueError('Incorrect phased founder counts')
        if kind == 'DATA' and sum(counts) != 2*n:
            raise ValueError('Wrong offspring copy count')
        target[rep] = tuple(counts)
    if set(initial) != set(range(replicates)) or set(data) != set(range(replicates)):
        raise ValueError('Missing or unexpected replicate')
    return [data[r] for r in range(replicates)]


def run(s=0.8, r=0.5, n=256, replicates=512, seed=SEED, executable='slim'):
    haplotype_probabilities(s, r)
    _integer(n, 'n', 16, 1024)
    if n % 16:
        raise ValueError('n must be divisible by sixteen')
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed), '-d', f'N={n}', '-d', f'REPLICATES={replicates}',
               '-d', f'S={s}', '-d', f'R={r}', str(HERE/'model.slim')]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, replicates)


def summarize_rows(rows, s, r, n):
    q = haplotype_probabilities(s, r)
    copies = 2*n
    total = copies*len(rows)
    observed = [sum(row[h] for row in rows)/total for h in range(4)]
    moments = ld_moments(q, copies)
    # Independent offspring gametes, conditional on identical fixed founders.
    for obs, p in zip(observed, q):
        if abs(obs-p) > 6*math.sqrt(p*(1-p)/total):
            raise AssertionError('Haplotype frequency outside six-standard-error bound')
    ld = [(row[0]*row[3]-row[1]*row[2])/copies**2 for row in rows]
    mean_ld = sum(ld)/len(ld)
    exact_se = math.sqrt(moments['variance_sample_ld']/len(rows))
    if abs(mean_ld-moments['mean_sample_ld']) > 6*exact_se:
        raise AssertionError('Mean LD outside six-standard-error bound')
    return {'s': s, 'recombination_fraction': r, 'observed_haplotype_frequencies_ab_aB_Ab_AB': observed,
            'observed_A_frequency': observed[2]+observed[3],
            'observed_B_frequency': observed[1]+observed[3],
            'mean_within_population_sample_ld': mean_ld,
            'exact_standard_error_of_mean_ld': exact_se,
            'exact_haplotype_probabilities_ab_aB_Ab_AB': list(q),
            'exact': moments}


def summary():
    treatments = {}
    for i, (name, (s,r)) in enumerate(TREATMENTS.items()):
        rows = run(s,r,seed=SEED+i)
        treatments[name] = summarize_rows(rows,s,r,256)
    return {'experiment': 'Dosage epistasis generates LD in one reproductive generation',
            'slim_version': '4.0.1', 'python_version': platform.python_version(),
            'population_size': 256, 'replicates_per_treatment': 512,
            'seed_by_treatment': {name:SEED+i for i,name in enumerate(TREATMENTS)},
            'synthetic_only': True, 'treatments': treatments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    result = summary()
    if args.check_sample:
        if result != json.loads((HERE/'sample_summary.json').read_text()):
            raise AssertionError('Checked output differs')
        print('Checked output reproduced exactly')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
