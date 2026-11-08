"""Same allele frequency, different heterozygosity through mating correlation."""
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
TREATMENTS = {'disassortative': 0.1, 'random': 0.5, 'assortative': 0.9,
              'opposite_only': 0.0, 'same_only': 1.0}
SEED = 202611080


def validate(n, q):
    _integer(n, 'n', 4, 1000)
    if n % 2:
        raise ValueError('n must be even')
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('q must be a finite probability')


def oracle(n, q):
    validate(n, q)
    v = q/(4*n)
    fourth = (n*q/16+3*n*(n-1)*(q/4)**2)/n**4
    return {'genotype_probabilities_aa_Aa_AA': [q/2, 1-q, q/2],
            'mean_frequency': 0.5, 'variance_frequency': v,
            'mean_observed_heterozygosity': 1-q,
            'variance_observed_heterozygosity': q*(1-q)/n,
            'mean_gene_diversity': 0.5-q/(2*n),
            'variance_gene_diversity': max(0.0, 4*(fourth-v*v))}


def parse_output(text, n, replicates):
    _integer(n, 'n', 4, 1000)
    _integer(replicates, 'replicates', 1, 2048)
    init, data = {}, {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT', 'DATA'):
            continue
        if len(row) != 5:
            raise ValueError('wrong row width')
        r, *counts = map(int, row[1:])
        dest = init if row[0] == 'INIT' else data
        if r in dest or any(c < 0 for c in counts) or sum(counts) != n:
            raise ValueError('duplicate or invalid counts')
        if row[0] == 'INIT' and counts != [n//2, 0, n//2]:
            raise ValueError('wrong founders')
        dest[r] = counts
    if set(init) != set(range(replicates)) or set(data) != set(range(replicates)):
        raise ValueError('missing or unexpected replicate')
    return [data[r] for r in range(replicates)]


def run(q=0.5, n=200, replicates=512, seed=SEED, executable='slim'):
    validate(n, q)
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for key, value in [('N', n), ('REPLICATES', replicates), ('Q', float(q))]:
        command += ['-d', f'{key}={value}']
    result = subprocess.run([*command, str(HERE/'model.slim')], capture_output=True,
                            text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, replicates)


def summarize(rows, n, q):
    expected = oracle(n, q)
    if not rows:
        raise ValueError('no observations')
    for row in rows:
        if len(row) != 3 or any(type(x) is not int or x < 0 for x in row) or sum(row) != n:
            raise ValueError('invalid counts')
    ps = [(row[1]+2*row[2])/(2*n) for row in rows]
    measurements = {'frequency': ps, 'observed_heterozygosity': [row[1]/n for row in rows],
                    'gene_diversity': [2*p*(1-p) for p in ps]}
    result = {'exact': expected, 'replicates': len(rows)}
    for key, values in measurements.items():
        mean = statistics.mean(values)
        se = math.sqrt(expected['variance_'+key]/len(rows))
        if abs(mean-expected['mean_'+key]) > 6*se+1e-12:
            raise AssertionError(key+' outside six exact standard errors')
        result['mean_'+key] = mean
        result['exact_standard_error_'+key] = se
    return result


def summary():
    return {'experiment': 'One-generation genotype-assortative mating',
            'slim_version': '4.0.1', 'python_version': platform.python_version(),
            'individuals_per_population': 200, 'replicates_per_treatment': 512,
            'synthetic_only': True,
            'seed_by_treatment': {name: SEED+i for i, name in enumerate(TREATMENTS)},
            'treatments': {name: summarize(run(q, seed=SEED+i), 200, q)
                           for i, (name, q) in enumerate(TREATMENTS.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    result = summary()
    if args.check_sample:
        if result != json.loads((HERE/'sample_summary.json').read_text()):
            raise AssertionError('checked output differs')
        print('Checked output reproduced exactly')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__': main()
