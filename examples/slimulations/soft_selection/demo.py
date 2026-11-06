"""Run actual SLiM and check one-generation soft/hard-selection predictions."""
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
MODES = {'neutral': 0, 'soft': 1, 'hard': 2}
STARTS = {1: (64, 32, 4), 2: (25, 50, 25), 3: (4, 32, 64)}
SEED = 202611060
TREATMENTS = {
    'soft_rare': ('soft', 1, 1.0), 'hard_rare': ('hard', 1, 1.0),
    'soft_common': ('soft', 3, 1.0), 'hard_common': ('hard', 3, 1.0),
    'soft_balanced': ('soft', 2, 1.0), 'neutral_rare': ('neutral', 1, 1.0),
    'soft_scaled_habitat': ('soft', 1, 2.0), 'hard_scaled_habitat': ('hard', 1, 2.0),
}


def oracle(mode='soft', start=1, scale=1.0):
    """Exact single-generation expectation conditional on the fixed founders."""
    if mode not in MODES or type(start) is not int or start not in STARTS:
        raise ValueError('Unknown mode or founder configuration')
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.01 <= scale <= 10:
        raise ValueError('scale must be finite and in [0.01,10]')
    f = [x/100 for x in STARTS[start]]
    weights = [[scale*x for x in (1, 2, 3)], [3, 2, 1]]
    if mode == 'neutral':
        weights = [[1, 1, 1], [1, 1, 1]]
    means = [sum(x*w for x,w in zip(f, ws)) for ws in weights]
    local = [sum(f[k]*ws[k]*k/2 for k in range(3))/mean
             for ws,mean in zip(weights, means)]
    h0 = 0.5 if mode != 'hard' else means[0]/sum(means)
    q = h0*local[0]+(1-h0)*local[1]
    return {'initial_allele_frequency': f[1]/2+f[2],
            'raw_mean_fitness_by_habitat': means,
            'selected_gamete_frequency_by_habitat': local,
            'habitat0_parent_probability': h0, 'allele_frequency': q,
            'genotype_probabilities_aa_Aa_AA': [(1-q)**2, 2*q*(1-q), q*q]}


def parse_output(text, n, replicates, mode, start, scale):
    expected = oracle(mode, start, scale)
    initial, weights, data = {}, {}, {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT', 'WEIGHT', 'DATA'):
            continue
        if row[0] == 'INIT':
            if len(row) != 6:
                raise ValueError('Malformed founder row')
            rep, habitat, *counts = map(int, row[1:])
            key = (rep, habitat)
            if key in initial:
                raise ValueError('Duplicate founder row')
            initial[key] = tuple(counts)
        elif row[0] == 'WEIGHT':
            if len(row) != 3:
                raise ValueError('Malformed weight row')
            rep, weight = int(row[1]), float(row[2])
            if rep in weights or not math.isfinite(weight) or abs(weight-expected['habitat0_parent_probability']) > 1e-12:
                raise ValueError('Invalid or duplicate realized habitat weight')
            weights[rep] = weight
        else:
            if len(row) != 6:
                raise ValueError('Malformed offspring row')
            rep, *values = map(int, row[1:])
            counts, habitat0_draws = values[:3], values[3]
            if rep in data or any(x < 0 for x in counts) or sum(counts) != n or not 0 <= habitat0_draws <= 2*n:
                raise ValueError('Invalid or duplicate offspring row')
            data[rep] = tuple(values)
    founders = tuple(x*(n//200) for x in STARTS[start])
    if set(initial) != {(r,h) for r in range(replicates) for h in (0,1)} or any(c != founders for c in initial.values()):
        raise ValueError('Missing or incorrect founders')
    if set(weights) != set(range(replicates)) or set(data) != set(range(replicates)):
        raise ValueError('Missing or unexpected replicate')
    return [data[r] for r in range(replicates)]


def run(mode='soft', start=1, scale=1.0, n=200, replicates=512, seed=SEED, executable='slim'):
    oracle(mode, start, scale)
    _integer(n, 'n', 200, 1000)
    if n % 200:
        raise ValueError('n must be divisible by 200')
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for key,value in [('N',n), ('REPLICATES',replicates), ('MODE',MODES[mode]), ('START',start), ('SCALE',scale)]:
        command += ['-d', f'{key}={value}']
    result = subprocess.run([*command, str(HERE/'model.slim')], capture_output=True, text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, replicates, mode, start, scale)


def summarize(rows, n, expected):
    r = len(rows)
    q, h0 = expected['allele_frequency'], expected['habitat0_parent_probability']
    freq = [(row[1]+2*row[2])/(2*n) for row in rows]
    observed = [sum(row[g] for row in rows)/(n*r) for g in range(3)]
    habitat = sum(row[3] for row in rows)/(2*n*r)
    se = math.sqrt(q*(1-q)/(2*n*r))
    checks = [(statistics.mean(freq), q, se),
              (habitat, h0, math.sqrt(h0*(1-h0)/(2*n*r)))]
    checks += [(x,p,math.sqrt(p*(1-p)/(n*r))) for x,p in zip(observed,expected['genotype_probabilities_aa_Aa_AA'])]
    if any(abs(obs-exp) > 6*err+1e-12 for obs,exp,err in checks):
        raise AssertionError('Observation exceeds six-standard-error prediction bound')
    return {'observed_allele_frequency': statistics.mean(freq),
            'exact_standard_error_of_mean_frequency': se,
            'empirical_standard_error_of_mean_frequency': statistics.stdev(freq)/math.sqrt(r) if r > 1 else None,
            'observed_genotype_frequencies_aa_Aa_AA': observed,
            'observed_habitat0_parent_fraction': habitat, 'exact': expected}


def summary():
    treatments = {}
    for i,(name,params) in enumerate(TREATMENTS.items()):
        treatments[name] = summarize(run(*params, seed=SEED+i), 200, oracle(*params))
    return {'experiment': 'Soft versus hard reproductive selection: one generation',
            'slim_version': '4.0.1', 'python_version': platform.python_version(),
            'diploids_per_population': 200, 'founders_per_habitat': 100,
            'replicates_per_treatment': 512, 'synthetic_only': True,
            'seed_by_treatment': {name: SEED+i for i,name in enumerate(TREATMENTS)},
            'treatments': treatments}


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
