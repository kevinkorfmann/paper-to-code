"""Exact offspring probabilities versus a genuine SLiM sexual WF simulation."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform
import subprocess
from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

HERE = Path(__file__).parent
MODES = {'neutral': 0, 'antagonistic': 1, 'concordant': 2}
SEED = 202611040


def gamete_probability(weights):
    """Founder genotype fractions (aa,Aa,AA) = (1/4,1/2,1/4)."""
    if len(weights) != 3 or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Three positive finite fitness weights required')
    scale = max(weights)
    a, h, d = (w/scale for w in weights)
    return (h+d)/(a+2*h+d)


def oracle(mode):
    if mode not in MODES:
        raise ValueError('Unknown mode')
    female = (1,1,1) if mode == 'neutral' else (1,3,9)
    male = (9,3,1) if mode == 'antagonistic' else female
    pm, pf = gamete_probability(male), gamete_probability(female)
    probs = ((1-pm)*(1-pf), pm*(1-pf)+(1-pm)*pf, pm*pf)
    p = (pm+pf)/2
    return {'male_gamete_A': pm, 'female_gamete_A': pf,
            'genotype_probabilities_aa_Aa_AA': list(probs), 'allele_frequency': p,
            'heterozygosity': probs[1], 'hwe_at_expected_frequency': 2*p*(1-p),
            'heterozygote_excess': (pm-pf)**2/2}


def parse_output(text, n, replicates):
    initial, data = {}, {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT', 'DATA'):
            continue
        if row[0] == 'INIT':
            if len(row) != 6:
                raise ValueError('Malformed founder row')
            _, rep, sex, *counts = row
            key = (int(rep), sex)
            if key in initial or sex not in ('M','F'):
                raise ValueError('Duplicate or invalid founder row')
            initial[key] = tuple(map(int, counts))
        else:
            if len(row) != 5:
                raise ValueError('Malformed offspring row')
            rep, *counts = map(int, row[1:])
            if rep in data or any(x < 0 for x in counts) or sum(counts) != n:
                raise ValueError('Invalid or duplicate offspring row')
            data[rep] = tuple(counts)
    if set(initial) != {(r,s) for r in range(replicates) for s in ('M','F')} or any(c != (n//8,n//4,n//8) for c in initial.values()):
        raise ValueError('Missing or incorrect founders')
    if set(data) != set(range(replicates)):
        raise ValueError('Missing or unexpected offspring replicate')
    return [data[r] for r in range(replicates)]


def run(mode, n=200, replicates=512, seed=SEED, executable='slim'):
    oracle(mode)
    _integer(n, 'n', 8, 1000)
    if n % 8:
        raise ValueError('n must be divisible by eight')
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed), '-d', f'N={n}', '-d', f'REPLICATES={replicates}', '-d', f'MODE={MODES[mode]}', str(HERE/'model.slim')]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    return parse_output(result.stdout, n, replicates)


def summary():
    treatments = {}
    for mode in MODES:
        counts = run(mode, seed=SEED+MODES[mode])
        total = 200*len(counts)
        observed = [sum(row[g] for row in counts)/total for g in range(3)]
        expected = oracle(mode)
        # Offspring are independent conditional on these identical, fixed founders.
        for obs, prob in zip(observed, expected['genotype_probabilities_aa_Aa_AA']):
            if abs(obs-prob) > 6*math.sqrt(prob*(1-prob)/total):
                raise AssertionError('Genotype frequency outside six-standard-error bound')
        treatments[mode] = {'observed_genotype_frequencies_aa_Aa_AA': observed,
                            'observed_allele_frequency': observed[2]+observed[1]/2,
                            'observed_heterozygosity': observed[1], 'exact': expected}
    return {'experiment': 'Sexually antagonistic reproductive selection: one generation',
            'slim_version': '4.0.1', 'python_version': platform.python_version(),
            'population_size': 200, 'replicates_per_treatment': 512,
            'seed_by_treatment': {m:SEED+i for m,i in MODES.items()},
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
