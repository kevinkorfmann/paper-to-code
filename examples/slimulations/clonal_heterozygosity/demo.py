"""Test actual SLiM genotype counts against exact finite-population moments."""
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
SEED = 202611110
TREATMENTS = {
    'clonal_mixed_founders': (1.0, 20, 0),
    'half_clonal_mixed_founders': (0.5, 20, 0),
    'sexual_mixed_founders': (0.0, 20, 0),
    'clonal_all_heterozygotes': (1.0, 40, 0),
    'sexual_all_heterozygotes': (0.0, 40, 0),
}


def validate(n, het, hom, clone_rate, steps):
    _integer(n, 'n', 2, 200)
    _integer(het, 'het', 0, n)
    _integer(hom, 'hom', 0, n-het)
    _integer(steps, 'steps', 0, 400)
    if (type(clone_rate) not in (int, float) or not math.isfinite(clone_rate)
            or not 0 <= clone_rate <= 1):
        raise ValueError('clone_rate must be a finite probability')


def validate_counts(counts, n):
    if (len(counts) != 3 or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != n):
        raise ValueError('invalid genotype counts')


def offspring_probabilities(counts, clone_rate):
    """Order: aa, Aa, AA. Sexual parents are independent, with replacement."""
    n = sum(counts)
    validate_counts(counts, n)
    validate(n, counts[1], counts[2], clone_rate, 0)
    p = (counts[1]+2*counts[2])/(2*n)
    sexual = [(1-p)**2, 2*p*(1-p), p*p]
    return [clone_rate*k/n+(1-clone_rate)*s for k, s in zip(counts, sexual)]


def expected_means(n=40, het=20, hom=0, clone_rate=1.0, steps=320):
    """Closed two-moment recursion, exact for this finite multinomial model."""
    validate(n, het, hom, clone_rate, steps)
    p = (het+2*hom)/(2*n)
    g, h = 2*p*(1-p), het/n
    for _ in range(steps):
        next_h = clone_rate*h+(1-clone_rate)*g
        g = (1-1/n)*g+next_h/(2*n)
        h = next_h
    answer = {'frequency': p, 'gene_diversity': g,
              'observed_heterozygosity': h,
              'variance_frequency': max(0.0, p*(1-p)-g/2)}
    if clone_rate == 1:
        initial_genotype_diversity = 1-sum((k/n)**2 for k in (n-het-hom, het, hom))
        answer['genotype_diversity'] = initial_genotype_diversity*(1-1/n)**steps
    return answer


def checkpoints(steps):
    return sorted({0, 1, steps//4, steps//2, steps})


def parse_output(text, n, het, hom, clone_rate, steps, replicates):
    validate(n, het, hom, clone_rate, steps)
    _integer(steps, 'steps', 1, 400)
    _integer(replicates, 'replicates', 1, 2048)
    observed = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] != 'DATA':
            continue
        if len(row) != 6:
            raise ValueError('wrong output row width')
        t, r, *counts = map(int, row[1:])
        validate_counts(counts, n)
        key = (t, r)
        if key in observed:
            raise ValueError('duplicate observation')
        if t == 0 and counts != [n-het-hom, het, hom]:
            raise ValueError('wrong founders')
        if clone_rate == 1 and any(a == 0 and b != 0 for a, b in zip((n-het-hom, het, hom), counts)):
            raise ValueError('cloning created an absent genotype')
        observed[key] = counts
    wanted = {(t, r) for t in checkpoints(steps) for r in range(replicates)}
    if set(observed) != wanted:
        raise ValueError('missing or unexpected observations')
    return {t: [observed[t, r] for r in range(replicates)] for t in checkpoints(steps)}


def run(n=40, het=20, hom=0, clone_rate=1.0, steps=320,
        replicates=512, seed=SEED, executable='slim'):
    validate(n, het, hom, clone_rate, steps)
    _integer(steps, 'steps', 1, 400)
    _integer(replicates, 'replicates', 1, 2048)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for name, value in [('N', n), ('HET', het), ('HOM', hom), ('C', float(clone_rate)),
                        ('STEPS', steps), ('REPLICATES', replicates)]:
        command += ['-d', f'{name}={value}']
    result = subprocess.run([*command, str(HERE/'model.slim')], capture_output=True,
                            text=True, check=True, timeout=180)
    return parse_output(result.stdout, n, het, hom, clone_rate, steps, replicates)


def summarize(rows, n, het, hom, clone_rate, steps):
    expected = expected_means(n, het, hom, clone_rate, steps)
    if len(rows) < 2:
        raise ValueError('at least two replicates required')
    for row in rows:
        validate_counts(row, n)
    ps = [(row[1]+2*row[2])/(2*n) for row in rows]
    values = {'frequency': ps, 'gene_diversity': [2*p*(1-p) for p in ps],
              'observed_heterozygosity': [row[1]/n for row in rows],
              'genotype_diversity': [1-sum((k/n)**2 for k in row) for row in rows]}
    result = {'exact_expectations': expected, 'replicates': len(rows)}
    for key, xs in values.items():
        mean = statistics.mean(xs)
        se = statistics.stdev(xs)/math.sqrt(len(xs))
        # Empirical Monte Carlo error is descriptive, not a proof of the oracle.
        # The 1/R floor handles near-absorbed, rare-event genotype diversity.
        tolerance = 6*se+1/len(rows)
        if key in expected and abs(mean-expected[key]) > tolerance:
            raise AssertionError(key+' differs from the exact expectation')
        result[key] = {'mean': mean, 'monte_carlo_standard_error': se}
    result['exact_standard_error_mean_frequency'] = math.sqrt(expected['variance_frequency']/len(rows))
    result['all_aa'] = sum(row[0] == n for row in rows)
    result['all_Aa'] = sum(row[1] == n for row in rows)
    result['all_AA'] = sum(row[2] == n for row in rows)
    result['genotype_monomorphic'] = result['all_aa']+result['all_Aa']+result['all_AA']
    result['allelic_polymorphic'] = sum(0 < p < 1 for p in ps)
    return result


def summary():
    result = {'experiment': 'Clonal genotype fixation need not be allele fixation',
              'slim_version': check_slim_version('slim'),
              'python_version': platform.python_version(), 'synthetic_only': True,
              'individuals_per_population': 40, 'replicates_per_treatment': 512,
              'generations': 320, 'treatments': {}}
    for i, (name, (rate, het, hom)) in enumerate(TREATMENTS.items()):
        observed = run(het=het, hom=hom, clone_rate=rate, seed=SEED+i)
        result['treatments'][name] = {
            'seed': SEED+i, 'cloning_probability': rate,
            'initial_counts_aa_Aa_AA': [40-het-hom, het, hom],
            'checkpoints': {str(t): summarize(rows, 40, het, hom, rate, t)
                            for t, rows in observed.items()}}
    return result


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


if __name__ == '__main__':
    main()
