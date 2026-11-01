"""Compare actual sexual WF populations with exact two-sex moment predictions."""

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

SCRIPT = Path(__file__).with_name('sex_ratio.slim')
SAMPLE = Path(__file__).with_name('sample_summary.json')
SEED = 202611010
TREATMENTS = [('balanced', 100, 100), ('few_males', 20, 180), ('few_females', 180, 20)]


def sex_sizes(nm, nf, *, founders=False):
    for name, n in [('nm', nm), ('nf', nf)]:
        _integer(n, name, 1, 1000)
        if founders and n % 4:
            raise ValueError('Sex counts must be multiples of four for exact founders')


def classical_ne(nm, nf):
    """Classical sex-ratio benchmark; not an exact finite-time diversity curve."""
    sex_sizes(nm, nf)
    return Fraction(4*nm*nf, nm+nf)


def moment_path(nm, nf, generations, pm=Fraction(1, 2), pf=Fraction(1, 2)):
    """Exact finite-model recursion; Fraction inputs retain rational arithmetic.

    State = (E[pM], E[pF], E[pM²], E[pF²], E[pM*pF]).
    Conditional offspring dosage is Bernoulli(pM)+Bernoulli(pF).
    """
    sex_sizes(nm, nf)
    _integer(generations, 'generations', 0, 100)
    if any(isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in (pm, pf)):
        raise ValueError('Initial sex frequencies must lie in [0,1]')
    pm, pf = Fraction(pm), Fraction(pf)
    state = (pm, pf, pm*pm, pf*pf, pm*pf)
    path = [state]
    for _ in range(generations):
        m, f, a, b, c = state
        mean = (m+f)/2
        u = (a+b+2*c)/4
        segregating = m+f-a-b
        state = (mean, mean, u+segregating/(4*nm), u+segregating/(4*nf), u)
        path.append(state)
    return path


def predicted(state, nm, nf):
    sex_sizes(nm, nf)
    m, f, a, b, c = state
    w = Fraction(nm, nm+nf)
    census_mean = w*m+(1-w)*f
    census_second = w*w*a+(1-w)**2*b+2*w*(1-w)*c
    role_mean = (m+f)/2
    role_second = (a+b+2*c)/4
    return {'mean_census_p': census_mean, 'mean_parental_role_p': role_mean,
            'census_p_variance': census_second-census_mean**2,
            'parental_role_p_variance': role_second-role_mean**2,
            'mean_census_gene_diversity': 2*(census_mean-census_second)}


def parse_output(text, nm, nf, replicates, generations, opposite):
    records = {}; init = {}; fitness = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0] == 'DATA':
            if len(row) != 8:
                raise ValueError('Malformed count row')
            t, rep, m, f, km, kf, het = map(int, row[1:])
            if ((t, rep) in records or (m, f) != (nm, nf) or
                    not 0 <= km <= 2*nm or not 0 <= kf <= 2*nf or not 0 <= het <= nm+nf):
                raise ValueError('Invalid or duplicate counts')
            copies = km+kf
            if het > min(copies, 2*(nm+nf)-copies) or (copies-het) % 2:
                raise ValueError('Impossible genotype counts')
            records[t, rep] = [km, kf, het]
        elif row[0] == 'INIT':
            if len(row) != 6 or (int(row[1]), row[2]) in init:
                raise ValueError('Invalid founder audit')
            init[int(row[1]), row[2]] = list(map(int, row[3:]))
        elif row[0] == 'FIT':
            fitness.append(row)
    times = sorted({0, 1, generations//4, generations} & set(range(generations+1)))
    if set(records) != {(t, r) for t in times for r in range(replicates)}:
        raise ValueError('Missing or unexpected checkpoint/replicate')
    wanted = {sex: ([n, 0, 0] if sex == 'M' else [0, 0, n]) if opposite
              else [n//4, n//2, n//4] for sex, n in [('M', nm), ('F', nf)]}
    if init != {(r, sex): counts for r in range(replicates) for sex, counts in wanted.items()}:
        raise ValueError('Founder genotype audit failed')
    initial = [0, 2*nf, 0] if opposite else [nm, nf, (nm+nf)//2]
    if any(records[0, r] != initial for r in range(replicates)) or fitness != [['FIT', '1']]:
        raise ValueError('Wrong initial counts or neutral fitness audit')
    if 'Eidos version 3.0.1\n' not in text or 'SLiM version 4.0.1\n' not in text:
        raise ValueError('Unexpected engine version')
    return {t: [records[t, r] for r in range(replicates)] for t in times}


def run_counts(*, nm=100, nf=100, replicates=512, generations=80, opposite=0,
               seed=SEED, executable='slim'):
    sex_sizes(nm, nf, founders=True)
    _integer(replicates, 'replicates', 1, 1024)
    _integer(generations, 'generations', 0, 100)
    _integer(opposite, 'opposite', 0, 1)
    _integer(seed, 'seed', 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for name, value in [('NM', nm), ('NF', nf), ('REPLICATES', replicates),
                        ('GENERATIONS', generations), ('OPPOSITE', opposite)]:
        command += ['-d', f'{name}={value}']
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError('SLiM failed: '+result.stderr[-2000:])
    return parse_output(result.stdout, nm, nf, replicates, generations, opposite)


def summarize(rows, nm, nf):
    census = [(km+kf)/(2*(nm+nf)) for km, kf, _ in rows]
    role = [(km/(2*nm)+kf/(2*nf))/2 for km, kf, _ in rows]
    diversity = [2*p*(1-p) for p in census]
    def se(values):
        return statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else None
    def variance(values):
        return statistics.variance(values) if len(values) > 1 else None
    return {'mean_census_p': statistics.mean(census), 'census_p_monte_carlo_se': se(census),
            'mean_parental_role_p': statistics.mean(role), 'parental_role_p_monte_carlo_se': se(role),
            'census_p_sample_variance': variance(census), 'parental_role_p_sample_variance': variance(role),
            'mean_census_gene_diversity': statistics.mean(diversity),
            'census_gene_diversity_monte_carlo_se': se(diversity),
            'mean_observed_heterozygosity': statistics.mean(het/(nm+nf) for _, _, het in rows),
            'allele_fixed_replicates': sum(p == 1 for p in census),
            'allele_lost_replicates': sum(p == 0 for p in census)}


def build_report(executable='slim', replicates=512, generations=80, seed=SEED, results=None):
    _integer(seed, 'seed', 1, 2**62-4)
    groups = []
    settings = [(name, nm, nf, generations, 0) for name, nm, nf in TREATMENTS]
    settings.append(('opposite_sex_fixation_control', 20, 180, 1, 1))
    for i, (name, nm, nf, g, opposite) in enumerate(settings):
        rows = results[name] if results is not None else run_counts(nm=nm, nf=nf,
               replicates=replicates, generations=g, opposite=opposite, seed=seed+i, executable=executable)
        path = moment_path(nm, nf, g, 0 if opposite else Fraction(1, 2), 1 if opposite else Fraction(1, 2))
        groups.append({'name': name, 'males': nm, 'females': nf, 'seed': seed+i,
                       'classical_sex_ratio_Ne_benchmark': float(classical_ne(nm, nf)),
                       'generations_elapsed': g, 'checkpoints': [
                           {'generation': t, 'observed': summarize(counts, nm, nf),
                            'exact_finite_expectation': {k: float(v) for k, v in predicted(path[t], nm, nf).items()}}
                           for t, counts in rows.items()]})
    return {'example': 'Unequal breeding sex ratios at fixed census size',
            'slim_version': '4.0.1', 'eidos_version': '3.0.1', 'python_version': platform.python_version(),
            'diploid_census_size': 200, 'replicates_per_treatment': replicates, 'treatments': groups,
            'interpretation': 'All replicates are included. Census p weights by sex counts; parental-role p gives each sex half the weight. Gene diversity is 2p(1-p), distinct from observed heterozygosity. Exact finite expectations come from a two-sex moment recursion, not a single-population Ne substitution.'}


def render(report):
    def rounded(x):
        if isinstance(x, float): return round(x, 9)
        if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(report), indent=2, allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim', default='slim')
    parser.add_argument('--replicates', type=int, default=512)
    parser.add_argument('--generations', type=int, default=80)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    try:
        text = render(build_report(args.slim, args.replicates, args.generations, args.seed))
        if args.check_sample and text != SAMPLE.read_text():
            raise ValueError('Output differs from checked sample')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(text, end='')


if __name__ == '__main__':
    main()
