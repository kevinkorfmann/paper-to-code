"""Clonal founder diversity under unequal reproductive contributions."""
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
MODES = {'uniform': 0, 'skew': 1, 'single': 2}
SEED = 202611070


def probabilities(n, mode):
    _integer(n, 'n', 2, 200)
    if mode not in MODES:
        raise ValueError('unknown mode')
    if mode == 'uniform':
        return [1/n]*n
    if mode == 'single':
        return [1.0]+[0.0]*(n-1)
    return [0.5]+[0.5/(n-1)]*(n-1)


def oracle(n, mode):
    p = probabilities(n, mode)
    s2 = math.fsum(x*x for x in p)
    s3 = math.fsum(x**3 for x in p)
    # Pair-match U-statistic variance: overlapping pairs share three draws.
    var_u = (2*s2*(1-s2)+4*(n-2)*(s3-s2*s2))/(n*(n-1))
    return {'pair_same_founder_probability': s2,
            'effective_contributors': 1/s2,
            'expected_diversity': (1-1/n)*(1-s2),
            'variance_diversity': max(0.0, (1-1/n)**2*var_u),
            'expected_richness': math.fsum(1-(1-x)**n for x in p)}


def parse_output(text, n, replicates, mode):
    probs = probabilities(n, mode)
    init, data = {}, {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT','DATA'):
            continue
        if len(row) != n+2:
            raise ValueError('wrong row width')
        r = int(row[1])
        dest = init if row[0] == 'INIT' else data
        if r in dest:
            raise ValueError('duplicate row')
        values = list(map(float if row[0] == 'INIT' else int, row[2:]))
        if row[0] == 'INIT':
            if any(not math.isfinite(x) or abs(x-y)>1e-12 for x,y in zip(values,probs)):
                raise ValueError('wrong realized weights')
        elif any(x<0 for x in values) or sum(values)!=n:
            raise ValueError('wrong founder counts')
        dest[r] = values
    if set(init) != set(range(replicates)) or set(data) != set(range(replicates)):
        raise ValueError('missing or unexpected replicate')
    return [data[r] for r in range(replicates)]


def run(mode='uniform', n=100, replicates=512, seed=SEED, scale=1.0, executable='slim'):
    probabilities(n, mode)
    _integer(replicates, 'replicates', 1, 1024)
    _integer(seed, 'seed', 1, 2**62-1)
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0.01<=scale<=10:
        raise ValueError('invalid scale')
    check_slim_version(executable)
    command = [executable, '-s', str(seed)]
    for key,value in [('N',n),('REPLICATES',replicates),('MODE',MODES[mode]),('SCALE',scale)]:
        command += ['-d',f'{key}={value}']
    result = subprocess.run([*command,str(HERE/'model.slim')], capture_output=True,text=True,check=True,timeout=120)
    return parse_output(result.stdout,n,replicates,mode)


def summarize(rows, n, mode):
    expected = oracle(n,mode)
    values = [1-sum(c*c for c in row)/n**2 for row in rows]
    mean = statistics.mean(values)
    se = math.sqrt(expected['variance_diversity']/len(rows))
    if abs(mean-expected['expected_diversity']) > 6*se+1e-12:
        raise AssertionError('diversity outside six exact standard errors')
    return {'mean_diversity':mean,
            'mean_founder_richness':statistics.mean(sum(c>0 for c in row) for row in rows),
            'mean_founder0_fraction':statistics.mean(row[0]/n for row in rows),
            'exact_standard_error_of_mean_diversity':se,
            'exact':expected}


def summary():
    treatments = {}
    for i,mode in enumerate(MODES):
        treatments[mode] = summarize(run(mode,seed=SEED+i),100,mode)
    return {'experiment':'One-generation clonal founder-contribution skew',
            'slim_version':'4.0.1','python_version':platform.python_version(),
            'individuals_per_population':100,'replicates_per_treatment':512,
            'seed_by_treatment':{mode:SEED+i for i,mode in enumerate(MODES)},
            'synthetic_only':True,'treatments':treatments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    result=summary()
    if args.check_sample:
        if result != json.loads((HERE/'sample_summary.json').read_text()):
            raise AssertionError('checked output differs')
        print('Checked output reproduced exactly')
    else:
        print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
