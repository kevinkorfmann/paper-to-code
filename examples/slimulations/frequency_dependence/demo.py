"""Exact conditional gamete expectations and an original SLiM experiment."""
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
SEED = 202611120


def validate(counts, strength):
    if len(counts) != 3 or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('invalid counts')
    _integer(sum(counts), 'N', 2, 1000)
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('invalid strength')


def prediction(counts, strength):
    validate(counts, strength)
    n = sum(counts)
    p = (counts[1]+2*counts[2])/(2*n)
    a = strength*(1-2*p)
    weights = [1-a, 1, 1+a]
    mean_weight = sum(k*w for k,w in zip(counts,weights))
    q = (counts[1]/2+counts[2]*(1+a))/mean_weight
    return {'initial_frequency':p, 'fitness_aa_Aa_AA':weights,
            'expected_frequency':q, 'expected_variance_frequency':q*(1-q)/(2*n),
            'expected_genotype_frequencies':[(1-q)**2,2*q*(1-q),q*q]}


def parse(text, counts, replicates):
    validate(counts, 0)
    _integer(replicates,'replicates',1,4096)
    observations = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] not in ('INIT','DATA'): continue
        if len(row) != 5: raise ValueError('wrong width')
        r, *c = map(int,row[1:])
        validate(c,0)
        if sum(c) != sum(counts) or not 0 <= r < replicates: raise ValueError('bad population')
        key = (row[0],r)
        if key in observations: raise ValueError('duplicate row')
        if row[0] == 'INIT' and c != list(counts): raise ValueError('wrong founders')
        observations[key]=c
    if set(observations) != {(tag,r) for tag in ('INIT','DATA') for r in range(replicates)}:
        raise ValueError('missing rows')
    return [observations['DATA',r] for r in range(replicates)]


def run(counts=(81,18,1),strength=.8,replicates=1024,seed=SEED,executable='slim'):
    validate(counts,strength)
    _integer(replicates,'replicates',1,4096)
    _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for key,value in zip(('AA','HET','HOM','S','REPLICATES'),(*counts,strength,replicates)):
        cmd += ['-d',f'{key}={value}']
    result=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(result.stdout,counts,replicates)


def summarize(rows,counts,strength):
    pred=prediction(counts,strength)
    n=sum(counts)
    if len(rows)<2: raise ValueError('at least two replicates required')
    for c in rows:
        validate(c,0)
        if sum(c)!=n: raise ValueError('wrong size')
    ps=[(c[1]+2*c[2])/(2*n) for c in rows]
    mean=statistics.mean(ps)
    se=math.sqrt(pred['expected_variance_frequency']/len(rows))
    if abs(mean-pred['expected_frequency'])>7*se+1e-12: raise AssertionError('frequency mismatch')
    for j,q in enumerate(pred['expected_genotype_frequencies']):
        observed=statistics.mean(c[j]/n for c in rows)
        if abs(observed-q)>7*math.sqrt(q*(1-q)/(n*len(rows)))+1e-12:
            raise AssertionError('genotype mismatch')
    return {**pred,'observed_mean_frequency':mean,'observed_variance_frequency':statistics.variance(ps),
            'mean_frequency_standard_error':se,'observed_mean_genotype_frequencies':
            [statistics.mean(c[j]/n for c in rows) for j in range(3)]}


def summary(executable='slim'):
    cases=[('rare_A',(81,18,1),.8),('intermediate_A',(25,50,25),.8),
           ('common_A',(1,18,81),.8),('neutral_rare_A',(81,18,1),0),
           ('absent_A',(100,0,0),.8),('fixed_A',(0,0,100),.8)]
    answer={'python':platform.python_version(),'slim':check_slim_version(executable),'generations':1,'replicates':1024,'treatments':{}}
    for i,(name,counts,s) in enumerate(cases):
        answer['treatments'][name]={'seed':SEED+i,'founders_aa_Aa_AA':counts,'strength':s,
                                   **summarize(run(counts,s,seed=SEED+i,executable=executable),counts,s)}
    return answer


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--slim',default='slim'); ap.add_argument('--check-sample',action='store_true'); args=ap.parse_args()
    result=json.dumps(summary(args.slim),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if result!=(HERE/'sample_summary.json').read_text(): raise SystemExit('sample mismatch')
        print('sample output matches')
    else: print(result,end='')
if __name__=='__main__': main()
