"""Actual SLiM bottlenecks, exact finite-state predictions and census recovery."""
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

SCRIPT = Path(__file__).with_name('bottleneck.slim')
SAMPLE = Path(__file__).with_name('sample_summary.json')
SEED = 202611020
SCHEDULES = {'constant': [100]*40, 'early_bottleneck': [5]*5+[100]*35,
             'late_bottleneck': [100]*34+[5]*5+[100]}


def validate(n0, sizes, initial_copies):
    _integer(n0, 'n0', 2, 200)
    _integer(initial_copies, 'initial_copies', 0, 2*n0)
    if not sizes or len(sizes)>100:
        raise ValueError('Schedule must have 1..100 generations')
    for n in sizes:
        _integer(n, 'size', 2, 200)


def binomial_row(copies, p):
    """Stable binomial probabilities, recursing outward from the mode."""
    _integer(copies, 'copies', 1, 400)
    if isinstance(p, bool) or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError('Invalid probability')
    row = [0.]*(copies+1)
    if p in (0,1):
        row[int(p)*copies] = 1.
        return row
    mode = min(copies, math.floor((copies+1)*p))
    row[mode] = 1.
    for k in range(mode, 0, -1):
        row[k-1] = row[k]*k/(copies-k+1)*(1-p)/p
    for k in range(mode, copies):
        row[k+1] = row[k]*(copies-k)/(k+1)*p/(1-p)
    total = math.fsum(row)
    return [x/total for x in row]


@lru_cache(maxsize=16)
def transition(old_n, new_n):
    return tuple(tuple(binomial_row(2*new_n, k/(2*old_n))) for k in range(2*old_n+1))


def expected_path(n0, sizes, initial_copies):
    """Exact finite Markov model, evaluated numerically; no diffusion approximation."""
    validate(n0, sizes, initial_copies)
    p0 = initial_copies/(2*n0)
    mass = [0.]*(2*n0+1)
    mass[initial_copies] = 1.
    old_n = n0
    h = 2*p0*(1-p0)
    results = []
    for t in range(len(sizes)+1):
        if t:
            new_n = sizes[t-1]
            matrix = transition(old_n, new_n)
            mass = [math.fsum(prob*row[j] for prob,row in zip(mass,matrix)) for j in range(2*new_n+1)]
            old_n = new_n
            h *= 1-1/(2*new_n)
        n = old_n
        results.append({'generation':t, 'diploids':n,
            'mean_p':math.fsum(prob*k/(2*n) for k,prob in enumerate(mass)),
            'mean_gene_diversity':math.fsum(prob*2*(k/(2*n))*(1-k/(2*n)) for k,prob in enumerate(mass)),
            'diversity_product_formula':h, 'loss_probability':mass[0],
            'fixation_probability':mass[-1], 'probability_mass':math.fsum(mass)})
    return results


def parse_output(text, n0, sizes, initial_copies, replicates):
    records = {}; founders = {}; fitness = []
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        if row[0]=='DATA':
            if len(row)!=6: raise ValueError('Malformed DATA')
            t,r,n,k,het = map(int,row[1:])
            if not 0<=t<=len(sizes) or not 0<=r<replicates:
                raise ValueError('Unexpected time or replicate')
            expected_n = n0 if t==0 else sizes[t-1]
            if (t,r) in records or n!=expected_n or not 0<=k<=2*n or not 0<=het<=min(k,2*n-k) or (k-het)%2:
                raise ValueError('Invalid counts, size or duplicate')
            records[t,r] = (k,het)
        elif row[0]=='INIT':
            if len(row)!=5 or int(row[1]) in founders: raise ValueError('Invalid founder record')
            founders[int(row[1])] = list(map(int,row[2:]))
        elif row[0]=='FIT': fitness.append(row)
    if set(records)!={(t,r) for t in range(len(sizes)+1) for r in range(replicates)}:
        raise ValueError('Missing records')
    if set(founders)!=set(range(replicates)):
        raise ValueError('Missing founders')
    for r, counts in founders.items():
        if any(x<0 for x in counts) or sum(counts)!=n0 or counts[1]+2*counts[2]!=initial_copies:
            raise ValueError('Wrong founder genotypes')
        if records[0,r]!=(initial_copies,counts[1]): raise ValueError('Wrong initial count')
    if fitness!=[['FIT','1']] or 'Eidos version 3.0.1\n' not in text or 'SLiM version 4.0.1\n' not in text:
        raise ValueError('Missing neutrality/version audit')
    for r in range(replicates):
        for t in range(len(sizes)):
            n = n0 if t==0 else sizes[t-1]
            k = records[t,r][0]
            if k==0 and records[t+1,r][0]!=0: raise ValueError('Lost allele reappeared')
            if k==2*n and records[t+1,r][0]!=2*sizes[t]: raise ValueError('Fixation reversed')
    return {t:[records[t,r] for r in range(replicates)] for t in range(len(sizes)+1)}


def run_counts(sizes, *, n0=100, initial_copies=20, replicates=1024, seed=SEED, executable='slim'):
    validate(n0,sizes,initial_copies)
    _integer(replicates,'replicates',1,2048)
    _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    command = [executable,'-s',str(seed)]
    for name,value in [('N0',n0),('INITIAL_COPIES',initial_copies),('REPLICATES',replicates),
                       ('SIZES','c('+','.join(map(str,sizes))+')')]:
        command += ['-d',f'{name}={value}']
    result = subprocess.run([*command,str(SCRIPT)],capture_output=True,text=True,timeout=120)
    if result.returncode: raise ValueError('SLiM failed: '+result.stderr[-2000:])
    return parse_output(result.stdout,n0,sizes,initial_copies,replicates)


def summarize(rows,n):
    p = [k/(2*n) for k,_ in rows]
    h = [2*x*(1-x) for x in p]
    def se(values):
        return statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    lost = [int(k==0) for k,_ in rows]
    return {'mean_p':statistics.mean(p),'mean_p_monte_carlo_se':se(p),
            'mean_gene_diversity':statistics.mean(h),'gene_diversity_monte_carlo_se':se(h),
            'mean_observed_heterozygosity':statistics.mean(het/n for _,het in rows),
            'lost_replicates':sum(lost),'loss_fraction':statistics.mean(lost),'loss_fraction_monte_carlo_se':se(lost),
            'fixed_replicates':sum(k==2*n for k,_ in rows),
            'segregating_replicates':sum(0<k<2*n for k,_ in rows)}


def build_report(executable='slim',replicates=1024,seed=SEED,results=None):
    _integer(seed,'seed',1,2**62-3)
    groups = []
    for i,(name,sizes) in enumerate(SCHEDULES.items()):
        counts = results[name] if results is not None else run_counts(sizes,replicates=replicates,seed=seed+i,executable=executable)
        expected = expected_path(100,sizes,20)
        checkpoints = [0,1,5,6,34,35,39,40]
        groups.append({'name':name,'seed':seed+i,'offspring_size_schedule':sizes,
            'checkpoints':[{'generation':t,'diploids':expected[t]['diploids'],
                            'observed':summarize(counts[t],expected[t]['diploids']),
                            'finite_model':expected[t]} for t in checkpoints],
            'alleles_lost_by_generation_5_still_lost_at_40':sum(counts[5][r][0]==0 and counts[40][r][0]==0 for r in range(replicates))})
    return {'example':'Bottleneck recovery: census size and retained variation differ',
            'slim_version':'4.0.1','eidos_version':'3.0.1','python_version':platform.python_version(),
            'founder_diploids':100,'initial_copies':20,'initial_frequency':.1,
            'replicates_per_treatment':replicates,'treatments':groups,
            'interpretation':'All replicates including absorption are counted. H=2p(1-p) is gene diversity, not observed heterozygosity. Finite-state probabilities are numerical evaluations of the exact variable-size binomial model; no mutation, migration or selection.'}


def render(report):
    def rounded(x):
        if isinstance(x,float): return round(x,9)
        if isinstance(x,dict): return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(report),indent=2,allow_nan=False)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slim',default='slim')
    parser.add_argument('--replicates',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    try:
        text=render(build_report(args.slim,args.replicates,args.seed))
        if args.check_sample and text!=SAMPLE.read_text(): raise ValueError('Output differs from checked sample')
    except (ValueError,OSError,subprocess.SubprocessError) as error: parser.error(str(error))
    print(text,end='')


if __name__=='__main__': main()
