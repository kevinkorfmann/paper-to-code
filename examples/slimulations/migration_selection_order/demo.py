"""Two sequential finite-population stages, with exact transition moments."""
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
SEED = 202611130


def validate(n, k, migration, weight, order):
    _integer(n, 'N', 2, 200)
    _integer(k, 'K', 0, n)
    for value, lo, hi in ((migration, 0, 1), (weight, .1, 10)):
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('invalid migration or weight')
    if type(order) is not str or order not in ('SM', 'MS'):
        raise ValueError('order must be SM or MS')


def select(p, weight):
    return weight*p/(1+(weight-1)*p)


def binomial(n, q):
    return [math.comb(n,k)*q**k*(1-q)**(n-k) for k in range(n+1)]


def prediction(n=100, k=50, migration=.3, weight=2, order='SM'):
    validate(n,k,migration,weight,order)
    p=k/n
    first=select(p,weight) if order=='SM' else (1-migration)*p
    probs=binomial(n,first)
    second=[(1-migration)*j/n if order=='SM' else select(j/n,weight) for j in range(n+1)]
    mean=math.fsum(a*b for a,b in zip(probs,second))
    second_moment=math.fsum(a*(b*b+b*(1-b)/n) for a,b in zip(probs,second))
    deterministic=(1-migration)*select(p,weight) if order=='SM' else select((1-migration)*p,weight)
    return {'expected_first_mean':first,'expected_first_variance':first*(1-first)/n,
            'expected_final_mean':mean,'expected_final_variance':max(0.,second_moment-mean*mean),
            'deterministic_final_frequency':deterministic}


def parse(text,n,k,replicates):
    validate(n,k,0,1,'SM'); _integer(replicates,'replicates',1,2048)
    seen={}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0]!='DATA': continue
        if len(row)!=4: raise ValueError('wrong row width')
        g,r,count=map(int,row[1:])
        if g not in range(3) or not 0<=r<replicates or not 0<=count<=n: raise ValueError('invalid row')
        if (g,r) in seen: raise ValueError('duplicate row')
        if g==0 and count!=k: raise ValueError('wrong founders')
        seen[g,r]=count
    if set(seen)!={(g,r) for g in range(3) for r in range(replicates)}: raise ValueError('missing row')
    return [[seen[g,r] for r in range(replicates)] for g in range(3)]


def run(n=100,k=50,migration=.3,weight=2,order='SM',replicates=1024,seed=SEED,executable='slim'):
    validate(n,k,migration,weight,order)
    _integer(replicates,'replicates',1,2048); _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for key,val in zip(('N','K','M','W','ORDER','REPLICATES'),(n,k,migration,weight,int(order=='MS'),replicates)):
        cmd+=['-d',f'{key}={val}']
    result=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(result.stdout,n,k,replicates)


def summarize(rows,n,k,migration,weight,order):
    pred=prediction(n,k,migration,weight,order)
    if len(rows)!=3 or len(rows[0])<2 or any(len(row)!=len(rows[0]) for row in rows): raise ValueError('bad rows')
    for g,row in enumerate(rows):
        for c in row:
            _integer(c,'count',0,n)
            if g==0 and c!=k: raise ValueError('bad founders')
    obs={}
    for g,label in ((1,'first'),(2,'final')):
        vals=[c/n for c in rows[g]]
        mean=statistics.mean(vals)
        se=math.sqrt(pred[f'expected_{label}_variance']/len(vals))
        if abs(mean-pred[f'expected_{label}_mean'])>7*se+1e-12: raise AssertionError('mean outside Monte Carlo tolerance')
        obs[f'observed_{label}_mean']=mean
        obs[f'observed_{label}_variance']=statistics.variance(vals)
        obs[f'{label}_mean_standard_error']=se
    return {**pred,**obs}


def summary(executable='slim'):
    cases=[('selection_then_migration',.3,2,'SM',50),('migration_then_selection',.3,2,'MS',50),
           ('neutral_SM',.3,1,'SM',50),('neutral_MS',.3,1,'MS',50),
           ('no_migration_SM',0,2,'SM',50),('no_migration_MS',0,2,'MS',50),
           ('complete_migration',1,2,'MS',50),('absent_A',.3,2,'SM',0)]
    answer={'python':platform.python_version(),'slim':check_slim_version(executable),
            'N':100,'replicates':1024,'stages':2,'source_frequency':0,'treatments':{}}
    for i,(name,m,w,order,k) in enumerate(cases):
        answer['treatments'][name]={'seed':SEED+i,'initial_A_individuals':k,'migration':m,'weight_A':w,'order':order,
                                  **summarize(run(k=k,migration=m,weight=w,order=order,seed=SEED+i,executable=executable),100,k,m,w,order)}
    return answer


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--slim',default='slim'); ap.add_argument('--check-sample',action='store_true'); args=ap.parse_args()
    output=json.dumps(summary(args.slim),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if output!=(HERE/'sample_summary.json').read_text(): raise SystemExit('sample mismatch')
        print('sample output matches')
    else: print(output,end='')
if __name__=='__main__': main()
