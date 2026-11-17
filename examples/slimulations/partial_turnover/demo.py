"""Original partial-turnover experiment and exact one-tick transition law."""
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
SEED = 202611170


def validate(n,k,m):
    _integer(n,'N',2,100)
    _integer(k,'K',0,n)
    _integer(m,'M',0,n)


def distribution(n,k,m):
    """New count K-D+B; independent hypergeometric deaths and binomial births."""
    validate(n,k,m)
    p=k/n
    result=[0.0]*(n+1)
    for d in range(max(0,m-(n-k)),min(m,k)+1):
        pd=math.comb(k,d)*math.comb(n-k,m-d)/math.comb(n,m)
        for b in range(m+1):
            pb=math.comb(m,b)*p**b*(1-p)**(m-b)
            result[k-d+b]+=pd*pb
    return result


def expected_msd(n,k,m):
    validate(n,k,m)
    p=k/n
    return p*(1-p)*(m+m*(n-m)/(n-1))/n**2


def parse(text,n,k,m,replicates):
    validate(n,k,m)
    _integer(replicates,'replicates',1,4096)
    rows={}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0]!='DATA': continue
        if len(row)!=4: raise ValueError('invalid width')
        t,r,c=map(int,row[1:])
        if t not in (0,1) or not 0<=r<replicates or not 0<=c<=n: raise ValueError('invalid row')
        if (t,r) in rows: raise ValueError('duplicate row')
        if t==0 and c!=k: raise ValueError('wrong founder count')
        if t==1 and not max(0,k-m)<=c<=min(n,k+m): raise ValueError('impossible change')
        rows[t,r]=c
    if set(rows)!={(t,r) for t in (0,1) for r in range(replicates)}: raise ValueError('missing rows')
    return [rows[1,r] for r in range(replicates)]


def run(n=40,k=20,m=5,replicates=4096,seed=SEED,executable='slim'):
    validate(n,k,m)
    _integer(replicates,'replicates',1,4096)
    _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for key,value in [('N',n),('K',k),('M',m),('R',replicates)]: cmd+=['-d',f'{key}={value}']
    out=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(out.stdout,n,k,m,replicates)


def summarize(rows,n,k,m):
    validate(n,k,m)
    if len(rows)<2: raise ValueError('need two replicates')
    for c in rows: _integer(c,'count',max(0,k-m),min(n,k+m))
    probs=distribution(n,k,m)
    second=expected_msd(n,k,m)
    fourth=sum(p*((c-k)/n)**4 for c,p in enumerate(probs))
    se=math.sqrt(max(0,fourth-second**2)/len(rows))
    observed=statistics.mean(((c-k)/n)**2 for c in rows)
    mean=statistics.mean(c/n for c in rows)
    if abs(observed-second)>7*se+1e-12: raise AssertionError('MSD mismatch')
    if abs(mean-k/n)>7*math.sqrt(second/len(rows))+1e-12: raise AssertionError('mean mismatch')
    return {'mean_frequency':mean,'observed_msd':observed,'expected_msd':second,'msd_monte_carlo_se':se}


def summary(executable='slim'):
    cases=[('one_replacement',20,1),('five_replacements',20,5),('full_replacement',20,40),
           ('no_replacement',20,0),('absent',0,5),('fixed',40,5)]
    result={'python':platform.python_version(),'slim':check_slim_version(executable),'N':40,
            'replicates':4096,'ticks':1,'treatments':{}}
    for i,(name,k,m) in enumerate(cases):
        result['treatments'][name]={'K':k,'M':m,'seed':SEED+i,
            **summarize(run(k=k,m=m,seed=SEED+i,executable=executable),40,k,m)}
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--slim',default='slim'); ap.add_argument('--check-sample',action='store_true'); args=ap.parse_args()
    text=json.dumps(summary(args.slim),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if text!=(HERE/'sample_summary.json').read_text(): raise SystemExit('sample mismatch')
        print('sample output matches')
    else: print(text,end='')
if __name__=='__main__': main()
