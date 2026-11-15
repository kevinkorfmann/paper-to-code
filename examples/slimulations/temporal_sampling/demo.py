"""Separate one-generation neutral drift from nondestructive observation noise."""
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
HERE=Path(__file__).parent
SEED=202611150

def validate(n,k,s):
    _integer(n,'n',2,100); _integer(k,'k',0,n); _integer(s,'s',1,n)

def hypergeom(n,k,s):
    validate(n,k,s)
    return [math.comb(k,j)*math.comb(n-k,s-j)/math.comb(n,s) if 0<=j<=k and 0<=s-j<=n-k else 0.0 for j in range(s+1)]

def delta_distribution(n,k,s):
    """Independent enumeration over offspring census and two observation samples."""
    validate(n,k,s); p=k/n; first=hypergeom(n,k,s); out=[0.0]*(2*s+1)
    for c in range(n+1):
        w=math.comb(n,c)*p**c*(1-p)**(n-c)
        for a,pa in enumerate(first):
            for b,pb in enumerate(hypergeom(n,c,s)):
                out[b-a+s]+=w*pa*pb
    return out

def theory(n,k,s):
    validate(n,k,s); p=k/n
    drift=p*(1-p)/n
    noise=p*(1-p)*(n-s)/(s*(n-1))*(2-1/n)
    return {'drift_msd':drift,'sampling_msd':noise,'observed_msd':drift+noise}

def parse(text,n,k,s,r):
    validate(n,k,s); _integer(r,'r',1,4096); seen={}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0]!='DATA': continue
        if len(row)!=5: raise ValueError('row width')
        t,i,c,a=map(int,row[1:])
        if t not in (0,1) or not 0<=i<r or not 0<=c<=n or not 0<=a<=s: raise ValueError('range')
        if not max(0,s-(n-c))<=a<=min(s,c): raise ValueError('impossible sample')
        if (t,i) in seen: raise ValueError('duplicate')
        if t==0 and c!=k: raise ValueError('founders')
        seen[t,i]=(c,a)
    if len(seen)!=2*r: raise ValueError('missing')
    return [(seen[0,i],seen[1,i]) for i in range(r)]

def run(n=40,k=20,s=5,r=4096,seed=SEED,executable='slim'):
    validate(n,k,s); _integer(r,'r',1,4096); _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable); cmd=[executable,'-s',str(seed)]
    for name,value in zip(('N','K','S','R'),(n,k,s,r)): cmd+=['-d',f'{name}={value}']
    out=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(out.stdout,n,k,s,r)

def summary(executable='slim'):
    result={'python':platform.python_version(),'slim':check_slim_version(executable),'N':40,'replicates':4096,'treatments':{}}
    for index,(label,k,s) in enumerate((('small_sample',20,5),('half_census',20,20),('full_census',20,40),('absent',0,5),('fixed',40,5))):
        rows=run(k=k,s=s,seed=SEED+index,executable=executable)
        ds=[(b[1]-a[1])/s for a,b in rows]; dc=[(b[0]-a[0])/40 for a,b in rows]
        pred=theory(40,k,s); dist=delta_distribution(40,k,s)
        second=sum(((j-s)/s)**2*w for j,w in enumerate(dist))
        fourth=sum(((j-s)/s)**4*w for j,w in enumerate(dist))
        se=math.sqrt(max(0,fourth-second*second)/len(rows))
        obs=statistics.mean(d*d for d in ds)
        if abs(obs-pred['observed_msd'])>7*se+1e-12: raise AssertionError('MSD tolerance')
        result['treatments'][label]={'seed':SEED+index,'initial_count':k,'sample_size':s,'expected':pred,'observed_msd':obs,'census_msd':statistics.mean(d*d for d in dc),'mean_sample_change':statistics.mean(ds),'msd_mc_se':se}
    return result

def rounded(obj):
    if isinstance(obj,float): return round(obj,10)
    if isinstance(obj,dict): return {k:rounded(v) for k,v in obj.items()}
    return obj

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check-sample',action='store_true'); parser.add_argument('--slim',default='slim')
    args=parser.parse_args(); text=json.dumps(rounded(summary(args.slim)),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if text!=(HERE/'sample_summary.json').read_text(): raise SystemExit('sample mismatch')
        print('sample matches')
    else: print(text,end='')
if __name__=='__main__': main()
