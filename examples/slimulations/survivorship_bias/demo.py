"""Neutral drift with all-replicate and survivor-conditioned summaries."""
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
SEED=202611140

def validate(n,k,t):
    _integer(n,'n',2,100); _integer(k,'k',0,n); _integer(t,'t',0,100)

def distribution(n,k,t):
    validate(n,k,t)
    matrix=[[math.comb(n,j)*(i/n)**j*(1-i/n)**(n-j) for j in range(n+1)] for i in range(n+1)]
    prob=[float(i==k) for i in range(n+1)]
    for _ in range(t):
        prob=[math.fsum(prob[i]*matrix[i][j] for i in range(n+1)) for j in range(n+1)]
    return prob

def moments(prob,n):
    if len(prob)!=n+1 or any(not math.isfinite(x) or x<0 for x in prob) or not math.isclose(sum(prob),1,abs_tol=1e-10):
        raise ValueError('invalid distribution')
    mean=math.fsum(j/n*x for j,x in enumerate(prob))
    second=math.fsum((j/n)**2*x for j,x in enumerate(prob))
    survival=1-prob[0]; segregating=math.fsum(prob[1:n])
    return {'mean':mean,'variance':max(0,second-mean**2),'survival':survival,
            'fixed':prob[n],'segregating':segregating,
            'survivor_mean':mean/survival if survival else None,
            'segregating_mean':(mean-prob[n])/segregating if segregating else None}

def parse(text,n,k,t,r):
    validate(n,k,t); _integer(r,'r',1,4096)
    seen={}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0]!='DATA': continue
        if len(row)!=4: raise ValueError('row width')
        g,i,c=map(int,row[1:])
        if not 0<=g<=t or not 0<=i<r or not 0<=c<=n: raise ValueError('row range')
        if (g,i) in seen: raise ValueError('duplicate row')
        if g==0 and c!=k: raise ValueError('founder count')
        seen[g,i]=c
    if len(seen)!=(t+1)*r: raise ValueError('missing row')
    rows=[[seen[g,i] for i in range(r)] for g in range(t+1)]
    for before,after in zip(rows,rows[1:]):
        if any(a in (0,n) and a!=b for a,b in zip(before,after)): raise ValueError('escaped absorption')
    return rows

def run(n=20,k=2,t=30,r=2048,seed=SEED,executable='slim'):
    validate(n,k,t); _integer(t,'t',1,100); _integer(r,'r',1,4096); _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for name,value in zip(('N','K','G','R'),(n,k,t,r)): cmd+=['-d',f'{name}={value}']
    out=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(out.stdout,n,k,t,r)

def observed(counts,n):
    if not counts: raise ValueError('empty counts')
    for c in counts: _integer(c,'count',0,n)
    alive=[c/n for c in counts if c>0]
    seg=[c/n for c in counts if 0<c<n]
    return {'replicates':len(counts),'survivors':len(alive),'fixed':sum(c==n for c in counts),
            'segregating':len(seg),'mean':statistics.mean(c/n for c in counts),
            'survivor_mean':statistics.mean(alive) if alive else None,
            'segregating_mean':statistics.mean(seg) if seg else None}

def summary(executable='slim'):
    result={'python':platform.python_version(),'slim':check_slim_version(executable),'N':20,'replicates':2048,'treatments':{}}
    for index,(label,k) in enumerate((('rare',2),('common',18),('absent',0),('fixed',20))):
        rows=run(k=k,seed=SEED+index,executable=executable)
        checkpoints={}
        for t in (0,5,15,30):
            pred=moments(distribution(20,k,t),20); obs=observed(rows[t],20)
            se=math.sqrt(pred['variance']/2048)
            if abs(obs['mean']-pred['mean'])>7*se+1e-12: raise AssertionError('mean tolerance')
            pse=math.sqrt(pred['survival']*(1-pred['survival'])/2048)
            if abs(obs['survivors']/2048-pred['survival'])>7*pse+1e-12: raise AssertionError('survival tolerance')
            checkpoints[str(t)]={'expected':pred,'observed':obs,'mean_standard_error':se}
        result['treatments'][label]={'seed':SEED+index,'initial_count':k,'checkpoints':checkpoints}
    return result

def rounded(obj):
    if isinstance(obj,float): return round(obj,10)
    if isinstance(obj,dict): return {k:rounded(v) for k,v in obj.items()}
    if isinstance(obj,list): return [rounded(x) for x in obj]
    return obj

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check-sample',action='store_true'); parser.add_argument('--slim',default='slim')
    args=parser.parse_args(); text=json.dumps(rounded(summary(args.slim)),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if text!=(HERE/'sample_summary.json').read_text(): raise SystemExit('sample mismatch')
        print('sample matches')
    else: print(text,end='')
if __name__=='__main__': main()
