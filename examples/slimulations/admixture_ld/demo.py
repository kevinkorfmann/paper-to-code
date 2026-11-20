"""One-generation source mixing and finite-sample signed haplotype LD."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from examples.slimulations.neutral_drift.demo import _integer,check_slim_version
HERE=Path(__file__).parent
SEED=202611200


def validate(n,m,phase):
    _integer(n,'N',2,100)
    if not math.isfinite(m) or not 0<=m<=1 or type(phase) is not int or phase not in (-1,1):raise ValueError('invalid mixture or phase')


def moments(n,m,phase):
    validate(n,m,phase)
    mean=phase*(1-1/n)*m*(1-m)
    variance=sum(math.comb(n,k)*m**k*(1-m)**(n-k)*(phase*(k/n)*(1-k/n)-mean)**2 for k in range(n+1))
    return mean,variance


def ld(counts):
    if len(counts)!=4 or any(type(c) is not int or c<0 for c in counts) or not sum(counts):raise ValueError('invalid haplotypes')
    ab,a_b,ab_,none=[c/sum(counts) for c in counts]
    return ab-(ab+a_b)*(ab+ab_)


def parse(text,n,m,phase,replicates):
    validate(n,m,phase);_integer(replicates,'replicates',2,1024)
    rows={}
    for line in text.splitlines():
        if not line.startswith('DATA,'):continue
        row=line.split(',')
        if len(row)!=7:raise ValueError('invalid width')
        r,j,*counts=map(int,row[1:])
        if not 0<=r<replicates or not 0<=j<=2 or (r,j) in rows or sum(counts)!=2*n or any(c<0 or c%2 for c in counts):raise ValueError('invalid row')
        if j==0 and counts!=([2*n,0,0,0] if phase==1 else [0,2*n,0,0]):raise ValueError('wrong source A')
        if j==1 and counts!=([0,0,0,2*n] if phase==1 else [0,0,2*n,0]):raise ValueError('wrong source B')
        if j==2 and (sum(counts[1:3]) if phase==1 else counts[0]+counts[3]):raise ValueError('unexpected target phase')
        rows[r,j]=counts
    if len(rows)!=3*replicates:raise ValueError('missing rows')
    return [rows[r,2] for r in range(replicates)]


def run(n=40,m=.5,phase=1,replicates=1024,seed=SEED,executable='slim'):
    validate(n,m,phase);_integer(replicates,'replicates',2,1024);_integer(seed,'seed',1,2**62-1);check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for key,v in [('N',n),('M',m),('PHASE',phase),('R',replicates)]:cmd+=['-d',f'{key}={v}']
    out=subprocess.run(cmd+[str(HERE/'model.slim')],text=True,capture_output=True,check=True,timeout=120)
    return parse(out.stdout,n,m,phase,replicates)


def summarize(rows,n,m,phase):
    expected,var=moments(n,m,phase);observed=statistics.mean(map(ld,rows));se=math.sqrt(var/len(rows))
    if abs(observed-expected)>7*se+1e-12:raise AssertionError('LD mean mismatch')
    return {'mean_ld':observed,'expected_plugin_mean_ld':expected,'infinite_mixture_ld':phase*m*(1-m),'mean_ld_monte_carlo_se':se,'first_target_haplotype_counts':rows[0]}


def summary(executable='slim'):
    out={'slim':check_slim_version(executable),'N':40,'replicates':1024,'haplotype_order':['AB','Ab','aB','ab'],'treatments':{}}
    cases=[('quarter',.25,1),('half',.5,1),('phase_reversed',.5,-1),('source_a_only',1.,1),('source_b_only',0.,1)]
    for i,(name,m,p) in enumerate(cases):out['treatments'][name]={'m':m,'phase':p,'seed':SEED+i,**summarize(run(m=m,phase=p,seed=SEED+i,executable=executable),40,m,p)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-sample',action='store_true');p.add_argument('--slim',default='slim');a=p.parse_args();s=json.dumps(summary(a.slim),sort_keys=True,indent=2)+'\n'
    if a.check_sample:
        if s!=HERE.joinpath('sample_output.json').read_text():raise SystemExit('sample mismatch')
        print('sample output matches')
    else:print(s,end='')

if __name__=='__main__':main()
