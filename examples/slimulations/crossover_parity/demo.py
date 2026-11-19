"""Actual one-generation SLiM inheritance versus breakpoint-parity oracles."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from examples.slimulations.neutral_drift.demo import _integer,check_slim_version
HERE=Path(__file__).parent
SEED=202611190


def theory(d,r):
    _integer(d,'distance',1,1000)
    if not isinstance(r,(int,float)) or not math.isfinite(r) or not 0<=r<=.5:
        raise ValueError('invalid per-gap rate')
    return {'mean_breakpoints':d*r,'probability_any':-math.expm1(d*math.log1p(-r)),
            'probability_odd':.5 if r==.5 else -.5*math.expm1(d*math.log1p(-2*r))}


def parse(text,n,d):
    _integer(n,'N',2,40000);_integer(d,'D',1,1000)
    lines=[line for line in text.splitlines() if line.startswith('DATA,')]
    if len(lines)!=1: raise ValueError('expected exactly one output row')
    fields=lines[0].split(',')
    if len(fields)!=6:raise ValueError('wrong row width')
    draws,events,positive,odd,recombinant=map(int,fields[1:])
    if draws!=2*n or not 0<=events<=d*draws or not 0<=odd<=positive<=draws or not positive<=events<=d*positive or odd!=recombinant:
        raise ValueError('invalid counts or parity-inheritance mismatch')
    return {'gametes':draws,'breakpoints':events,'any_breakpoint':positive,'odd_breakpoints':odd,'recombinant_markers':recombinant}


def run(d=100,r=.01,n=20000,seed=SEED,executable='slim'):
    theory(d,r);_integer(n,'N',2,40000);_integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed),'-d',f'N={n}','-d',f'D={d}','-d',f'R={r}',str(HERE/'model.slim')]
    result=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60)
    return parse(result.stdout,n,d)


def summarize(counts,d,r):
    expected=theory(d,r);m=counts['gametes']
    observed={'mean_breakpoints':counts['breakpoints']/m,'probability_any':counts['any_breakpoint']/m,'probability_odd':counts['recombinant_markers']/m}
    for key,p in expected.items():
        variance=d*r*(1-r) if key=='mean_breakpoints' else p*(1-p)
        if abs(observed[key]-p)>7*math.sqrt(variance/m)+1e-12:
            raise AssertionError('Monte Carlo mismatch: '+key)
    return {'counts':counts,'expected':expected,'observed':observed}


def summary(executable='slim'):
    cases=[('no_recombination',100,0.0),('one_gap',1,.1),('sparse',100,.001),('one_expected',100,.01),('many_expected',100,.1)]
    out={'slim':check_slim_version(executable),'N':20000,'generations':1,'treatments':{}}
    for i,(name,d,r) in enumerate(cases):
        out['treatments'][name]={'distance':d,'per_gap_rate':r,'seed':SEED+i,**summarize(run(d,r,seed=SEED+i,executable=executable),d,r)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-sample',action='store_true');p.add_argument('--slim',default='slim');a=p.parse_args()
    output=json.dumps(summary(a.slim),indent=2,sort_keys=True)+'\n'
    if a.check_sample:
        if output!=HERE.joinpath('sample_output.json').read_text():raise SystemExit('sample output mismatch')
        print('sample output matches')
    else:print(output,end='')

if __name__=='__main__':main()
