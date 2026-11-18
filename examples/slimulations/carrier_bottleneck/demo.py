"""Adult survival samples diploids, not independent chromosome copies."""
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
SEED = 202611180


def validate(n, het, hom, survivors):
    _integer(n, 'N', 2, 100)
    _integer(het, 'heterozygotes', 0, n)
    _integer(hom, 'homozygotes', 0, n-het)
    _integer(survivors, 'survivors', 1, n)


def genotype_distribution(n, het, hom, survivors):
    """Multivariate hypergeometric law for sampled genotype counts."""
    validate(n, het, hom, survivors)
    result = {}
    for a in range(min(het, survivors)+1):
        for b in range(min(hom, survivors-a)+1):
            c = survivors-a-b
            if c <= n-het-hom:
                result[a, b] = math.comb(het,a)*math.comb(hom,b)*math.comb(n-het-hom,c)/math.comb(n,survivors)
    return result


def loss_probability(n, het, hom, survivors):
    validate(n, het, hom, survivors)
    noncarriers=n-het-hom
    return math.comb(noncarriers,survivors)/math.comb(n,survivors) if survivors<=noncarriers else 0.0


def frequency_variance(n, het, hom, survivors):
    validate(n, het, hom, survivors)
    mean_dosage=(het+2*hom)/n
    dosage_variance=(het+4*hom)/n-mean_dosage**2
    return max(0.0,dosage_variance)*(n-survivors)/(4*survivors*(n-1))


def parse(text,n,het,hom,survivors,replicates):
    validate(n,het,hom,survivors)
    _integer(replicates,'replicates',1,4096)
    support=genotype_distribution(n,het,hom,survivors)
    rows={}
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0]!='DATA': continue
        if len(row)!=5: raise ValueError('invalid width')
        t,r,a,b=map(int,row[1:])
        if t not in (0,1) or not 0<=r<replicates: raise ValueError('invalid index')
        if (t,r) in rows: raise ValueError('duplicate row')
        if t==0 and (a,b)!=(het,hom): raise ValueError('wrong founders')
        if t==1 and (a,b) not in support: raise ValueError('impossible survivors')
        rows[t,r]=(a,b)
    if set(rows)!={(t,r) for t in (0,1) for r in range(replicates)}: raise ValueError('missing rows')
    return [rows[1,r] for r in range(replicates)]


def run(n=40,het=8,hom=0,survivors=5,replicates=4096,seed=SEED,executable='slim'):
    validate(n,het,hom,survivors)
    _integer(replicates,'replicates',1,4096)
    _integer(seed,'seed',1,2**62-1)
    check_slim_version(executable)
    cmd=[executable,'-s',str(seed)]
    for key,value in [('N',n),('HET',het),('HOM',hom),('S',survivors),('R',replicates)]: cmd+=['-d',f'{key}={value}']
    out=subprocess.run(cmd+[str(HERE/'model.slim')],capture_output=True,text=True,check=True,timeout=120)
    return parse(out.stdout,n,het,hom,survivors,replicates)


def summarize(rows,n,het,hom,survivors):
    probs=genotype_distribution(n,het,hom,survivors)
    if len(rows)<2: raise ValueError('need two replicates')
    for row in rows:
        if row not in probs: raise ValueError('impossible survivors')
    expected=(het+2*hom)/(2*n)
    variance=frequency_variance(n,het,hom,survivors)
    loss=loss_probability(n,het,hom,survivors)
    observed_loss=sum(a+b==0 for a,b in rows)/len(rows)
    observed_mean=statistics.mean((a+2*b)/(2*survivors) for a,b in rows)
    se=math.sqrt(loss*(1-loss)/len(rows))
    if abs(observed_loss-loss)>7*se+1e-12: raise AssertionError('loss check failed')
    if abs(observed_mean-expected)>7*math.sqrt(variance/len(rows))+1e-12: raise AssertionError('mean check failed')
    return dict(expected_frequency=expected,mean_frequency=observed_mean,losses=sum(a+b==0 for a,b in rows),observed_loss=observed_loss,expected_loss=loss,loss_monte_carlo_se=se,expected_frequency_variance=variance)


def experiment(executable='slim'):
    treatments={}
    for i,(name,het,hom,s) in enumerate([('spread',8,0,5),('clustered',0,4,5),('mixed',4,2,5),('no_bottleneck',8,0,40),('absent',0,0,5),('fixed',0,40,5)]):
        seed=SEED+i
        rows=run(het=het,hom=hom,survivors=s,seed=seed,executable=executable)
        treatments[name]=dict(heterozygotes=het,homozygotes=hom,survivors=s,seed=seed,**summarize(rows,40,het,hom,s))
    return dict(slim_version='4.0.1',python_version=platform.python_version(),population_size=40,replicates=4096,treatments=treatments)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--slim',default='slim')
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    result=experiment(args.slim)
    if args.check_sample:
        if result!=json.loads((HERE/'sample_summary.json').read_text()): raise AssertionError('saved output differs')
        print('Saved output matches actual seeded SLiM runs.')
    else: print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
