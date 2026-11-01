"""Call real BCFtools/RoH on synthetic genotype likelihoods and check independent controls."""

import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile

from .install import BINARY, check_version
from .reference import emissions, infer, phred_error

SEED=20261101
FIXTURE=Path(__file__).with_name('synthetic_input.json')
SAMPLE=Path(__file__).with_name('sample_summary.json')
SWITCH=.02


def generate_fixture():
    rng=random.Random(SEED);n=180;samples={}
    for name in ('tract','outbred','all_autozygous'):
        states=[int(name=='all_autozygous' or (name=='tract' and 60<=i<120)) for i in range(n)]
        genotypes=[]
        for state in states:
            first=int(rng.random()<.5)
            genotypes.append(2*first if state else first+int(rng.random()<.5))
        pls=[[30*abs(g-c) for c in range(3)] for g in genotypes]
        samples[name]={'genotypes':genotypes,'PL':pls,'truth_states':states}
    return {'seed':SEED,'marker_positions':list(range(1,n+1)),
            'ALT_frequencies':[.5]*n,'samples':samples,
            'construction':'Independent alleles in HW, one allele copied twice in AZ; constructed PLs, not simulated sequencing reads.'}


def vcf_text(pl_by_sample,frequencies):
    if not pl_by_sample or not frequencies:raise ValueError('Empty data')
    n=len(frequencies)
    for name,rows in pl_by_sample.items():
        if not name or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in name):raise ValueError('Invalid synthetic sample ID')
        if len(rows)!=n:raise ValueError('Wrong number of PL rows')
        for pl,f in zip(rows,frequencies):
            emissions(pl,f)
            if len(set(pl))==1:raise ValueError('Flat PLs are skipped by this tool; use informative rows')
    header=['##fileformat=VCFv4.2',f'##contig=<ID=1,length={n}>',
            '##INFO=<ID=AF,Number=A,Type=Float,Description="Supplied population ALT frequency">',
            '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Constructed genotype likelihoods">',
            '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(pl_by_sample)]
    rows=['\t'.join(['1',str(i+1),f'm{i+1}','A','C','.','PASS',f'AF={frequencies[i]:.9g}','PL',
                     *[','.join(map(str,pl_by_sample[name][i])) for name in pl_by_sample]]) for i in range(n)]
    return '\n'.join(header+rows)+'\n'


def parse_output(text,names,sites):
    calls={name:{} for name in names};regions={name:[] for name in names}
    if '# This file was produced by: bcftools roh(1.23.1+htslib-1.24)' not in text:
        raise ValueError('Wrong BCFtools/HTSlib output version')
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        row=line.split('\t');tag=row[0]
        if tag=='ST':
            if len(row)!=6 or row[1] not in calls or row[2]!='1':raise ValueError('Malformed state row')
            name,pos,state,q=row[1],int(row[3]),int(row[4]),float(row[5])
            if pos in calls[name] or not 1<=pos<=sites or state not in (0,1) or not 0<=q<=999:
                raise ValueError('Invalid or duplicate state output')
            calls[name][pos]={'state':state,'reported_quality':q}
        elif tag=='RG':
            if len(row)!=8 or row[1] not in regions or row[2]!='1':raise ValueError('Malformed region row')
            start,end,length,count=map(int,row[3:7]);quality=float(row[7])
            if not 1<=start<=end<=sites or length!=end-start+1 or count!=length or not 0<=quality<=999:
                raise ValueError('Invalid region output on consecutive grid')
            regions[row[1]].append({'start':start,'end':end,'marker_count':count,'reported_quality':quality})
        else:raise ValueError('Unexpected BCFtools output record')
    for name in names:
        if set(calls[name])!=set(range(1,sites+1)):raise ValueError('Missing output sites')
        covered=[position for region in regions[name] for position in range(region['start'],region['end']+1)]
        if covered!=[p for p in range(1,sites+1) if calls[name][p]['state']==1]:raise ValueError('Regions do not match the state path')
    return {name:{'states':[calls[name][i]['state'] for i in range(1,sites+1)],
                  'reported_quality':[calls[name][i]['reported_quality'] for i in range(1,sites+1)],
                  'regions':regions[name]} for name in names}


def run_tool(pl_by_sample,frequencies,executable=BINARY,switch=SWITCH):
    check_version(executable)
    # Validate the transition and model dimensions independently of the tool.
    infer(next(iter(pl_by_sample.values()),[]),frequencies,switch)
    text=vcf_text(pl_by_sample,frequencies)
    with tempfile.TemporaryDirectory(prefix='paper-to-code-roh-') as directory:
        path=Path(directory)/'synthetic.vcf';path.write_text(text)
        result=subprocess.run([str(executable),'roh','--AF-tag','AF','-a',str(switch),'-H',str(switch),
                               '-b','0','--threads','0',str(path)],capture_output=True,text=True,check=True,timeout=30)
    return parse_output(result.stdout,list(pl_by_sample),len(frequencies))


def evaluate(result,rows,frequencies,truth=None):
    reference=infer(rows,frequencies)
    states=result['states']
    return {'markers':len(rows),'inferred_AZ_markers':sum(states),
            'inferred_AZ_marker_fraction':statistics.mean(states),
            'independent_viterbi_disagreements':sum(a!=b for a,b in zip(states,reference['viterbi'])),
            'designed_AZ_markers':None if truth is None else sum(truth),
            'state_disagreements_with_design':None if truth is None else sum(a!=b for a,b in zip(states,truth)),
            'regions':result['regions']}


def report(executable=BINARY):
    data=json.loads(FIXTURE.read_text());frequencies=data['ALT_frequencies']
    versions=check_version(executable)
    input_pl={name:item['PL'] for name,item in data['samples'].items()}
    actual=run_tool(input_pl,frequencies,executable)
    main={name:evaluate(actual[name],item['PL'],frequencies,item['truth_states']) for name,item in data['samples'].items()}
    # Same most likely genotypes, but deliberately less discriminating PLs.
    weak={name:[[x//10 for x in row] for row in rows] for name,rows in input_pl.items()}
    weak_calls=run_tool(weak,frequencies,executable)
    uncertainty={name:evaluate(weak_calls[name],weak[name],frequencies,data['samples'][name]['truth_states']) for name in weak}
    control_pl=[[30,0,30]]*8+[[0,30,60]]*24+[[30,0,30]]*8
    sensitivity=[]
    for f in (.5,.01):
        output=run_tool({'frequency_control':control_pl},[f]*40,executable)['frequency_control']
        hw,az=emissions([0,30,60],f)
        sensitivity.append({'supplied_ALT_frequency':f,'homref_emission_ratio_AZ_over_HW':az/hw,
                            **evaluate(output,control_pl,[f]*40)})
    tiny=[[30,0,30],[0,30,60],[0,30,60],[30,0,30]]
    probe=run_tool({'quality_control':tiny},[.5]*4,executable)['quality_control']
    oracle=infer(tiny,[.5]*4)
    q=[phred_error(p if state==0 else 1-p) for state,p in zip(probe['states'],oracle['posterior_AZ'])]
    return {'example':'BCFtools/RoH: within-individual autozygosity and likelihood controls',
            'versions':{**versions,'python':platform.python_version()},'synthetic_seed':data['seed'],
            'transition_HW_to_AZ_and_AZ_to_HW_per_marker':SWITCH,
            'coordinate_system':'Consecutive synthetic positions 1..180; not realistic physical SNP density',
            'main_constructed_PLs':main,'weaker_PLs_same_best_genotypes':uncertainty,
            'frequency_sensitivity_same_PLs':sensitivity,
            'quality_diagnostic':{'PL':tiny,'states':probe['states'],
                'tool_reported_quality':probe['reported_quality'],
                'independent_same_site_posterior_AZ':oracle['posterior_AZ'],
                'independent_same_site_phred_error':q,
                'interpretation':'The pinned tool quality column disagrees with same-site independent marginals in this diagnostic. Do not convert these reported Q fields into calibrated site probabilities; state paths are checked separately.'},
            'limitations':'Conditional HMM state inference with supplied frequencies and fixed transitions; no genome-wide inbreeding estimate, pedigree reconstruction, empirical accuracy or calibrated confidence claim. PLs are constructed, not read-derived.'}


def render(value):
    def rounded(x):
        if isinstance(x,float):return round(x,9)
        if isinstance(x,dict):return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,list):return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(value),indent=2,allow_nan=False)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bcftools',default=str(BINARY));parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    try:
        text=render(report(args.bcftools))
        if args.check_sample and text!=SAMPLE.read_text():raise ValueError('Output differs from checked sample')
    except (ValueError,OSError,subprocess.SubprocessError) as error:parser.error(str(error))
    print(text,end='')


if __name__=='__main__':main()
