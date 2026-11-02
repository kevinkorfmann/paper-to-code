"""Deterministic examples of tangent compression, merging and a chunk-ranking failure."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

from .model import chunk_probabilities,envelope,log_mass,merge,normalized,score,summarize

SAMPLE=Path(__file__).with_name('sample_output.json')


def build_report():
    keys=[[-2.],[2.]];references=[[-1.],[0.],[1.]]
    summaries=[summarize(keys,r) for r in references]
    rows=[]
    for query in ([-2.],[-1.],[0.],[.5],[1.],[2.]):
        exact=log_mass(keys,query)
        estimate=envelope(summaries,query)
        rows.append({'query':query[0],'exact_log_mass':exact,'tangent_envelope':estimate,'gap':exact-estimate})
    # Shared-reference recursive compression should recover a direct flat summary.
    children=[[[0.],[1.]],[[2.],[3.],[4.]]];r=[.7]
    direct=summarize(children[0]+children[1],r)
    combined=merge([summarize(c,r) for c in children],r)
    # Equal means hide a high-logit key when only the zero reference is supplied.
    ranking_keys=[[-6.],[6.],[2.],[2.]]
    zero=chunk_probabilities(ranking_keys,[1.],[[0.]],2)
    enriched=chunk_probabilities(ranking_keys,[1.],[[0.],[1.]],2)
    # Illustrative top-one-chunk decision, not the paper's top-p tree algorithm.
    # Values are one throughout chunk A, zero throughout chunk B.
    for result in (zero,enriched):
        selected=max(range(2),key=lambda i:result['estimated_log_masses'][i])
        result['top_one_chunk_diagnostic']=selected
        result['diagnostic_selected_chunk_output']=float(selected==0)
    prefix=chunk_probabilities(ranking_keys+[[-10000.],[10000.]],[1.],[[0.],[1.]],2,visible=4)
    p_ref,log_ref,_=normalized([-2.,2.])
    _,log_query,_=normalized([-4.,4.])
    kl=sum(p*(a-b) for p,a,b in zip(p_ref,log_ref,log_query))
    return {'example':'ZHA reference-query tangent key summaries',
            'python_version':platform.python_version(),'randomness':'None; all inputs and references are constructed.',
            'symmetric_chunk':{'keys':keys,'references':references,'summaries':[asdict(x) for x in summaries],'queries':rows},
            'shared_reference_merge':{'reference':r,'direct':asdict(direct),'merged':asdict(combined),
                'max_parameter_difference':max(abs(a-b) for a,b in zip(direct.key+(direct.bias,),combined.key+(combined.bias,)))},
            'kl_gap_control':{'reference':1.,'query':2.,'gap':log_mass(keys,[2.])-score(summaries[2],[2.]),'KL_reference_to_query':kl},
            'chunk_ranking_counterexample':{'keys':ranking_keys,'query':[1.],
                'dense_attention_output_for_values_1100':zero['exact_chunk_probabilities'][0],
                'zero_reference':zero,'zero_and_one_references':enriched,
                'extreme_future_tokens_leave_visible_prefix_unchanged':prefix==enriched_without_diagnostic(enriched)},
            'limitations':'Supplied reference queries; mathematical summary and merge component only. No learned references, full hierarchy, top-p routing, RoPE, sink handling, training, optimized kernel, speed or model-gain claim. Lower log-mass bounds do not bound normalized chunk probabilities.'}


def enriched_without_diagnostic(result):
    return {k:v for k,v in result.items() if k not in ('top_one_chunk_diagnostic','diagnostic_selected_chunk_output')}


def render(value):
    def rounded(x):
        if isinstance(x,float): return round(x,9)
        if isinstance(x,dict): return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(value),indent=2,allow_nan=False)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    text=render(build_report())
    if args.check_sample and text!=SAMPLE.read_text(): parser.error('Output differs from checked sample')
    print(text,end='')


if __name__=='__main__': main()
