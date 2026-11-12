"""Deterministic constructed controls; no training or real-model claims."""
import argparse
import json
import math
from pathlib import Path
from .model import allocate, prior, read, rollout


def rounded(x):
    if isinstance(x, float): return round(x, 9)
    if isinstance(x, list): return [rounded(v) for v in x]
    if isinstance(x, dict): return {k: rounded(v) for k,v in x.items()}
    return x


def result():
    # Exact target score vector [log(6), 0, 0, 0] is supplied, not learned.
    scores = [math.log(6), 0., 0., 0.]
    weights = allocate(scores, 2, .5)
    reads, _, _ = rollout([10.], [lambda h: [h[0]+1.]], [[0.]]*4, .5)
    unstable, _, _ = rollout([1.], [lambda h: [3*h[0]]], [[0.]]*8, .5)
    bad=[]
    current=10.
    for n in range(1,5):
        current += prior(n,1,.5)[1]
        bad.append(current)
    cumulative=[10.+1.,10.+1.-2.]
    update_values=[1.,-2.]
    return rounded({
        'environment': 'CPython 3.14.5; standard library; no randomness',
        'source': 'Ouyang et al., arXiv:2611.07226v1, 2026-11-04; eqs. 14-16',
        'scope': 'Supplied depth reader and scalar toy blocks; not a trained Transformer',
        'boundary_prior_K2_lambda_half': [[math.exp(x) for x in prior(n,2,.5)[0]] for n in range(1,5)],
        'four_update_prior': [.25,.25,.5,.5],
        'four_update_total_mass': sum(weights),
        'scores_log6_0_0_0_weights': weights,
        'values_1_0_0_0_readout_with_initial_10': 10+weights[0],
        'cumulative_states_11_9_equal_read': sum(cumulative)/2,
        'updates_1_minus2_equal_read_plus_initial_10': 10+sum(update_values)/2,
        'replacement_states_unit_updates': [x[0] for x in reads],
        'wrong_accumulate_each_complete_mixture': bad,
        'bounded_mass_not_bounded_state': {
            'block': 'f(h)=3h; K=1; lambda=0.5; zero queries',
            'states': [x[0] for x in unstable],
            'mass_after_8': prior(8,1,.5)[1],
            'mass_limit': 1.,
        },
        'RMS_read_example': read([10.,0.],[[1.,0.],[0.,2.]],[1.,-1.],2,.5)[0],
    })


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-sample',action='store_true')
    args=p.parse_args()
    output=json.dumps(result(),indent=2,allow_nan=False)+'\n'
    if args.check_sample:
        if output != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample output differs')
        print('Sample output verified.')
    else: print(output,end='')

if __name__=='__main__': main()
