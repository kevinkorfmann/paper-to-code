"""Constructed SwiLA responsibilities, query routing, and limitation controls."""

import argparse
import json
import math
from pathlib import Path
import platform

from .model import column_softmax, read, run, update

SAMPLE = Path(__file__).with_name('sample_output.json')


def render(value):
    def rounded(x):
        if isinstance(x, float): return round(x, 12)
        if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(value), indent=2, allow_nan=False)+'\n'


def report():
    initial = [[[2.,0.],[-2.,0.]], [[-2.,0.],[2.,0.]]]
    key, value = [1.,0.], [1.,1.]
    uniform = [[0.,0.],[0.,0.]]
    rates = [[1.,1.],[1.,1.]]
    updated, responsibilities = update(initial,key,value,uniform,rates)
    query_logits = [[math.log(9),0.],[0.,math.log(9)]]
    reversed_logits = list(reversed(query_logits))
    zero = [[[0.,0.],[0.,0.]], [[0.,0.],[0.,0.]]]
    tokens = [{'key':key,'value':value,'write_logits':uniform,'beta':rates,
               'query':key,'read_logits':query_logits} for _ in range(3)]
    collapsed, collapsed_outputs = run(zero,tokens)
    overshoot, _ = update([[[0.]]],[2.],[1.],[[0.]],[[1.]])
    overshoot_output = read(overshoot,[2.],[[0.]])[0]
    return {'example':'Switching Linear Attention: coordinate-wise mixture updates',
            'paper':'https://arxiv.org/abs/2609.39034v1','paper_first_posted':'2026-09-30',
            'python_version':platform.python_version(),'randomness':'none; all states, projections and priors constructed',
            'coordinate_responsibilities':{'initial_state':initial,'key':key,'target_value':value,
                'write_priors':column_softmax(uniform),'learning_rates':rates,
                'old_prediction_errors':[[-1.,3.],[3.,-1.]],'write_responsibilities':responsibilities,
                'updated_state':updated,'query':key,'query_priors':column_softmax(query_logits),
                'read_with_supplied_query_priors':read(updated,key,query_logits),
                'read_with_uniform_query_priors':read(updated,key,uniform),
                'read_with_reversed_query_priors':read(updated,key,reversed_logits)},
            'identical_expert_control':{'initial_state':zero,'final_state_after_three_identical_tokens':collapsed,
                                       'outputs':collapsed_outputs,
                                       'interpretation':'Uniform write priors and equal rates preserve identical experts; query priors cannot distinguish identical readouts.'},
            'large_key_control':{'initial_state':[[[0.]]],'key':[2.],'target_value':[1.],'beta':1.,
                                 'updated_state':overshoot,'read_at_key':overshoot_output,
                                 'squared_error_before':1.,'squared_error_after':(overshoot_output-1.)**2,
                                 'interpretation':'Beta in [0,1] alone does not ensure contraction when key norm is large.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args = parser.parse_args()
    text = render(report())
    if args.check_sample and text != SAMPLE.read_text():
        parser.error('Output differs from checked sample')
    print(text,end='')


if __name__ == '__main__':
    main()
