"""Constructed competing-head votes and token-budget bookkeeping."""
import json
import math
from .core import route, attend, softmax


def run():
    c=[[1.,0.],[0.,1.],[math.sqrt(.5),math.sqrt(.5)]]
    q=[[2.,0.],[0.,2.]]
    labels=[0,1,2,0,1,2]
    r=route(q,c,labels,5,[],1)
    # Deliberately DIFFERENT comparator: add cosine logits before softmax.
    wrong=softmax([(v[0]+v[1])/.1 for v in c])
    keys=[c[i] for i in labels]
    values=[[float(i)] for i in range(6)]
    visible=route(q,c,labels+[0,1,2],4,[4],2)
    all_keys=route(q,c,labels,5,[],6)
    return {'per_head_votes':[[round(x,9) for x in row] for row in r['head_votes']],
            'penalized_vote_sums':[round(x,9) for x in r['scores']],
            'selected_clusters':r['clusters'],'selected_indices':r['indices'],
            'historical_budget':1,'actual_historical_tokens':r['historical_count'],
            'softmax_of_summed_logits':[round(x,9) for x in wrong],
            'alternative_selected_cluster':max(range(3),key=wrong.__getitem__),
            'causal_recent_control':{'query_index':4,'dense_positions':[4],
                                    'visible_block_sizes':visible['visible_block_sizes'],
                                    'selected_indices':visible['indices']},
            'all_keys_max_error':max(abs(a[0]-b[0]) for a,b in zip(
                attend(q,keys,values,all_keys['indices']),attend(q,keys,values,list(range(6)))))}

if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
