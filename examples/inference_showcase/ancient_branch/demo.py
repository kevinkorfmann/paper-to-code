"""Synthetic tskit branch divergence and dated-sample TMRCA control."""
import argparse
import json
import math
from pathlib import Path
import tskit


def fixture(ages=(0, 0, 6), roots=(10, 20), boundary=75, length=100):
    if (len(ages) != 3 or len(roots) != 2 or
        any(not math.isfinite(x) or x < 0 for x in ages+roots) or
        min(roots) <= max(ages) or not 0 < boundary < length):
        raise ValueError('invalid synthetic tree specification')
    tables=tskit.TableCollection(length)
    tables.time_units='generations'
    for age in ages: tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age)
    for left,right,root_time in [(0,boundary,roots[0]),(boundary,length,roots[1])]:
        root=tables.nodes.add_row(time=root_time)
        for child in range(3): tables.edges.add_row(left,right,root,child)
    tables.sort()
    return tables.tree_sequence()


def estimate(ts, a, b):
    """Mean cross-set MRCA age, assuming connected dated trees and disjoint samples."""
    for group in (a,b):
        if not group or len(set(group)) != len(group): raise ValueError('empty or duplicated group')
        if any(type(x) is not int or x not in ts.samples() for x in group):
            raise ValueError('groups must contain sample IDs')
    if set(a)&set(b): raise ValueError('groups must be disjoint')
    # A missing common ancestor makes the TMRCA interpretation undefined.
    for tree in ts.trees():
        for u in a:
            for v in b:
                if tree.mrca(u,v)==tskit.NULL: raise ValueError('disconnected pair')
    branch=float(ts.divergence([a,b],mode='branch'))
    age_a=sum(ts.node(u).time for u in a)/len(a)
    age_b=sum(ts.node(u).time for u in b)/len(b)
    return {'branch_divergence':branch,'naive_half_branch':branch/2,
            'mean_sample_age_sum':age_a+age_b,
            'corrected_mean_tmrca':(branch+age_a+age_b)/2}


def demo():
    ts=fixture()
    return {'tskit_version':tskit.__version__,'time_units':ts.time_units,
            'sample_times':[0,0,6],'tree_intervals':[[0,75],[75,100]],
            'root_times':[10,20],
            'modern_pair':estimate(ts,[0],[1]),
            'modern_ancient_pair':estimate(ts,[0],[2]),
            'modern_vs_mixed_group':estimate(ts,[0],[1,2]),
            'mixed_group_per_tree_branch':ts.divergence([[0],[1,2]],mode='branch',windows='trees').tolist(),
            'mixed_group_unnormalised_branch':float(ts.divergence([[0],[1,2]],mode='branch',span_normalise=False)),
            'site_divergence_without_mutations':float(ts.divergence([[0],[1,2]],mode='site')),
            'note':'Exact constructed-tree summary, not a genealogy-inference run.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-sample',action='store_true');args=parser.parse_args()
    out=json.dumps(demo(),indent=2,sort_keys=True)+'\n'
    if args.check_sample:
        if out != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample output mismatch')
        print('sample output matches')
    else: print(out,end='')

if __name__=='__main__': main()
