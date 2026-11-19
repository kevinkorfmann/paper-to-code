from fractions import Fraction
import json
from pathlib import Path
import unittest
from examples.inference_showcase.ancient_branch import demo as d


def oracle(ts,a,b):
    # Independent parent-chain intersection using edge-table intervals, not mrca/statistics.
    total=Fraction(0)
    cuts=sorted({0,ts.sequence_length}|set(ts.edges_left)|set(ts.edges_right))
    for left,right in zip(cuts,cuts[1:]):
        parent={e.child:e.parent for e in ts.edges() if e.left<=left<e.right}
        for u in a:
            for v in b:
                ancestors=set(); node=u
                while True:
                    ancestors.add(node)
                    if node not in parent: break
                    node=parent[node]
                node=v
                while node not in ancestors: node=parent[node]
                total+=Fraction(right-left)*Fraction(ts.node(node).time)
    return float(total/Fraction(ts.sequence_length)/len(a)/len(b))


class Tests(unittest.TestCase):
    def test_grid_against_independent_oracle(self):
        for age in (0,1,6,9):
            for boundary in (1,25,75,99):
                ts=d.fixture(ages=(0,2,age),boundary=boundary)
                for a,b in [([0],[1]),([0],[2]),([0],[1,2]),([1,2],[0])]:
                    with self.subTest(age=age,boundary=boundary,a=a,b=b):
                        self.assertAlmostEqual(d.estimate(ts,a,b)['corrected_mean_tmrca'],oracle(ts,a,b))
    def test_measured(self):
        x=d.demo()
        self.assertEqual(x['modern_pair']['branch_divergence'],25)
        self.assertEqual(x['modern_ancient_pair']['branch_divergence'],19)
        self.assertEqual(x['modern_ancient_pair']['naive_half_branch'],9.5)
        self.assertEqual(x['modern_vs_mixed_group']['branch_divergence'],22)
        self.assertEqual(x['mixed_group_per_tree_branch'],[17,37])
        self.assertEqual(x['mixed_group_unnormalised_branch'],2200)
    def test_all_modern(self):
        r=d.estimate(d.fixture(ages=(0,0,0)),[0],[1,2])
        self.assertEqual(r['naive_half_branch'],r['corrected_mean_tmrca'])
    def test_time_scale(self):
        x=d.estimate(d.fixture(ages=(0,0,12),roots=(20,40)),[0],[2])
        self.assertEqual(x['corrected_mean_tmrca'],25)
        self.assertEqual(x['branch_divergence'],38)
    def test_time_origin_shift(self):
        x=d.estimate(d.fixture(ages=(5,5,11),roots=(15,25)),[0],[2])
        self.assertEqual(x['branch_divergence'],19)
        self.assertEqual(x['corrected_mean_tmrca'],17.5)
    def test_coordinate_scale(self):
        self.assertEqual(d.estimate(d.fixture(boundary=750,length=1000),[0],[2]),d.estimate(d.fixture(),[0],[2]))
    def test_span_not_tree_count(self):
        self.assertEqual(oracle(d.fixture(),[0],[2]),12.5)
        self.assertNotEqual(oracle(d.fixture(),[0],[2]),15)
    def test_mutation_invariance(self):
        ts=d.fixture(); tables=ts.dump_tables();site=tables.sites.add_row(10,'A');tables.mutations.add_row(site,0,'T');mut=tables.tree_sequence()
        self.assertEqual(d.estimate(mut,[0],[2]),d.estimate(ts,[0],[2]))
        self.assertAlmostEqual(float(mut.divergence([[0],[2]],mode='site')),.01)
    def test_group_permutation(self):
        ts=d.fixture();self.assertEqual(d.estimate(ts,[0],[1,2]),d.estimate(ts,[2,1],[0]))
    def test_disconnected_rejected(self):
        tables=d.fixture().dump_tables();tables.edges.clear()
        with self.assertRaises(ValueError):d.estimate(tables.tree_sequence(),[0],[2])
    def test_bad_groups(self):
        for a,b in [([],[1]),([0],[0]),([0,0],[1]),([0],[3]),([0],[True])]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):d.estimate(d.fixture(),a,b)
    def test_bad_fixture(self):
        for kw in [dict(ages=(0,0,10)),dict(boundary=0),dict(roots=(5,20)),dict(ages=(0,0,float('nan')))]:
            with self.assertRaises(ValueError):d.fixture(**kw)
    def test_sample(self):
        self.assertEqual(json.loads(Path(d.__file__).with_name('sample_output.json').read_text()),d.demo())

if __name__=='__main__':unittest.main()
