import itertools
import math
import random
import unittest
from . import demo as d


class Tests(unittest.TestCase):
    def test_scalar_closed_form(self):
        a=.5;expected=2*math.exp(a*a/2)/(1+2*math.exp(a*a/2))
        got=d.approximate([[1]],[[-a],[a],[0]],[1,1,0],[[0,1],[2]],[2])[0]
        self.assertAlmostEqual(got,expected)
        exact=2*math.cosh(a)/(1+2*math.cosh(a))
        self.assertAlmostEqual(d.dense([[1]],[[-a],[a],[0]],[1,1,0])[0],exact)
    def test_all_selected(self):
        rng=random.Random(19)
        for n in range(1,12):
            q=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(4)]
            k=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(n)];v=[rng.random() for _ in k]
            self.assertEqual(d.approximate(q,k,v,[list(range(n))],list(range(n))),d.dense(q,k,v))
    def test_identical_cluster_exact(self):
        q=[[1],[2]];k=[[.5],[.5],[-1]];v=[1,1,0]
        self.assertEqual(d.approximate(q,k,v,[[0,1],[2]],[2]),d.dense(q,k,v))
    def test_singletons_exact(self):
        self.assertEqual(d.approximate([[1]],[[1],[2]],[1,2],[[0],[1]],[]),d.dense([[1]],[[1],[2]],[1,2]))
    def test_head_average_not_per_query(self):
        # alpha=(1+9)/(2*2)=2.5; m2=.25; shared correction=.625.
        out=d.approximate([[1],[3]],[[-.5],[.5],[0]],[1,1,0],[[0,1],[2]],[2])
        expected=2*math.exp(.625)/(1+2*math.exp(.625))
        for x in out:self.assertAlmostEqual(x,expected)
    def test_fixed_center_after_selection(self):
        # Remaining key is -1, but full-cluster center is 0 and m2=1.
        out=d.approximate([[1]],[[-1],[1],[0]],[1,1,0],[[0,1],[2]],[1,2])[0]
        mass=math.e+math.exp(.5)
        self.assertAlmostEqual(out,mass/(mass+1))
    def test_fixed_value_after_selection(self):
        got=d.approximate([[0]],[[-1],[1]],[0,2],[[0,1]],[1])[0]
        self.assertEqual(got,1.5) # selected v=2, full-cluster mean v=1
    def test_cap(self):
        got=d.approximate([[10]],[[-1],[1],[0]],[1,1,0],[[0,1],[2]],[2],cap=.2)[0]
        self.assertAlmostEqual(got,2*math.exp(.2)/(1+2*math.exp(.2)))
    def test_permutation(self):
        q=[[1]];k=[[-1],[0],[1]];v=[2,3,4]
        expected=d.approximate(q,k,v,[[0,2],[1]],[1])
        for p in itertools.permutations(range(3)):
            inv={old:new for new,old in enumerate(p)}
            got=d.approximate(q,[k[i] for i in p],[v[i] for i in p],[[inv[0],inv[2]],[inv[1]]],[inv[1]])
            self.assertEqual(got,expected)
    def test_help_and_failure_controls(self):
        c=d.demo()['cases']
        self.assertLess(c['symmetric_spread']['vcm_absolute_error'],c['symmetric_spread']['mean_absolute_error'])
        self.assertEqual(c['orthogonal_spread']['mean_absolute_error'],0)
        for name in ['orthogonal_spread','selection_mean_shift']:
            self.assertGreater(c[name]['vcm_absolute_error'],c[name]['mean_absolute_error'])
    def test_invalid(self):
        args=([[1]],[[0],[1]],[0,1],[[0],[1]],[])
        for clusters,selected,cap in [([[0,0],[1]],[],1),([[0]],[],1),([[0],[1]],[2],1),([[0],[1]],[],float('nan')),([[0],[1]],[],-1)]:
            with self.assertRaises(ValueError):d.approximate(*args[:3],clusters,selected,cap)
    def test_zero_queries(self):
        self.assertEqual(d.approximate([[0]],[[1],[2],[3]],[1,2,3],[[0,1,2]],[]),[2])

if __name__=='__main__':unittest.main()
