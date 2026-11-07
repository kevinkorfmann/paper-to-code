"""Exact enumerations, failure cases and actual SLiM integration."""
from fractions import Fraction as F
from itertools import product
import math
import unittest
from examples.slimulations.reproductive_skew import demo

class OracleTests(unittest.TestCase):
    def test_exact_enumeration(self):
        for n in range(2,6):
            for mode in demo.MODES:
                p = ([F(1,n)]*n if mode=='uniform' else
                     [F(1)]+[F(0)]*(n-1) if mode=='single' else
                     [F(1,2)]+[F(1,2*(n-1))]*(n-1))
                mean=second=rich=F(0)
                for outcome in product(range(n),repeat=n):
                    prob=math.prod(p[x] for x in outcome)
                    counts=[outcome.count(i) for i in range(n)]
                    d=1-sum(F(c*c,n*n) for c in counts)
                    mean+=prob*d; second+=prob*d*d
                    rich+=prob*len(set(outcome))
                oracle=demo.oracle(n,mode)
                self.assertAlmostEqual(float(mean),oracle['expected_diversity'])
                self.assertAlmostEqual(float(second-mean*mean),oracle['variance_diversity'])
                self.assertAlmostEqual(float(rich),oracle['expected_richness'])
    def test_uniform(self):
        self.assertAlmostEqual(demo.oracle(100,'uniform')['expected_diversity'],.9801)
    def test_skew(self):
        o=demo.oracle(100,'skew')
        self.assertAlmostEqual(o['effective_contributors'],3.96)
        self.assertAlmostEqual(o['expected_diversity'],.74)
    def test_single(self):
        self.assertEqual(demo.oracle(100,'single')['expected_diversity'],0)
    def test_probability_sums(self):
        for n in range(2,201):
            for mode in demo.MODES:
                self.assertAlmostEqual(sum(demo.probabilities(n,mode)),1)
    def test_invalid_n(self):
        for n in [True,1,201,2.5]:
            with self.assertRaises(ValueError): demo.probabilities(n,'uniform')
    def test_invalid_mode(self):
        with self.assertRaises(ValueError): demo.oracle(5,'bad')
    def test_invalid_scale(self):
        for x in [True,0,float('nan'),float('inf'),11]:
            with self.assertRaises(ValueError): demo.run(scale=x)

class ParserTests(unittest.TestCase):
    valid='INIT,0,0.5,0.5\nDATA,0,1,1\n'
    def test_valid(self): self.assertEqual(demo.parse_output(self.valid,2,1,'uniform'),[[1,1]])
    def test_duplicate(self):
        with self.assertRaises(ValueError): demo.parse_output(self.valid+'DATA,0,1,1',2,1,'uniform')
    def test_missing(self):
        with self.assertRaises(ValueError): demo.parse_output('DATA,0,1,1',2,1,'uniform')
    def test_count(self):
        with self.assertRaises(ValueError): demo.parse_output(self.valid.replace('DATA,0,1,1','DATA,0,-1,3'),2,1,'uniform')
    def test_weight(self):
        with self.assertRaises(ValueError): demo.parse_output(self.valid.replace('0.5,0.5','nan,0.5'),2,1,'uniform')
    def test_width(self):
        with self.assertRaises(ValueError): demo.parse_output(self.valid.replace('DATA,0,1,1','DATA,0,2'),2,1,'uniform')
    def test_extra_rep(self):
        with self.assertRaises(ValueError): demo.parse_output(self.valid.replace('DATA,0','DATA,1'),2,1,'uniform')

class IntegrationTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(demo.run(n=10,replicates=8),demo.run(n=10,replicates=8))
    def test_common_scaling(self):
        self.assertEqual(demo.run('skew',n=10,replicates=8),demo.run('skew',n=10,replicates=8,scale=2))
    def test_single_parent(self):
        self.assertEqual(demo.run('single',n=10,replicates=8),[[10]+[0]*9]*8)
    def test_two_parent_modes_identical(self):
        self.assertEqual(demo.run('uniform',n=2,replicates=8),demo.run('skew',n=2,replicates=8))
    def test_checked_sample(self):
        import json
        self.assertEqual(demo.summary(),json.loads((demo.HERE/'sample_summary.json').read_text()))

if __name__=='__main__': unittest.main()
