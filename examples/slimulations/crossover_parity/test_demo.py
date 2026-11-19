import itertools
import math
import unittest
from examples.slimulations.crossover_parity import demo as d


class Theory(unittest.TestCase):
    def test_exhaustive_bernoulli_oracle(self):
        for n in range(1,9):
            for r in [0,.01,.1,.5]:
                mean=positive=odd=0
                for bits in itertools.product((0,1),repeat=n):
                    k=sum(bits);p=r**k*(1-r)**(n-k)
                    mean+=k*p;positive+=(k>0)*p;odd+=(k%2)*p
                t=d.theory(n,r)
                for a,b in zip((mean,positive,odd),t.values()):self.assertAlmostEqual(a,b,places=12)
    def test_tiny_rate(self):
        t=d.theory(100,1e-20)
        self.assertTrue(math.isclose(t["probability_odd"],1e-18,rel_tol=1e-14))
        self.assertTrue(math.isclose(t["probability_any"],1e-18,rel_tol=1e-14))
    def test_saturation(self):
        self.assertEqual(d.theory(100,.1)['mean_breakpoints'],10)
        self.assertAlmostEqual(d.theory(100,.1)['probability_odd'],.5,places=8)
        self.assertGreater(d.theory(100,.1)['probability_any'],.99)
    def test_one_gap(self):
        for r in [0,.1,.5]:
            for value in d.theory(1,r).values():self.assertAlmostEqual(value,r)
    def test_invalid(self):
        for n,r in [(0,.1),(True,.1),(1,-1),(1,.51),(1,float('nan'))]:
            with self.assertRaises(ValueError):d.theory(n,r)


class Parsing(unittest.TestCase):
    def test_good(self):self.assertEqual(d.parse('DATA,4,5,3,1,1\n',2,3)['breakpoints'],5)
    def test_bad(self):
        for row in ['', 'DATA,4,1,1,1,1\nDATA,4,1,1,1,1', 'DATA,4,1,1,1','DATA,5,1,1,1,1','DATA,4,1,2,1,1','DATA,4,1,1,1,0','DATA,4,13,4,2,2']:
            with self.subTest(row=row),self.assertRaises(ValueError):d.parse(row,2,3)
    def test_bad_summary(self):
        with self.assertRaises(AssertionError):d.summarize(dict(gametes=40000,breakpoints=0,any_breakpoint=0,recombinant_markers=0),100,.1)


class Engine(unittest.TestCase):
    def test_replay(self):self.assertEqual(d.run(n=1000),d.run(n=1000))
    def test_zero(self):
        c=d.run(n=1000,r=0);self.assertEqual(c,dict(gametes=2000,breakpoints=0,any_breakpoint=0,odd_breakpoints=0,recombinant_markers=0))
    def test_one_gap_identity(self):
        c=d.run(d=1,r=.5,n=2000)
        self.assertEqual(c['breakpoints'],c['any_breakpoint']);self.assertEqual(c['breakpoints'],c['recombinant_markers'])
    def test_multiple_seeds(self):
        for seed in [41,42,43]:
            c=d.run(d=100,r=.01,n=2000,seed=seed);d.summarize(c,100,.01)
    def test_invalid_seed(self):
        with self.assertRaises(ValueError):d.run(seed=0)

if __name__=='__main__':unittest.main()
