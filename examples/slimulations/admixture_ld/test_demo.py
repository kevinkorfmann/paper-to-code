import itertools
import math
import unittest
from examples.slimulations.admixture_ld import demo as d


class Theory(unittest.TestCase):
    def test_exhaustive_source_draws(self):
        for n in range(2,9):
            for m in [0,.25,.5,1]:
                for phase in [-1,1]:
                    e=e2=0
                    for draws in itertools.product((0,1),repeat=n):
                        k=sum(draws);p=m**k*(1-m)**(n-k);x=phase*k*(n-k)/n**2
                        e+=p*x;e2+=p*x*x
                    mean,var=d.moments(n,m,phase)
                    self.assertAlmostEqual(mean,e);self.assertAlmostEqual(var,e2-e*e)
    def test_finite_factor(self):
        self.assertEqual(d.moments(40,.5,1)[0],.24375)
        self.assertNotEqual(d.moments(40,.5,1)[0],(1-1/80)*.25)
    def test_phase_identity(self):
        for k in range(41):
            self.assertAlmostEqual(d.ld([2*k,0,0,80-2*k]),k/40*(1-k/40))
            self.assertAlmostEqual(d.ld([0,2*k,80-2*k,0]),-k/40*(1-k/40))
    def test_fixed_source_ld(self):
        for i in range(4):
            c=[0]*4;c[i]=80;self.assertEqual(d.ld(c),0)
    def test_invalid(self):
        for n,m,p in [(1,.5,1),(40,-1,1),(40,float('nan'),1),(40,.5,0),(40,.5,True)]:
            with self.assertRaises(ValueError):d.moments(n,m,p)
        for c in [[],[0]*4,[-1,0,0,2],[1.5,0,0,0]]:
            with self.assertRaises(ValueError):d.ld(c)


class Parser(unittest.TestCase):
    good='DATA,0,0,80,0,0,0\nDATA,0,1,0,0,0,80\nDATA,0,2,40,0,0,40\nDATA,1,0,80,0,0,0\nDATA,1,1,0,0,0,80\nDATA,1,2,40,0,0,40\n'
    def test_good(self):self.assertEqual(d.parse(self.good,40,.5,1,2),[[40,0,0,40]]*2)
    def test_bad(self):
        for text in ['',self.good+self.good,self.good.replace('40,0,0,40','39,0,0,41'),self.good.replace('40,0,0,40','0,40,40,0'),self.good.replace('80,0,0,0','0,0,0,80')]:
            with self.assertRaises(ValueError):d.parse(text,40,.5,1,2)
    def test_mean_check(self):
        with self.assertRaises(AssertionError):d.summarize([[80,0,0,0]]*1024,40,.5,1)


class Engine(unittest.TestCase):
    def test_seed_replay(self):self.assertEqual(d.run(replicates=16),d.run(replicates=16))
    def test_endpoints(self):
        for m,expected in [(0,[0,0,0,80]),(1,[80,0,0,0])]:self.assertEqual(d.run(m=m,replicates=16),[expected]*16)
    def test_both_phases_and_seeds(self):
        for phase in [-1,1]:
            for seed in [30,31]:d.summarize(d.run(phase=phase,seed=seed,replicates=128),40,.5,phase)
    def test_invalid_seed(self):
        with self.assertRaises(ValueError):d.run(seed=0)

if __name__=='__main__':unittest.main()
