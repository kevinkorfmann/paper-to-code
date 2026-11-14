import itertools
import math
from fractions import Fraction
import unittest
from . import demo as d

class Theory(unittest.TestCase):
    def test_initial(self): self.assertEqual(d.distribution(4,1,0),[0,1,0,0,0])
    def test_absent(self): self.assertEqual(d.distribution(5,0,20),[1,0,0,0,0,0])
    def test_fixed(self): self.assertEqual(d.distribution(5,5,20),[0,0,0,0,0,1])
    def test_mass(self):
        for n in (2,5,20): self.assertAlmostEqual(sum(d.distribution(n,1,30)),1)
    def test_martingale(self):
        for k in range(6): self.assertAlmostEqual(d.moments(d.distribution(5,k,20),5)['mean'],k/5)
    def test_variance(self):
        p=.1; t=30; pred=d.moments(d.distribution(20,2,t),20)
        self.assertAlmostEqual(pred['variance'],p*(1-p)*(1-(1-1/20)**t))
    def test_survival_identity(self):
        p=d.moments(d.distribution(20,2,30),20)
        self.assertAlmostEqual(p['survival']*p['survivor_mean'],.1)
    def test_survival_monotonic(self):
        probs=[d.moments(d.distribution(5,1,t),5)['survival'] for t in range(10)]
        self.assertTrue(all(a>=b for a,b in zip(probs,probs[1:])))
    def test_survivor_increase(self): self.assertGreater(d.moments(d.distribution(20,2,30),20)['survivor_mean'],.1)
    def test_reflection(self):
        for a,b in zip(d.distribution(7,2,12),reversed(d.distribution(7,5,12))): self.assertAlmostEqual(a,b)
    def test_parent_enumeration(self):
        n=3; counts=[0]*(n+1)
        for parents in itertools.product(range(n),repeat=n): counts[sum(i==0 for i in parents)]+=1
        for a,b in zip(d.distribution(n,1,1),counts): self.assertAlmostEqual(a,b/n**n)
    def test_two_steps_fraction(self):
        n=3; prob=[Fraction(0),Fraction(1),Fraction(0),Fraction(0)]
        for _ in range(2):
            prob=[sum(prob[i]*math.comb(n,j)*Fraction(i,n)**j*Fraction(n-i,n)**(n-j) for i in range(n+1)) for j in range(n+1)]
        for a,b in zip(d.distribution(n,1,2),prob): self.assertAlmostEqual(a,float(b))
    def test_undefined(self):
        a=d.moments(d.distribution(4,0,3),4); self.assertIsNone(a['survivor_mean']); self.assertIsNone(a['segregating_mean'])
    def test_fixed_survives_not_segregates(self):
        a=d.moments(d.distribution(4,4,3),4); self.assertEqual(a['survivor_mean'],1); self.assertIsNone(a['segregating_mean'])
    def test_invalid_parameters(self):
        for args in ((1,0,1),(4,-1,2),(4,5,2),(4,1,-1),(4,True,2),(4,1,101)):
            with self.assertRaises(ValueError): d.distribution(*args)
    def test_invalid_distribution(self):
        for p in ([0,0,0],[-1,1,1],[float('nan'),0,1]):
            with self.assertRaises(ValueError): d.moments(p,2)
    def test_observation_identity(self):
        o=d.observed([0,0,1,4],4); self.assertEqual(o['survivor_mean']*o['survivors']/o['replicates'],o['mean'])
    def test_observation_categories(self):
        o=d.observed([0,0,1,4],4); self.assertEqual((o['survivors'],o['fixed'],o['segregating']),(2,1,1))
    def test_observation_invalid(self):
        for c in ([],[True],[5],[-1]):
            with self.assertRaises(ValueError): d.observed(c,4)
    def test_parse_good(self): self.assertEqual(d.parse('DATA,0,0,1\nDATA,1,0,2',2,1,1,1),[[1],[2]])
    def test_parse_bad(self):
        for text in ('DATA,0,0,1','DATA,0,0,1\nDATA,0,0,1','DATA,0,0,0\nDATA,1,0,1','DATA,0,0,1\nDATA,1,0,3','DATA,0,0,1,2'):
            with self.assertRaises(ValueError): d.parse(text,2,1,1,1)
    def test_absorption_parser(self):
        with self.assertRaises(ValueError): d.parse('DATA,0,0,0\nDATA,1,0,1',2,0,1,1)

class Integration(unittest.TestCase):
    def test_boundaries(self):
        for k in (0,4): self.assertEqual(d.run(n=4,k=k,t=2,r=4),[[k]*4]*3)
    def test_replay(self): self.assertEqual(d.run(n=4,k=1,t=3,r=8),d.run(n=4,k=1,t=3,r=8))
    def test_sample(self):
        import json
        self.assertEqual(d.rounded(d.summary()),json.loads((d.HERE/'sample_summary.json').read_text()))

if __name__=='__main__': unittest.main()
