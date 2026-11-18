"""Independent labelled-subset enumeration and real-engine checks."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
import unittest
from . import demo


def enumeration(n,het,hom,s):
    dosages=[1]*het+[2]*hom+[0]*(n-het-hom)
    counts=Counter((sum(dosages[i]==1 for i in ids),sum(dosages[i]==2 for i in ids)) for ids in combinations(range(n),s))
    total=sum(counts.values())
    return {key:Fraction(value,total) for key,value in counts.items()}


class Theory(unittest.TestCase):
    def test_all_small_populations(self):
        for n in range(2,9):
            for het in range(n+1):
                for hom in range(n-het+1):
                    for s in range(1,n+1):
                        expected=enumeration(n,het,hom,s)
                        got=demo.genotype_distribution(n,het,hom,s)
                        self.assertEqual(set(expected),set(got))
                        for key in expected: self.assertAlmostEqual(float(expected[key]),got[key],places=14)
    def test_moments(self):
        for n in (3,8,40):
            for het,hom in ((0,0),(n,0),(0,n),(n//2,n//3)):
                for s in (1,n//2,n):
                    probs=demo.genotype_distribution(n,het,hom,s)
                    mean=(het+2*hom)/(2*n)
                    self.assertAlmostEqual(sum(probs.values()),1)
                    self.assertAlmostEqual(sum(p*(a+2*b)/(2*s) for (a,b),p in probs.items()),mean)
                    self.assertAlmostEqual(sum(p*((a+2*b)/(2*s)-mean)**2 for (a,b),p in probs.items()),demo.frequency_variance(n,het,hom,s))
                    self.assertAlmostEqual(probs.get((0,0),0),demo.loss_probability(n,het,hom,s))
    def test_main_order(self):
        a=demo.loss_probability(40,8,0,5)
        b=demo.loss_probability(40,4,2,5)
        c=demo.loss_probability(40,0,4,5)
        self.assertLess(a,b); self.assertLess(b,c)
    def test_loss_counts_carriers_not_copies(self):
        self.assertEqual(demo.loss_probability(40,6,0,5),demo.loss_probability(40,0,6,5))
    def test_census(self): self.assertEqual(demo.genotype_distribution(40,8,0,40),{(8,0):1})
    def test_obligate_carrier(self): self.assertEqual(demo.loss_probability(40,40,0,1),0)
    def test_absent(self): self.assertEqual(demo.loss_probability(40,0,0,5),1)
    def test_fixed(self): self.assertEqual(demo.frequency_variance(40,0,40,5),0)
    def test_invalid(self):
        for x in [(1,0,0,1),(40,41,0,5),(40,5,36,5),(40,0,0,0),(40,0,0,41),(40,1.5,0,5),(True,0,0,1)]:
            with self.subTest(x=x),self.assertRaises(ValueError): demo.genotype_distribution(*x)


class Parsing(unittest.TestCase):
    good='DATA,0,0,2,1\nDATA,1,0,1,1\n'
    def test_good(self): self.assertEqual(demo.parse(self.good,5,2,1,2,1),[(1,1)])
    def test_missing(self):
        with self.assertRaises(ValueError): demo.parse('',5,2,1,2,1)
    def test_duplicate(self):
        with self.assertRaises(ValueError): demo.parse(self.good+self.good,5,2,1,2,1)
    def test_founders(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('0,0,2,1','0,0,1,1'),5,2,1,2,1)
    def test_impossible(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('1,0,1,1','1,0,2,1'),5,2,1,2,1)
    def test_bad_index(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('1,0,1,1','1,1,1,1'),5,2,1,2,1)
    def test_bad_width(self):
        with self.assertRaises(ValueError): demo.parse('DATA,0,0,2\n',5,2,1,2,1)
    def test_bad_tick(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('1,0,1,1','3,0,1,1'),5,2,1,2,1)
    def test_invalid_summary(self):
        with self.assertRaises(ValueError): demo.summarize([(3,3)]*2,5,2,1,2)
    def test_short_summary(self):
        with self.assertRaises(ValueError): demo.summarize([(1,1)],5,2,1,2)


class Engine(unittest.TestCase):
    def test_replay(self):
        a=demo.run(n=10,het=2,hom=2,survivors=3,replicates=32,seed=123)
        self.assertEqual(a,demo.run(n=10,het=2,hom=2,survivors=3,replicates=32,seed=123))
    def test_census_control(self): self.assertEqual(demo.run(n=10,het=2,hom=3,survivors=10,replicates=16),[(2,3)]*16)
    def test_absent_fixed(self):
        for hom in (0,10): self.assertEqual(demo.run(n=10,het=0,hom=hom,survivors=3,replicates=16),[(0,0 if hom==0 else 3)]*16)
    def test_all_heterozygotes(self): self.assertEqual(demo.run(n=10,het=10,hom=0,survivors=3,replicates=16),[(3,0)]*16)
    def test_seed_validation(self):
        with self.assertRaises(ValueError): demo.run(seed=0)

if __name__=='__main__': unittest.main()
