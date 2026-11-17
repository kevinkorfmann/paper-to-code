"""Independent exact microstate checks plus actual SLiM controls."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import math
import unittest
from . import demo


def enumerate_paths(n,k,m):
    # Labelled parents and death subsets, rather than binomial/hypergeometric formula.
    counts=Counter()
    for deaths in combinations(range(n),m):
        d=sum(x<k for x in deaths)
        for parents in product(range(n),repeat=m):
            b=sum(x<k for x in parents)
            counts[k-d+b]+=1
    total=sum(counts.values())
    return [Fraction(counts[i],total) for i in range(n+1)]


class Theory(unittest.TestCase):
    def test_exact_paths(self):
        for n in range(2,6):
            for k in range(n+1):
                for m in range(min(n,3)+1):
                    with self.subTest(n=n,k=k,m=m):
                        exact=enumerate_paths(n,k,m)
                        observed=demo.distribution(n,k,m)
                        for a,b in zip(exact,observed): self.assertAlmostEqual(float(a),b,places=13)
    def test_moments(self):
        for n in (2,5,40):
            for k in range(n+1):
                for m in (0,1,n//2,n):
                    p=demo.distribution(n,k,m)
                    self.assertAlmostEqual(sum(p),1)
                    self.assertAlmostEqual(sum(i*q for i,q in enumerate(p)),k)
                    self.assertAlmostEqual(sum(q*((i-k)/n)**2 for i,q in enumerate(p)),demo.expected_msd(n,k,m))
    def test_moran_limit(self): self.assertEqual(demo.expected_msd(40,20,1),2*.25/40**2)
    def test_wf_limit(self): self.assertEqual(demo.expected_msd(40,20,40),.25/40)
    def test_no_turnover(self): self.assertEqual(demo.distribution(40,20,0)[20],1)
    def test_absent(self): self.assertEqual(demo.distribution(40,0,5)[0],1)
    def test_fixed(self): self.assertEqual(demo.distribution(40,40,5)[40],1)
    def test_complement(self):
        for a,b in zip(demo.distribution(10,3,4),reversed(demo.distribution(10,7,4))): self.assertAlmostEqual(a,b)
    def test_bad_parameters(self):
        for args in [(True,0,0),(1,0,0),(101,0,0),(10,-1,0),(10,11,0),(10,2,11),(10,2,-1),(10,2,1.0)]:
            with self.subTest(args=args),self.assertRaises(ValueError): demo.distribution(*args)
    def test_turnover_order(self):
        self.assertLess(demo.expected_msd(40,20,1),demo.expected_msd(40,20,5))
        self.assertLess(demo.expected_msd(40,20,5),demo.expected_msd(40,20,40))

class Parsing(unittest.TestCase):
    good='DATA,0,0,2\nDATA,1,0,3\n'
    def test_good(self): self.assertEqual(demo.parse(self.good,4,2,1,1),[3])
    def test_missing(self):
        with self.assertRaises(ValueError): demo.parse('',4,2,1,1)
    def test_duplicate(self):
        with self.assertRaises(ValueError): demo.parse(self.good+self.good,4,2,1,1)
    def test_bad_founder(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('DATA,0,0,2','DATA,0,0,1'),4,2,1,1)
    def test_impossible(self):
        with self.assertRaises(ValueError): demo.parse(self.good.replace('DATA,1,0,3','DATA,1,0,4'),4,2,1,1)
    def test_width(self):
        with self.assertRaises(ValueError): demo.parse('DATA,0,0\n',4,2,1,1)
    def test_bad_replicate(self):
        with self.assertRaises(ValueError): demo.parse('DATA,0,1,2\n',4,2,1,1)
    def test_bad_tick(self):
        with self.assertRaises(ValueError): demo.parse('DATA,2,0,2\n',4,2,1,1)
    def test_short_summary(self):
        with self.assertRaises(ValueError): demo.summarize([2],4,2,1)
    def test_bad_count(self):
        with self.assertRaises(ValueError): demo.summarize([2,4],4,2,1)

class Engine(unittest.TestCase):
    def test_seed_replay(self):
        a=demo.run(n=10,k=5,m=2,replicates=16,seed=1234)
        self.assertEqual(a,demo.run(n=10,k=5,m=2,replicates=16,seed=1234))
    def test_engine_controls(self):
        for k,m in [(0,5),(10,5),(5,0)]:
            self.assertEqual(demo.run(n=10,k=k,m=m,replicates=16),[k]*16)
    def test_full_replacement_support(self):
        rows=demo.run(n=10,k=5,m=10,replicates=32)
        self.assertTrue(all(0<=x<=10 for x in rows))
    def test_invalid_seed(self):
        with self.assertRaises(ValueError): demo.run(seed=0)

if __name__=='__main__': unittest.main()
