"""Exact subset and finite-state checks plus executed SLiM controls."""
from fractions import Fraction
from itertools import combinations, product
import math
import unittest
from . import demo as d

class TheoryTests(unittest.TestCase):
    def test_subsets(self):
        for n in range(2,8):
            for k in range(n+1):
                for s in range(1,n+1):
                    counts=[0]*(s+1)
                    for subset in combinations(range(n),s): counts[sum(i<k for i in subset)]+=1
                    for got,c in zip(d.hypergeom(n,k,s),counts): self.assertAlmostEqual(got,float(Fraction(c,math.comb(n,s))))
    def test_distribution_mass(self):
        for k in range(5): self.assertAlmostEqual(sum(d.delta_distribution(4,k,2)),1)
    def test_zero_change_mean(self):
        for k in range(5): self.assertAlmostEqual(sum((i-2)/2*w for i,w in enumerate(d.delta_distribution(4,k,2))),0)
    def test_formula(self):
        for n in (3,5,8):
            for k in range(n+1):
                for s in (1,n):
                    v=sum(((j-s)/s)**2*w for j,w in enumerate(d.delta_distribution(n,k,s)))
                    self.assertAlmostEqual(v,d.theory(n,k,s)['observed_msd'])
    def test_full_census(self):
        self.assertEqual(d.theory(40,20,40)['sampling_msd'],0)
    def test_boundary(self):
        for k in (0,40): self.assertEqual(d.theory(40,k,5)['observed_msd'],0)
    def test_complement(self):
        a=d.delta_distribution(5,1,3); b=d.delta_distribution(5,4,3)
        for x,y in zip(a,reversed(b)): self.assertAlmostEqual(x,y)
    def test_smaller_samples(self):
        vals=[d.theory(40,20,s)['observed_msd'] for s in (1,5,20,40)]
        self.assertEqual(vals,sorted(vals,reverse=True))
    def test_exact_small_path(self):
        # Enumerate all ordered clonal offspring sequences and both one-clone samples.
        n=3; k=1; acc=Fraction(0)
        for seq in product(range(n),repeat=n):
            for a in range(n):
                for b in range(n):
                    delta=int(seq[b]<k)-int(a<k)
                    acc+=Fraction(delta*delta,n**(n+2))
        self.assertAlmostEqual(float(acc),d.theory(n,k,1)['observed_msd'])
    def test_invalid(self):
        for args in ((1,0,1),(3,-1,1),(3,4,1),(3,1,0),(3,1,4),(3.5,1,1),(True,0,1)):
            with self.assertRaises(ValueError): d.theory(*args)

class ParserTests(unittest.TestCase):
    good='DATA,0,0,2,1\nDATA,1,0,3,2\n'
    def test_valid(self): self.assertEqual(d.parse(self.good,4,2,2,1),[((2,1),(3,2))])
    def test_missing(self):
        with self.assertRaises(ValueError): d.parse('DATA,0,0,2,1',4,2,2,1)
    def test_duplicate(self):
        with self.assertRaises(ValueError): d.parse(self.good+self.good,4,2,2,1)
    def test_impossible(self):
        with self.assertRaises(ValueError): d.parse(self.good.replace('1,0,3,2','1,0,4,1'),4,2,2,1)
    def test_bad_founder(self):
        with self.assertRaises(ValueError): d.parse(self.good.replace('0,0,2,1','0,0,1,1'),4,2,2,1)
    def test_bad_time(self):
        with self.assertRaises(ValueError): d.parse(self.good.replace('1,0,3,2','2,0,3,2'),4,2,2,1)
    def test_bad_width(self):
        with self.assertRaises(ValueError): d.parse('DATA,0,0,2',4,2,2,1)

class SimulationTests(unittest.TestCase):
    def test_absent(self): self.assertTrue(all(a==b==(0,0) for a,b in d.run(n=8,k=0,s=3,r=8)))
    def test_fixed(self): self.assertTrue(all(a==b==(8,3) for a,b in d.run(n=8,k=8,s=3,r=8)))
    def test_census(self): self.assertTrue(all(a[0]==a[1] and b[0]==b[1] for a,b in d.run(n=8,k=4,s=8,r=32)))
    def test_seed_replay(self): self.assertEqual(d.run(n=8,k=4,s=3,r=32),d.run(n=8,k=4,s=3,r=32))
    def test_marker_bounds(self): self.assertEqual(len(d.run(n=8,k=1,s=1,r=16)),16)

if __name__=='__main__': unittest.main()
