from fractions import Fraction
import itertools
import json
import unittest
import numpy as np
from examples.inference_showcase.net_divergence import demo

class DivergenceTests(unittest.TestCase):
    def test_identical(self):
        r = demo.estimate([[5,5]], [[5,5]])
        self.assertAlmostEqual(r['dxy'][0], 1/2)
        self.assertAlmostEqual(r['net'][0], -1/18)

    def test_fixed_difference(self):
        self.assertEqual(demo.estimate([[10,0]], [[0,10]])['net'][0], 1)

    def test_shared_fixed(self):
        self.assertEqual(demo.estimate([[10,0]], [[10,0]])['dxy'][0], 0)

    def test_exhaustive_small_counts(self):
        for n,m in itertools.product(range(2,8),repeat=2):
            for i,j in itertools.product(range(n+1),range(m+1)):
                a,b=[[n-i,i]],[[m-j,j]]
                actual=demo.estimate(a,b)
                exact=demo.count_oracle(a,b)
                pairs=demo.pair_oracle([0]*(n-i)+[1]*i,[0]*(m-j)+[1]*j)
                for key in actual:
                    self.assertEqual(exact[key][0],pairs[key])
                    self.assertAlmostEqual(actual[key][0],float(pairs[key]),places=12)

    def test_sampling_expectations(self):
        for p,q in itertools.product((Fraction(0),Fraction(1,5),Fraction(1,2),Fraction(1)),repeat=2):
            r=demo.binomial_expectation(4,5,p,q)
            self.assertEqual(r['dxy'], p*(1-q)+q*(1-p))
            self.assertEqual(r['pi_a'],2*p*(1-p))
            self.assertEqual(r['pi_b'],2*q*(1-q))
            self.assertEqual(r['net'],(p-q)**2)

    def test_swap_populations(self):
        a,b=[[8,2],[1,4]],[[3,7],[2,4]]
        x,y=demo.estimate(a,b),demo.estimate(b,a)
        np.testing.assert_array_equal(x['net'],y['net'])
        np.testing.assert_array_equal(x['pi_a'],y['pi_b'])

    def test_reverse_alleles(self):
        a,b=np.array([[8,2],[1,4]]),np.array([[3,7],[2,4]])
        for k,v in demo.estimate(a,b).items():
            np.testing.assert_array_equal(v,demo.estimate(a[:,::-1],b[:,::-1])[k])

    def test_site_permutation(self):
        a,b=np.array([[8,2],[1,4]]),np.array([[3,7],[2,4]])
        np.testing.assert_array_equal(demo.estimate(a,b)['net'][::-1],demo.estimate(a[::-1],b[::-1])['net'])

    def test_duplicate_chromosomes(self):
        a=np.array([[5,5]])
        x,y=demo.estimate(a,a),demo.estimate(a*2,a*2)
        self.assertEqual(x['dxy'][0],y['dxy'][0])
        self.assertNotEqual(x['net'][0],y['net'][0])
        self.assertAlmostEqual(y['net'][0],-1/38)

    def test_bad_shape(self):
        with self.assertRaises(ValueError): demo.estimate([5,5],[[5,5]])

    def test_empty(self):
        with self.assertRaises(ValueError): demo.estimate([],[])

    def test_floats(self):
        with self.assertRaises(ValueError): demo.estimate([[5.,5.]],[[5,5]])

    def test_negative(self):
        with self.assertRaises(ValueError): demo.estimate([[-1,5]],[[5,5]])

    def test_one_chromosome(self):
        with self.assertRaises(ValueError): demo.estimate([[0,1]],[[5,5]])

    def test_unaligned(self):
        with self.assertRaises(ValueError): demo.estimate([[5,5],[2,2]],[[5,5]])

    def test_nonmutation(self):
        a=np.array([[5,5]])
        demo.estimate(a,a)
        np.testing.assert_array_equal(a,[[5,5]])

    def test_exact_expected_zero_not_conditioned_zero(self):
        self.assertEqual(demo.binomial_expectation(10,10,Fraction(1,2),Fraction(1,2))['net'],0)
        self.assertLess(demo.count_oracle([[5,5]],[[5,5]])['net'][0],0)

    def test_sample(self):
        self.assertEqual(demo.summary(),json.loads((demo.HERE/'sample_summary.json').read_text()))

if __name__ == '__main__': unittest.main()
