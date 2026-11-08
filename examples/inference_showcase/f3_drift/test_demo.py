import itertools
import json
import math
from fractions import Fraction as F
import unittest
import numpy as np
from examples.inference_showcase.f3_drift import demo

class Tests(unittest.TestCase):
    def test_intermediate(self):
        self.assertAlmostEqual(demo.estimate(demo.fixture([10,10]))['f3'],float(-F(49,475)))
    def test_endpoints(self):
        self.assertAlmostEqual(demo.estimate(demo.fixture([0,20]))['f3'],.16)
    def test_moderate(self):
        self.assertAlmostEqual(demo.estimate(demo.fixture([4,16]))['f3'],-.16/19)
    def test_undefined_normalized(self):
        self.assertIsNone(demo.estimate(demo.fixture([0,20]))['f3_star'])
    def test_normalized(self):
        r=demo.estimate(demo.fixture([10,10])); self.assertAlmostEqual(r['f3_star'],r['f3']/r['target_heterozygosity_mean'])
    def test_source_swap(self):
        x=np.array(demo.fixture([4,16])); self.assertEqual(demo.estimate(x),demo.estimate(x[:,[0,2,1]]))
    def test_allele_reversal(self):
        x=np.array(demo.fixture([10,10])); self.assertEqual(demo.estimate(x),demo.estimate(x[:,:,::-1]))
    def test_row_order(self):
        x=demo.fixture([4,16]); self.assertEqual(demo.estimate(x),demo.estimate(x[::-1]))
    def test_no_mutation(self):
        x=np.array(demo.fixture([4,16])); y=x.copy(); demo.estimate(x); np.testing.assert_array_equal(x,y)
    def test_pair_oracle_all_small_counts(self):
        checked=0
        for sizes in [(2,2,2),(3,2,4),(4,3,2)]:
            for ks in itertools.product(*(range(n+1) for n in sizes)):
                site=[[n-k,k] for n,k in zip(sizes,ks)]
                self.assertAlmostEqual(demo.estimate([site])['f3'],float(demo.ordered_pair_oracle(site)),places=13)
                checked+=1
        self.assertEqual(checked,147)
    def test_exact_unbiased_binomial_expectation(self):
        # Independently enumerate all count outcomes; tool means must equal population f3.
        for probs in [(F(1,2),F(1,5),F(4,5)),(F(1,5),F(1,5),F(4,5)),(F(0),F(1,4),F(3,4))]:
            for n in [2,3,5]:
                expected=F(0); tool=0.; naive=F(0)
                for ks in itertools.product(range(n+1),repeat=3):
                    weight=math.prod(F(math.comb(n,k))*p**k*(1-p)**(n-k) for k,p in zip(ks,probs))
                    site=[[n-k,k] for k in ks]
                    expected+=weight*demo.ordered_pair_oracle(site)
                    tool+=float(weight)*demo.estimate([site])['f3']
                    c,a,b=map(lambda k:F(k,n),ks); naive+=weight*(c-a)*(c-b)
                c,a,b=probs; truth=(c-a)*(c-b)
                self.assertEqual(expected,truth); self.assertAlmostEqual(tool,float(truth),places=13)
                self.assertEqual(naive,truth+c*(1-c)/n)
    def test_mean_preserving_dispersion_identity(self):
        p,a,b=F(1,2),F(1,5),F(4,5)
        for d in [F(0),F(1,10),F(3,10),F(1,2)]:
            self.assertEqual(sum((c-a)*(c-b) for c in [p-d,p+d])/2,(p-a)*(p-b)+d*d)
    def test_zero_all_sites(self):
        r=demo.estimate([[[2,0]]*3]); self.assertEqual(r['f3'],0); self.assertIsNone(r['f3_star'])
    def test_shape(self):
        for x in [[],[[1,1]],np.zeros((1,4,2),int)]:
            with self.assertRaises(ValueError): demo.estimate(x)
    def test_types(self):
        for dtype in [float,bool,str]:
            with self.assertRaises(ValueError): demo.estimate(np.ones((1,3,2),dtype=dtype))
    def test_missing_and_singleton(self):
        for row in [[0,0],[1,0]]:
            with self.assertRaises(ValueError): demo.estimate([[row,[2,2],[2,2]]])
    def test_bounds(self):
        for v in [-1,1000001,2**63]:
            with self.assertRaises(ValueError): demo.estimate([[[v,2],[2,2],[2,2]]])
    def test_oracle_bound(self):
        with self.assertRaises(ValueError): demo.ordered_pair_oracle([[8,1],[1,1],[1,1]])
    def test_sample(self):
        self.assertEqual(demo.run(),json.loads((demo.HERE/'sample_output.json').read_text()))

if __name__=='__main__': unittest.main()
