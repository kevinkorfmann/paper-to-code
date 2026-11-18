"""Independent likelihood and rational-mixture controls."""
import json
import math
import unittest
from fractions import Fraction
from unittest.mock import patch
import numpy as np
from . import demo

SHAPE=[120,60,40,30]
COUNTS=[102,56,44,48]

class TestMisid(unittest.TestCase):
    def test_unfolded_fit(self):
        self.assertEqual(demo.scan(COUNTS,SHAPE)['grid_maximizers'],[.2])
    def test_zero_error(self):
        self.assertEqual(demo.scan(SHAPE,SHAPE)['grid_maximizers'],[0.])
    def test_upper_boundary(self):
        counts=[75,50,50,75]
        self.assertEqual(demo.scan(counts,SHAPE)['grid_maximizers'],[.5])
    def test_folded_flat(self):
        r=demo.scan(COUNTS,SHAPE,folded=True)
        self.assertEqual(len(r['grid_maximizers']),51)
        self.assertLess(r['log_likelihood_range'],1e-9)
    def test_symmetric_flat(self):
        self.assertEqual(len(demo.scan(COUNTS,[1,2,2,1])['grid_maximizers']),51)
    def test_model_scale_invariance(self):
        a=demo.scan(COUNTS,SHAPE)['likelihood_scores']
        b=demo.scan(COUNTS,[v*7 for v in SHAPE])['likelihood_scores']
        np.testing.assert_allclose(a,b,atol=1e-10)
    def test_reverse_both(self):
        self.assertEqual(demo.scan(COUNTS[::-1],SHAPE[::-1])['grid_maximizers'],[.2])
    def test_exact_mixtures(self):
        for n in range(3,10):
            s=list(range(1,n))
            for numerator in range(11):
                p=Fraction(numerator,10)
                expected=[float((1-p)*a+p*b) for a,b in zip(s,s[::-1])]
                np.testing.assert_allclose(demo.misorient(s,float(p)).compressed(),expected,atol=1e-12)
    def test_mass_preserved(self):
        for p in (0,.1,.5,.9,1):
            self.assertAlmostEqual(float(demo.misorient(SHAPE,p).sum()),250)
    def test_fold_invariance_even_and_odd(self):
        for n in range(3,10):
            shape=list(range(1,n))
            base=demo.spectrum(shape).fold()
            for p in (0,.1,.5,.9,1):
                np.testing.assert_allclose(demo.misorient(shape,p).fold().compressed(),base.compressed())
    def test_independent_poisson_scores(self):
        scores=demo.scan(COUNTS,SHAPE)['likelihood_scores']
        for j,actual in enumerate(scores):
            p=Fraction(j,100)
            means=[float((1-p)*a+p*b) for a,b in zip(SHAPE,SHAPE[::-1])]
            expected=sum(k*math.log(mu)-mu-math.lgamma(k+1) for k,mu in zip(COUNTS,means))
            self.assertAlmostEqual(actual,expected,places=10)
    def test_input_not_mutated(self):
        a,b=COUNTS.copy(),SHAPE.copy()
        demo.scan(a,b)
        self.assertEqual(a,COUNTS); self.assertEqual(b,SHAPE)
    def test_masked_endpoints(self):
        self.assertTrue(demo.misorient(SHAPE,.2).mask[[0,-1]].all())
    def test_bad_rates(self):
        for p in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): demo.misorient(SHAPE,p)
    def test_bad_spectra(self):
        for s in ([],[1],[1,0],[1,-1],[1,float('nan')],[[1,2],[3,4]]):
            with self.assertRaises(ValueError): demo.spectrum(s)
    def test_size_mismatch(self):
        with self.assertRaises(ValueError): demo.scan(COUNTS,[1,2])
    def test_fixture_counts(self):
        f=json.loads((demo.HERE/'synthetic_sfs.json').read_text())
        self.assertEqual(f['observed_counts'],COUNTS)
        self.assertEqual(sum(COUNTS),sum(SHAPE))
    def test_saved_output(self):
        self.assertEqual(demo.report(),json.loads((demo.HERE/'sample_output.json').read_text()))
    def test_real_dadi_called(self):
        real=demo.dadi.Numerics.apply_anc_state_misid
        with patch.object(demo.dadi.Numerics,'apply_anc_state_misid',wraps=real) as wrapped:
            demo.scan(COUNTS,SHAPE)
            self.assertEqual(wrapped.call_count,51)

if __name__=='__main__': unittest.main()
