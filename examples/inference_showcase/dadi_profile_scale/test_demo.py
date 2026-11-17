"""Independent rational scale and Poisson-likelihood controls."""
from fractions import Fraction
import itertools
import json
import math
import subprocess
import sys
import unittest

import dadi
import numpy as np
from examples.inference_showcase.dadi_profile_scale import demo


def oracle(counts, shape, excluded=()):
    pairs = [(k,Fraction(m)) for i,(k,m) in enumerate(zip(counts,shape),1) if i not in excluded]
    scale = Fraction(sum(k for k,_ in pairs),1)/sum(m for _,m in pairs)
    means = [float(scale*m) for _,m in pairs]
    ll = math.fsum(k*math.log(mu)-mu-math.lgamma(k+1) for (k,_),mu in zip(pairs,means))
    return float(scale),means,ll


class TestProfileScale(unittest.TestCase):
    def setUp(self):
        self.counts = [60,30,20,15]
        self.shape = [1,1/2,1/3,1/4]

    def test_exact_fit(self):
        fitted = demo.fit(self.counts,self.shape)
        self.assertAlmostEqual(fitted['scale'],60)
        np.testing.assert_allclose(fitted['fitted_means'],self.counts,atol=1e-12,rtol=0)

    def test_model_scale_invariance(self):
        original = demo.fit(self.counts,self.shape)
        for multiplier in (0.001,0.1,2,10,1000):
            fitted=demo.fit(self.counts,[multiplier*x for x in self.shape])
            self.assertAlmostEqual(fitted['scale']*multiplier,original['scale'],places=10)
            np.testing.assert_allclose(fitted['fitted_means'],original['fitted_means'],atol=1e-11,rtol=0)
            self.assertAlmostEqual(fitted['profile_poisson_log_likelihood'],original['profile_poisson_log_likelihood'],places=10)

    def test_exhaustive_small_count_oracle(self):
        cases=0
        shape=[Fraction(1),Fraction(1,2),Fraction(1,3)]
        for counts in itertools.product(range(5),repeat=3):
            if not sum(counts):
                continue
            for multiplier in (Fraction(1,10),Fraction(1),Fraction(10)):
                model=[x*multiplier for x in shape]
                fitted=demo.fit(list(counts),list(map(float,model)))
                scale,means,ll=oracle(counts,model)
                self.assertAlmostEqual(fitted['scale'],scale,places=11)
                self.assertAlmostEqual(fitted['profile_poisson_log_likelihood'],ll,places=11)
                np.testing.assert_allclose(fitted['fitted_means'],means,atol=1e-12,rtol=0)
                cases+=1
        self.assertEqual(cases,372)

    def test_scale_is_poisson_maximum(self):
        model,data=demo.spectra([12,3,5,1],self.shape)
        scale=dadi.Inference.optimal_sfs_scaling(model,data)
        best=dadi.Inference.ll(scale*model,data)
        for factor in (0.1,0.5,0.99,1.01,2,10):
            self.assertLess(dadi.Inference.ll(scale*factor*model,data),best)

    def test_profile_matches_scaled_poisson(self):
        model,data=demo.spectra(self.counts,self.shape)
        self.assertAlmostEqual(dadi.Inference.ll_multinom(model,data),dadi.Inference.ll(60*model,data),places=11)

    def test_multinomial_differs_by_data_constant(self):
        total=sum(self.counts)
        probs=np.array(self.shape)/sum(self.shape)
        conditional=math.lgamma(total+1)+sum(k*math.log(p)-math.lgamma(k+1) for k,p in zip(self.counts,probs))
        constant=-total+total*math.log(total)-math.lgamma(total+1)
        fitted=demo.fit(self.counts,self.shape)
        self.assertAlmostEqual(fitted['profile_poisson_log_likelihood'],conditional+constant,places=10)

    def test_shape_still_matters(self):
        actual=demo.fit(self.counts,self.shape)
        flat=demo.fit(self.counts,[1,1,1,1])
        self.assertGreater(actual['profile_poisson_log_likelihood'],flat['profile_poisson_log_likelihood'])
        np.testing.assert_allclose(flat['fitted_means'],[31.25]*4)

    def test_data_scaling_changes_scale(self):
        for factor in (2,3,10):
            fitted=demo.fit([factor*x for x in self.counts],self.shape)
            self.assertAlmostEqual(fitted['scale'],60*factor)
            np.testing.assert_allclose(fitted['fitted_means'],[factor*x for x in self.counts])

    def test_singleton_mask(self):
        counts=[120,30,20,15]
        self.assertAlmostEqual(demo.fit(counts,self.shape)['scale'],88.8)
        fitted=demo.fit(counts,self.shape,(1,))
        self.assertAlmostEqual(fitted['scale'],60)
        self.assertIsNone(fitted['fitted_means'][0])
        self.assertEqual(fitted['active_bins'],[2,3,4])

    def test_mask_intersection(self):
        for masked_side in ('data','model'):
            model,data=demo.spectra([120,30,20,15],self.shape)
            target=data if masked_side=='data' else model
            target.mask[1]=True
            self.assertAlmostEqual(dadi.Inference.optimal_sfs_scaling(model,data),60)
            self.assertAlmostEqual(float(dadi.Inference.ll_multinom(model,data)),demo.fit([120,30,20,15],self.shape,(1,))['profile_poisson_log_likelihood'],places=10)

    def test_masked_oracle(self):
        for excluded in ((1,),(2,),(1,3),(2,3,4)):
            fitted=demo.fit(self.counts,self.shape,excluded)
            scale,means,ll=oracle(self.counts,[Fraction(1,i) for i in range(1,5)],excluded)
            self.assertAlmostEqual(fitted['scale'],scale,places=11)
            self.assertAlmostEqual(fitted['profile_poisson_log_likelihood'],ll,places=11)
            np.testing.assert_allclose([x for x in fitted['fitted_means'] if x is not None],means,atol=1e-12,rtol=0)

    def test_bin_permutation(self):
        original=demo.fit([10,2,0,5],self.shape)
        order=[3,0,2,1]
        permuted=demo.fit([[10,2,0,5][i] for i in order],[self.shape[i] for i in order])
        self.assertAlmostEqual(permuted['scale'],original['scale'])
        self.assertAlmostEqual(permuted['profile_poisson_log_likelihood'],original['profile_poisson_log_likelihood'])

    def test_zero_count_bins_are_kept(self):
        fitted=demo.fit([10,0],[1,1])
        np.testing.assert_allclose(fitted['fitted_means'],[5,5])
        self.assertEqual(fitted['active_bins'],[1,2])

    def test_invalid_counts(self):
        for counts in ([0,0],[-1,3],[1.5,2],[True,2],[1],[]):
            with self.assertRaises(ValueError):
                demo.fit(counts,[1]*len(counts))

    def test_invalid_model(self):
        for shape in ([1],[0,1],[-1,1],[math.inf,1],[math.nan,1],[True,1]):
            with self.assertRaises(ValueError):
                demo.fit([1,2],shape)

    def test_invalid_masks(self):
        for mask in ((0,),(5,),(1,1),(1.0,),(True,),(1,2,3,4)):
            with self.assertRaises(ValueError):
                demo.fit(self.counts,self.shape,mask)
        with self.assertRaises(ValueError):
            demo.fit([1,0],[1,1],(1,))

    def test_corners_and_polarity(self):
        model,data=demo.spectra(self.counts,self.shape)
        self.assertEqual(model.sample_sizes.tolist(),[5])
        self.assertFalse(data.folded)
        self.assertTrue(data.mask[0] and data.mask[-1])
        self.assertFalse(any(data.mask[1:-1]))

    def test_sample_and_versions(self):
        output=demo.report()
        self.assertEqual(output,json.loads((demo.HERE/'sample_output.json').read_text()))
        self.assertEqual(output['versions'],{'python':'3.12.11','dadi':'2.4.4','numpy':'2.5.4'})

    def test_cli(self):
        result=subprocess.run([sys.executable,'-m','examples.inference_showcase.dadi_profile_scale.demo','--check-sample'],check=True,capture_output=True,text=True)
        self.assertEqual(result.stdout,'Checked sample matches\n')

if __name__=='__main__':
    unittest.main()
