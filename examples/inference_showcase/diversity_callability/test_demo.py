import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest
import warnings
import numpy as np
from examples.inference_showcase.diversity_callability import demo


class DiversityTests(unittest.TestCase):
    def setUp(self):
        self.pos = [10, 15, 20]
        self.ac = [[3, 1], [2, 2], [1, 3]]
        self.mask = np.ones(100, dtype=bool)

    def estimate(self, pos=None, ac=None, mask=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return demo.estimate(self.pos if pos is None else pos, self.ac if ac is None else ac,
                                 self.mask if mask is None else mask)

    def test_exact_full(self):
        self.assertAlmostEqual(self.estimate()['pi'], 1/60, places=11)

    def test_callable_twenty(self):
        self.assertAlmostEqual(self.estimate(mask=np.arange(100)<20)['pi'], 1/12, places=11)

    def test_snp_only(self):
        mask = np.zeros(100, dtype=bool); mask[np.array(self.pos)-1] = True
        self.assertAlmostEqual(self.estimate(mask=mask)['pi'], 5/9, places=11)

    def test_mask_removes_numerator(self):
        mask = np.arange(100)<20; mask[14] = False
        result = self.estimate(mask=mask)
        self.assertEqual(result['retained_records'], 2)
        self.assertAlmostEqual(result['pi'], 1/19, places=11)

    def test_zero_callable_is_null(self):
        self.assertIsNone(self.estimate(mask=np.zeros(100, dtype=bool))['pi'])

    def test_empty_records_rejected(self):
        with self.assertRaises(ValueError):
            self.estimate(pos=np.array([], dtype=int), ac=np.empty((0,2), dtype=int))

    def test_monomorphic_padding(self):
        self.assertEqual(self.estimate(pos=[1,10,15,20,100], ac=[[4,0]]+self.ac+[[0,4]])['pi'], self.estimate()['pi'])

    def test_allele_reversal(self):
        self.assertEqual(self.estimate(ac=np.array(self.ac)[:,::-1])['pi'], self.estimate()['pi'])

    def test_multiallelic(self):
        self.assertAlmostEqual(self.estimate(pos=[1],ac=[[2,1,1]])['pi'], 5/600, places=11)

    def test_boundary_coordinates(self):
        result = self.estimate(pos=[1,100],ac=[[1,1],[1,1]])
        self.assertEqual(result['window_1based_inclusive'], [1,100])
        self.assertEqual(result['pi'], .02)

    def test_all_small_masks_and_counts(self):
        # 33 allele count tables (n=2..7) x all 8 masks; two records and one invariant base.
        for n in range(2,8):
            for k in range(n+1):
                ac = [[n-k,k],[1,n-1]]
                for mask in itertools.product([False,True],repeat=3):
                    expected = demo.pair_oracle([1,3],ac,mask)
                    actual = self.estimate(pos=[1,3],ac=ac,mask=np.array(mask))['pi']
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual,float(expected),places=11)

    def test_invalid_positions(self):
        for p in ([0,15,20],[10,15,101],[15,10,20],[10,10,20],[10.,15.,20.]):
            with self.assertRaises(ValueError): self.estimate(pos=p)

    def test_invalid_masks(self):
        for mask in ([],[1,1,1],[[True,False]]):
            with self.assertRaises(ValueError): self.estimate(mask=mask)

    def test_invalid_counts(self):
        for ac in ([[3,-1],[2,2],[1,3]],[[3.,1.],[2.,2.],[1.,3.]],[[4],[4],[4]],[[4,0]],[[0,1]]*3,[[3,1],[1,1],[1,3]],[[1000001,0]]*3):
            with self.assertRaises(ValueError): self.estimate(ac=ac)

    def test_inputs_not_modified(self):
        p,a,m=np.array(self.pos),np.array(self.ac),self.mask.copy()
        self.estimate(p,a,m)
        np.testing.assert_array_equal(p,self.pos); np.testing.assert_array_equal(a,self.ac)
        np.testing.assert_array_equal(m,self.mask)

    def test_expected_warning(self):
        with self.assertWarnsRegex(UserWarning,'inaccessible'):
            demo.estimate(self.pos,self.ac,np.zeros(100,dtype=bool))

    def test_default_span(self):
        self.assertAlmostEqual(demo.run()['default_variant_span_pi'],5/33,places=11)

    def test_sample(self):
        self.assertEqual(demo.run(),json.loads(demo.SAMPLE.read_text()))

    def test_cli(self):
        p=subprocess.run([sys.executable,'-m','examples.inference_showcase.diversity_callability.demo','--check-sample'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('matches',p.stdout)

    def test_oracle_bound(self):
        with self.assertRaises(ValueError): demo.pair_oracle([1],[[33,1]],[True])


if __name__ == '__main__':
    unittest.main()
