import unittest
import numpy as np
from examples.inference_showcase.dadi_joint_folding.demo import spectra, oracle, fixtures, run

class FoldingTests(unittest.TestCase):
    def test_all_cells_even(self):
        for i in range(11):
            for j in range(11):
                p=[[i,j]]; np.testing.assert_array_equal(spectra(p)[1].filled(0),oracle(p))
    def test_all_cells_unequal_odd(self):
        for i in range(6):
            for j in range(8):
                p=[[i,j]]; np.testing.assert_array_equal(spectra(p,(5,7))[1].filled(0),oracle(p,(5,7)))
    def test_same_vs_opposite(self):
        a,b=[spectra(fixtures()[k])[1:] for k in ('same_minor_allele','opposite_minor_alleles')]
        np.testing.assert_array_equal(a[1],b[1]);self.assertFalse(np.array_equal(a[0].filled(0),b[0].filled(0)))
    def test_ties_split(self):
        f=spectra([[1,9]])[1];self.assertEqual(f[1,9],.5);self.assertEqual(f[9,1],.5)
    def test_self_complement_tie(self): self.assertEqual(spectra([[5,5]])[1][5,5],1)
    def test_fixed_differences_retained(self): self.assertEqual(spectra([[0,10]])[1].S(),1)
    def test_shared_corners_masked(self): self.assertEqual(spectra([[0,0],[10,10]])[1].S(),0)
    def test_flip_both_invariant(self):
        p=np.array([[1,2],[3,9],[0,10],[4,6]])
        np.testing.assert_array_equal(spectra(p)[1].filled(0),spectra(10-p)[1].filled(0))
    def test_flip_one_not_invariant(self):
        self.assertFalse(np.array_equal(spectra([[1,1]])[1].filled(0),spectra([[1,9]])[1].filled(0)))
    def test_population_swap(self):
        p=np.array([[1,2],[4,5],[0,7]])
        np.testing.assert_array_equal(spectra(p,(5,7))[1].filled(0).T,spectra(p[:,::-1],(7,5))[1].filled(0))
    def test_row_order(self):
        p=[[1,2],[3,7],[9,9]]
        np.testing.assert_array_equal(spectra(p)[1].filled(0),spectra(p[::-1])[1].filled(0))
    def test_mass(self):
        p=[[i,j] for i in range(11) for j in range(11)]
        self.assertEqual(spectra(p)[1].S(),119)
    def test_linearity(self):
        a,b=[[1,2],[2,8]],[[0,10],[9,9]]
        np.testing.assert_array_equal(spectra(a+b)[1].filled(0),spectra(a)[1].filled(0)+spectra(b)[1].filled(0))
    def test_folded_flag(self): self.assertTrue(spectra([[1,2]])[1].folded)
    def test_fold_twice_rejected(self):
        with self.assertRaises(ValueError): spectra([[1,2]])[1].fold()
    def test_bad_counts(self):
        for p in ([],[[1]],[[1.,2.]],[[True,False]],[[-1,2]],[[11,2]]):
            with self.subTest(p=p), self.assertRaises(ValueError): spectra(p)
    def test_bad_sizes(self):
        for n in ((1,10),(10.,10),(True,10),(10,101),(10,)):
            with self.subTest(n=n),self.assertRaises(ValueError): spectra([[1,1]],n)
    def test_sample(self):
        import json
        from examples.inference_showcase.dadi_joint_folding.demo import SAMPLE
        self.assertEqual(run(),json.loads(SAMPLE.read_text()))

if __name__=='__main__':unittest.main()
