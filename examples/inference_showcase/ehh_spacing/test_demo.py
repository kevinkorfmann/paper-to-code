"""Independent exact EHH counts and invariance/spacing controls."""
from fractions import Fraction
import itertools
import json
import unittest
import numpy as np
from examples.inference_showcase.ehh_spacing import demo as d


class TestEHH(unittest.TestCase):
    def test_exact_panel(self):
        expected = [Fraction(1), Fraction(3,7), Fraction(1,7), Fraction(0)]
        self.assertEqual(d.exact_pair_oracle(d.panel()), expected)
        self.assertEqual(d.prefix_count_oracle(d.panel()), expected)
        np.testing.assert_allclose(d.estimate(d.panel()), list(map(float, expected)), rtol=0, atol=1e-15)

    def test_exhaustive_binary_panels(self):
        count = 0
        for n in (2,3,4):
            for values in itertools.product((0,1), repeat=3*n):
                h = np.array(values, dtype=np.int8).reshape(3,n)
                oracle = d.exact_pair_oracle(h)
                self.assertEqual(oracle, d.prefix_count_oracle(h))
                np.testing.assert_allclose(d.estimate(h), list(map(float,oracle)), rtol=0, atol=1e-15)
                count += 1
        self.assertEqual(count, 4672)

    def test_column_permutation(self):
        h = d.panel()
        np.testing.assert_array_equal(d.estimate(h[:,[7,1,4,2,0,3,6,5]]),d.estimate(h))

    def test_allele_relabeling(self):
        h = d.panel(); h[[0,2]] = 1-h[[0,2]]
        np.testing.assert_array_equal(d.estimate(h),d.estimate(d.panel()))

    def test_monotone_curve(self):
        self.assertTrue(np.all(np.diff(d.estimate(d.panel()))<=0))

    def test_identical(self):
        np.testing.assert_array_equal(d.estimate(np.zeros((5,4),dtype=int)),np.ones(5))

    def test_distinct_core(self):
        np.testing.assert_array_equal(d.estimate([[0,1],[0,0]]),[0,0])

    def test_pair_without_replacement(self):
        # All eight complete haplotypes distinct: zero, not sum(f_i**2)=1/8.
        self.assertEqual(d.estimate(d.panel())[-1], 0)
        self.assertAlmostEqual(d.estimate(np.repeat(d.panel(),2,axis=1))[-1],1/15)

    def test_marker_removal(self):
        h=d.panel(); orig=d.estimate(h); sparse=d.estimate(h[[0,1,3]])
        np.testing.assert_allclose(sparse,[1,3/7,1/7],rtol=0,atol=1e-15)
        self.assertGreater(sparse[-1],orig[-1])

    def test_truncated_window(self):
        y=d.estimate(d.panel()[:3]); self.assertEqual(y[-1],1/7)
        self.assertLess(d.finite_area(y,[0,100,200]),d.finite_area(d.estimate(d.panel()),[0,100,200,300]))

    def test_nonuniform_area_exact(self):
        y=d.exact_pair_oracle(d.panel()); pos=[0,3,17,41]
        exact=sum((y[i]+y[i+1])*Fraction(pos[i+1]-pos[i],2) for i in range(3))
        self.assertAlmostEqual(d.finite_area(d.estimate(d.panel()),pos),float(exact),places=13)

    def test_stretch_and_units(self):
        y=d.estimate(d.panel())
        self.assertAlmostEqual(d.finite_area(y,[0,100,200,300]),float(Fraction(750,7)))
        self.assertAlmostEqual(d.finite_area(y,[0,1000,2000,3000]),float(Fraction(7500,7)))
        self.assertAlmostEqual(d.finite_area(y,[0,.1,.2,.3])*1000,d.finite_area(y,[0,100,200,300]))

    def test_translation(self):
        y=d.estimate(d.panel())
        self.assertEqual(d.finite_area(y,[100,200,300,400]),d.finite_area(y,[0,100,200,300]))

    def test_single_marker_area(self):
        self.assertEqual(d.finite_area([1],[3]),0)

    def test_no_mutation(self):
        h=d.panel(); before=h.copy(); d.estimate(h); np.testing.assert_array_equal(h,before)

    def test_invalid_haplotypes(self):
        for h in ([], [0,1], [[0]], [[0.,1.]], [[0,-1]], [[0,2]], [[True,False]], np.zeros((1,100001),dtype=int)):
            with self.subTest(h=str(h)[:60]), self.assertRaises(ValueError): d.estimate(h)

    def test_invalid_positions(self):
        for pos in ([0,0], [2,1], [0,float('nan')], [0,float('inf')], [0], [['a','b']], ['a','b'], [True,False]):
            with self.subTest(pos=pos), self.assertRaises(ValueError): d.finite_area([1,0],pos)

    def test_invalid_curves(self):
        for y in ([], [1,-.1], [1,1.1], [1,float('nan')], [[1,0]]):
            with self.subTest(y=y), self.assertRaises(ValueError): d.finite_area(y,[0,1])

    def test_sample_output(self):
        self.assertEqual(d.run(),json.loads((d.HERE/'sample_output.json').read_text()))

    def test_versions(self):
        self.assertEqual(d.run()['versions'],{'python':'3.12.11','scikit_allel':'1.3.13','numpy':'2.3.5'})


if __name__=='__main__':unittest.main()
