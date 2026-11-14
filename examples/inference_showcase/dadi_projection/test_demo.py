import itertools
import json
import unittest
from fractions import Fraction
from math import comb
import dadi
import numpy as np
from examples.inference_showcase.dadi_projection.demo import (
    project, exact_projection, panel, statistics, run, SAMPLE)


class ProjectionTests(unittest.TestCase):
    def test_all_one_site_spectra(self):
        for n in range(2, 13):
            for m in range(2, n+1):
                for k in range(n+1):
                    project(panel(k,1,n), n, m)

    def test_independent_labeled_subsets(self):
        for n in range(2,9):
            for m in range(2,n+1):
                for k in range(n+1):
                    tally=[0]*(m+1)
                    for subset in itertools.combinations(range(n),m):
                        tally[sum(i<k for i in subset)]+=1
                    expected=[Fraction(c,comb(n,m)) for c in tally]
                    self.assertEqual(expected, exact_projection(panel(k,1,n),n,m))

    def test_singletons_retained_fraction(self):
        self.assertEqual(exact_projection(panel(1),20,4),[80,20,0,0,0])

    def test_derived_fixed_stays_fixed(self):
        np.testing.assert_allclose(project(panel(20))[1].data,[0,0,0,0,100],atol=1e-12)

    def test_absent_stays_absent(self):
        np.testing.assert_allclose(project(panel(0))[1].data,[100,0,0,0,0],atol=1e-12)

    def test_total_mass(self):
        c=list(range(21)); self.assertAlmostEqual(project(c)[1].sum(),sum(c))

    def test_first_moment(self):
        c=list(range(21)); p=project(c)[1]
        self.assertAlmostEqual(sum(j*p[j] for j in range(5)),sum(k*v for k,v in enumerate(c))*4/20)

    def test_pairwise_diversity_preserved(self):
        for k in range(1,20):
            a,b=project(panel(k));self.assertAlmostEqual(statistics(a)['pairwise_difference_total'],statistics(b)['pairwise_difference_total'])

    def test_exact_pairwise_identity(self):
        for n in range(2,15):
            for m in range(2,n+1):
                for k in range(n+1):
                    q=exact_projection(panel(k,1,n),n,m)
                    self.assertEqual(sum(q[j]*Fraction(j*(m-j),comb(m,2)) for j in range(m+1)),Fraction(k*(n-k),comb(n,2)))

    def test_sequential_projection(self):
        a=project(list(range(21)),20,12)[1].project([4])
        b=project(list(range(21)),20,4)[1]
        np.testing.assert_allclose(a.data,b.data,rtol=1e-12,atol=1e-12)

    def test_identity(self):
        c=list(range(21));np.testing.assert_array_equal(project(c,20,20)[1].data,c)

    def test_reversal(self):
        c=list(range(21));np.testing.assert_allclose(project(c)[1].data[::-1],project(c[::-1])[1].data,rtol=1e-12,atol=1e-12)

    def test_linearity(self):
        a,b=panel(1),panel(10)
        np.testing.assert_allclose(project([x+y for x,y in zip(a,b)])[1].data,project(a)[1].data+project(b)[1].data)

    def test_input_unmodified(self):
        c=list(range(21));orig=c[:];project(c);self.assertEqual(c,orig)

    def test_not_integer_resample(self):
        p=project(panel(10,1))[1];self.assertTrue(any(0<x<1 for x in p.data))

    def test_expected_s_not_conditioned(self):
        r=run();self.assertAlmostEqual(r['panels']['mixed']['after']['expected_segregating_sites'],34.266253869969)
        self.assertGreater(r['condition_on_polymorphism_then_renormalize_to_100_sites']['pairwise_difference_total'],r['panels']['mixed']['after']['pairwise_difference_total'])

    def test_empty_spectrum(self):
        a,b=project([0]*21);self.assertEqual(statistics(b)['expected_segregating_sites'],0)

    def test_invalid_sizes(self):
        for n,m in [(1,1),(20,21),(101,4),(20,0),(20.,4),(20,True)]:
            with self.subTest(n=n,m=m),self.assertRaises(ValueError):project([0]*21,n,m)

    def test_invalid_counts(self):
        for c in [[],[0]*20,[0]*20+[-1],[0]*20+[1.5],[0]*20+[True],np.zeros(21)]:
            with self.subTest(c=str(c)),self.assertRaises(ValueError):project(c)

    def test_sample(self):self.assertEqual(run(),json.loads(SAMPLE.read_text()))

if __name__ == '__main__': unittest.main()
