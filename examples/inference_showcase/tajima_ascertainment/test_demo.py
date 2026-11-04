import itertools
import json
import math
import subprocess
import sys
import unittest

import allel
import numpy as np
from examples.inference_showcase.tajima_ascertainment import demo


class TestTajima(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(demo.run(), json.loads(demo.SAMPLE.read_text()))

    def test_cli(self):
        p = subprocess.run([sys.executable, '-m', demo.__name__, '--check-sample'],
                           check=True, capture_output=True, text=True)
        self.assertEqual(p.stdout, demo.SAMPLE.read_text())

    def test_filter_changes_sign(self):
        f = demo.fixtures()
        self.assertLess(demo.estimate(f['all_sites'])['scikit_allel_D'], 0)
        self.assertGreater(demo.estimate(f['minor_count_at_least_2'])['scikit_allel_D'], 0)
        self.assertEqual(f['minor_count_at_least_2'], [x for x in f['all_sites'] if min(x)>=2])

    def test_exact_pi(self):
        r = demo.reference(demo.fixtures()['all_sites'])
        self.assertAlmostEqual(r['pairwise_differences'], 352/19)
        self.assertEqual(r['segregating_sites'], 100)

    def test_pair_enumeration(self):
        for n in range(4, 21):
            for k in range(n+1):
                h = [0]*(n-k)+[1]*k
                pairs = list(itertools.combinations(h, 2))
                expected = sum(x != y for x,y in pairs)/len(pairs)
                observed = allel.mean_pairwise_difference([[n-k,k]])[0]
                self.assertAlmostEqual(observed, expected, places=14)

    def test_exhaustive_three_site_spectra(self):
        checked = 0
        for n in range(4, 13):
            for ks in itertools.combinations_with_replacement(range(1,n), 3):
                r = demo.estimate([[n-k,k] for k in ks])
                self.assertAlmostEqual(r['scikit_allel_D'], r['reference_D'], places=11)
                checked += 1
        self.assertEqual(checked, 996)

    def test_all_fixture_reference_values(self):
        for a in demo.fixtures().values():
            r = demo.estimate(a)
            if r['reference_D'] is not None:
                self.assertAlmostEqual(r['reference_D'], r['scikit_allel_D'], places=12)

    def test_reversal(self):
        f = demo.fixtures()
        self.assertEqual(demo.estimate(f['all_sites']), demo.estimate(f['alleles_reversed']))

    def test_site_order(self):
        a = demo.fixtures()['all_sites']
        self.assertAlmostEqual(demo.estimate(a)['scikit_allel_D'],
                               demo.estimate(a[::-1])['scikit_allel_D'])

    def test_padding(self):
        f = demo.fixtures()
        self.assertAlmostEqual(demo.estimate(f['all_sites'])['scikit_allel_D'],
                               demo.estimate(f['monomorphic_padding'])['scikit_allel_D'], places=12)

    def test_singletons_negative(self):
        self.assertLess(demo.estimate(demo.fixtures()['singletons_only'])['scikit_allel_D'],0)

    def test_intermediate_positive(self):
        self.assertGreater(demo.estimate(demo.fixtures()['intermediate_only'])['scikit_allel_D'],0)

    def test_zero_sites_null(self):
        self.assertIsNone(demo.estimate([[20,0]]*10)['scikit_allel_D'])

    def test_one_site_null(self):
        self.assertIsNone(demo.estimate([[19,1]])['scikit_allel_D'])

    def test_two_sites_null(self):
        self.assertIsNone(demo.estimate([[19,1]]*2)['scikit_allel_D'])

    def test_three_sites_finite(self):
        self.assertTrue(math.isfinite(demo.estimate([[19,1]]*3)['scikit_allel_D']))

    def test_reject_missing_calls(self):
        with self.assertRaises(ValueError): demo.estimate([[19,1],[18,1],[10,10]])

    def test_reject_empty(self):
        with self.assertRaises(ValueError): demo.estimate([])

    def test_reject_shape(self):
        for a in ([1,2], [[1,2,3]], [[[1,2]]]):
            with self.assertRaises(ValueError): demo.estimate(a)

    def test_reject_negative(self):
        with self.assertRaises(ValueError): demo.estimate([[-1,21]])

    def test_reject_float(self):
        with self.assertRaises(ValueError): demo.estimate([[19.,1.]])

    def test_reject_bool(self):
        with self.assertRaises(ValueError): demo.estimate([[True,False]])

    def test_reject_small_sample(self):
        for n in range(4):
            with self.assertRaises(ValueError): demo.estimate([[n,0]])

    def test_reject_overflow(self):
        with self.assertRaises(ValueError): demo.estimate(np.array([[2**63,1]],dtype=np.uint64))

    def test_fixture_not_mutated(self):
        a = demo.fixtures()['all_sites']
        before = json.dumps(a)
        demo.estimate(a)
        self.assertEqual(json.dumps(a),before)

    def test_json_has_no_nan(self):
        json.dumps(demo.run(), allow_nan=False)


if __name__ == '__main__': unittest.main()
