from fractions import Fraction
import itertools
import json
import subprocess
import sys
import unittest

import allel
import numpy as np

from examples.inference_showcase.garud_phase import demo


class TestGarudPhase(unittest.TestCase):
    def assert_reference(self, h):
        actual = demo.estimate(h)
        for name, exact in demo.pair_reference(h).items():
            self.assertAlmostEqual(actual['scikit_allel'][name], float(exact), places=14)

    def test_actual_values(self):
        expected = ((.5, 1., 1., .5), (.25, .375, .625, .75),
                    (.125, .15625, .21875, .875))
        for h, values in zip(demo.fixtures().values(), expected):
            self.assertEqual(tuple(demo.estimate(h)['scikit_allel'].values()), values)

    def test_all_genotypes_identical(self):
        expected = np.broadcast_to([0, 1], (3, 20, 2))
        for h in demo.fixtures().values():
            np.testing.assert_array_equal(demo.unphased_genotypes(h), expected)

    def test_allele_counts_unchanged(self):
        for h in demo.fixtures().values():
            np.testing.assert_array_equal(h.sum(axis=1), [20, 20, 20])

    def test_exact_fixture_pair_reference(self):
        for h in demo.fixtures().values():
            self.assert_reference(h)

    def test_all_two_locus_multisets(self):
        labels = list(itertools.product((0, 1), repeat=2))
        checked = 0
        for n in range(2, 9):
            for sample in itertools.combinations_with_replacement(labels, n):
                h = np.array(sample, dtype=np.int8).T
                self.assert_reference(h)
                scores = tuple(demo.estimate(h)['scikit_allel'].values())
                self.assertLessEqual(scores[0], scores[1] + 1e-14)
                self.assertLessEqual(scores[1], scores[2] + 1e-14)
                self.assertLessEqual(scores[2], 1 + 1e-14)
                checked += 1
        self.assertEqual(checked, 490)

    def test_all_small_three_locus_phase_assignments(self):
        checked = 0
        for n in range(1, 5):
            for assignment in itertools.product(range(4), repeat=n):
                h = demo.phase_panel([assignment.count(i) for i in range(4)])
                np.testing.assert_array_equal(demo.unphased_genotypes(h),
                                              np.broadcast_to([0, 1], (3, n, 2)))
                self.assert_reference(h)
                checked += 1
        self.assertEqual(checked, 340)

    def test_unequal_counts_and_sorting(self):
        h = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        h = np.repeat(h, [1, 2, 3, 4], axis=1)
        r = demo.estimate(h)['scikit_allel']
        self.assertAlmostEqual(r['H1'], .3)
        self.assertAlmostEqual(r['H12'], .54)
        self.assertAlmostEqual(r['H123'], .82)
        self.assertAlmostEqual(r['H2_H1'], 7 / 15)
        self.assert_reference(h)

    def test_all_1771_phase_count_vectors_and_sharp_h12_bounds(self):
        observed = []
        for a in range(21):
            for b in range(21 - a):
                for c in range(21 - a - b):
                    counts = [a, b, c, 20 - a - b - c]
                    h12 = float(allel.garud_h(demo.phase_panel(counts))[1])
                    expected = Fraction(sum(k * k for k in counts) + max(counts)**2, 800)
                    self.assertAlmostEqual(h12, float(expected), places=14)
                    observed.append(h12)
        self.assertEqual(len(observed), 1771)
        self.assertEqual(min(observed), 5 / 32)
        self.assertEqual(max(observed), 1.)

    def test_naive_reshape_is_not_phase_inference(self):
        h = demo.fixtures()['eight_haplotypes']
        unphased = allel.GenotypeArray(demo.unphased_genotypes(h))
        self.assertIsNone(unphased.is_phased)
        reshaped = unphased.to_haplotypes()
        np.testing.assert_array_equal(reshaped, demo.fixtures()['two_haplotypes'])
        self.assertFalse(np.array_equal(reshaped, h))
        self.assertEqual(float(allel.garud_h(reshaped)[1]), 1.)

    def test_single_haplotype_is_defined(self):
        scores = demo.estimate(np.zeros((3, 10), dtype=np.int8))['scikit_allel']
        self.assertEqual(tuple(scores.values()), (1., 1., 1., 0.))

    def test_single_variant(self):
        self.assert_reference([[0, 0, 0, 1, 1]])

    def test_all_unique_haplotypes(self):
        h = np.array(list(itertools.product((0, 1), repeat=3))).T
        self.assertEqual(tuple(demo.estimate(h)['scikit_allel'].values()),
                         (1 / 8, 5 / 32, 7 / 32, 7 / 8))

    def test_sample_h1_is_with_replacement(self):
        h = demo.fixtures()['two_haplotypes']
        columns = list(map(tuple, h.T))
        pairs = list(itertools.combinations(columns, 2))
        distinct = Fraction(sum(a == b for a, b in pairs), len(pairs))
        self.assertEqual(distinct, Fraction(19, 39))
        self.assertEqual(demo.pair_reference(h)['H1'], Fraction(1, 2))
        self.assertNotEqual(distinct, demo.pair_reference(h)['H1'])

    def test_global_chromosome_permutation(self):
        h = demo.phase_panel([1, 2, 3, 4])
        self.assertEqual(demo.estimate(h), demo.estimate(h[:, ::-1]))

    def test_whole_chromosome_swap_within_diploid(self):
        h = demo.phase_panel([1, 2, 3, 4])
        swapped = h.reshape(3, -1, 2)[:, :, ::-1].reshape(3, -1)
        self.assertEqual(demo.estimate(h), demo.estimate(swapped))
        np.testing.assert_array_equal(demo.unphased_genotypes(h), demo.unphased_genotypes(swapped))

    def test_site_permutation(self):
        h = demo.phase_panel([1, 2, 3, 4])
        self.assertEqual(demo.estimate(h)['scikit_allel'], demo.estimate(h[::-1])['scikit_allel'])

    def test_sitewise_allele_reversal(self):
        h = demo.phase_panel([1, 2, 3, 4])
        changed = h.copy()
        changed[1] = 1 - changed[1]
        self.assertEqual(demo.estimate(h)['scikit_allel'], demo.estimate(changed)['scikit_allel'])

    def test_monomorphic_padding(self):
        h = demo.phase_panel([1, 2, 3, 4])
        padded = np.vstack((np.zeros((2, h.shape[1]), dtype=np.int8), h))
        self.assertEqual(demo.estimate(h)['scikit_allel'], demo.estimate(padded)['scikit_allel'])

    def test_duplicate_site_not_new_haplotype(self):
        h = demo.phase_panel([1, 2, 3, 4])
        self.assertEqual(demo.estimate(h)['scikit_allel'], demo.estimate(np.repeat(h, 3, axis=0))['scikit_allel'])

    def test_uniform_sample_replication(self):
        h = demo.phase_panel([1, 2, 3, 4])
        self.assertEqual(demo.estimate(h)['scikit_allel'], demo.estimate(np.tile(h, 3))['scikit_allel'])

    def test_input_not_mutated(self):
        h = demo.fixtures()['eight_haplotypes']
        before = h.copy()
        demo.estimate(h)
        demo.unphased_genotypes(h)
        np.testing.assert_array_equal(before, h)

    def test_reject_empty_or_wrong_shape(self):
        for h in ([], [0, 1], np.zeros((0, 4), dtype=int), np.zeros((2, 0), dtype=int), [[[0, 1]]]):
            with self.subTest(h=str(h)), self.assertRaises(ValueError):
                demo.estimate(h)

    def test_reject_one_chromosome(self):
        with self.assertRaises(ValueError):
            demo.estimate([[0], [1]])

    def test_small_example_oracle_limit(self):
        with self.assertRaisesRegex(ValueError, '256'):
            demo.estimate(np.zeros((3, 257), dtype=np.int8))

    def test_reject_missing_and_nonbinary(self):
        for x in (-1, 2, 2**63):
            with self.assertRaises(ValueError):
                demo.estimate(np.array([[0, x]], dtype=np.int64 if x < 2**63 else np.uint64))

    def test_reject_noninteger(self):
        for h in ([[0., 1.]], [[False, True]], [['0', '1']], [[None, 1]]):
            with self.assertRaises(ValueError):
                demo.estimate(h)

    def test_reject_odd_diploid_conversion(self):
        with self.assertRaises(ValueError):
            demo.unphased_genotypes([[0, 1, 0]])
        self.assert_reference([[0, 1, 0]])  # Valid haploid sample for garud_h.

    def test_reject_invalid_pair_counts(self):
        for counts in ([1, 2], [0, 0, 0, 0], [-1, 2, 0, 0], [1., 0, 0, 0],
                       [True, False, False, False], [10001, 0, 0, 0]):
            with self.assertRaises(ValueError):
                demo.phase_panel(counts)

    def test_tool_call_not_reference_substitution(self):
        from unittest.mock import patch
        sentinel = (.11, .22, .33, .44)
        with patch.object(allel, 'garud_h', return_value=sentinel) as call:
            actual = demo.estimate(demo.fixtures()['two_haplotypes'])
            call.assert_called_once()
            self.assertEqual(tuple(actual['scikit_allel'].values()), sentinel)

    def test_sample(self):
        self.assertEqual(demo.run(), json.loads(demo.SAMPLE.read_text()))

    def test_cli(self):
        p = subprocess.run([sys.executable, '-m', demo.__name__, '--check-sample'],
                           check=True, capture_output=True, text=True)
        self.assertEqual(p.stdout, demo.SAMPLE.read_text())

    def test_json_finite(self):
        json.dumps(demo.run(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
