"""Exact arithmetic, spectral invariants, metamorphic tests and real CLI checks."""
from fractions import Fraction
import itertools
import json
import math
import unittest

from examples.inference_showcase.plink_pca_weighting import demo as d


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = d.tool(d.panel())
        cls.copied = d.tool(d.panel(2, 4))
        cls.tied = d.tool(d.panel(1, 1))
        cls.a, cls.b = d.axes()

    def test_checked_fixture(self):
        self.assertEqual(json.loads((d.HERE/'synthetic_genotypes.json').read_text()), d.panel())

    def test_checked_summary(self):
        self.assertEqual(json.loads((d.HERE/'sample_summary.json').read_text()), d.run())

    def test_hwe_marginals(self):
        for col in zip(*d.panel().values()):
            self.assertEqual([col.count(i) for i in range(3)], [4, 8, 4])

    def test_factorial_axes_are_orthonormal(self):
        self.assertAlmostEqual(d.dot(self.a, self.b), 0)
        for axis in (self.a, self.b):
            self.assertAlmostEqual(d.dot(axis, axis), 1)
            self.assertAlmostEqual(sum(axis), 0)

    def test_exact_matrix_is_centered_symmetric(self):
        r = d.exact_grm(d.panel())
        self.assertEqual(sum(r[i][i] for i in range(16)), 16)
        for i in range(16):
            self.assertEqual(sum(r[i]), 0)
            for j in range(16):
                self.assertEqual(r[i][j], r[j][i])

    def test_binary_grm_matches_exact_oracle(self):
        self.assertLess(d.matrix_error(self.base['grm'], d.exact_grm(d.panel())), 1e-12)

    def test_printed_eigenvalues_match_analytic(self):
        for observed, expected in zip(self.base['eigenvalues'], (32/3, 16/3)):
            self.assertAlmostEqual(observed, expected, delta=5e-5)

    def test_leading_axis_swaps(self):
        self.assertAlmostEqual(abs(d.dot(self.base['columns'][0], self.a)), 1, places=10)
        self.assertAlmostEqual(abs(d.dot(self.copied['columns'][0], self.b)), 1, places=10)
        self.assertAlmostEqual(d.dot(self.base['columns'][0], self.copied['columns'][0]), 0, places=10)

    def test_eigenvalue_spectrum_unchanged_despite_swap(self):
        self.assertEqual(self.base['eigenvalues'], self.copied['eigenvalues'])
        self.assertGreater(d.matrix_error(self.base['grm'], self.copied['grm']), 1)

    def test_two_dimensional_subspace_unchanged(self):
        self.assertLess(d.matrix_error(d.projector(self.base['columns']),
                                      d.projector(self.copied['columns'])), 2e-6)

    def test_tie_compare_subspace_not_individual_vectors(self):
        self.assertEqual(self.tied['eigenvalues'], [8, 8])
        self.assertLess(d.matrix_error(d.projector(self.tied['columns']),
                                      d.projector([self.a, self.b])), 2e-6)

    def test_rotated_basis_has_same_tied_projector(self):
        rotated = [[(a+b)/math.sqrt(2) for a, b in zip(self.a, self.b)],
                   [(a-b)/math.sqrt(2) for a, b in zip(self.a, self.b)]]
        self.assertLess(d.matrix_error(d.projector(rotated), d.projector([self.a, self.b])), 1e-14)

    def test_uniform_copying_keeps_matrix(self):
        copied = d.tool(d.panel(4, 2))
        self.assertLess(d.matrix_error(copied['grm'], self.base['grm']), 1e-12)
        self.assertEqual(copied['eigenvalues'], self.base['eigenvalues'])

    def test_marker_permutation(self):
        samples = {name: list(reversed(row)) for name, row in d.panel().items()}
        result = d.tool(samples)
        self.assertLess(d.matrix_error(result['grm'], self.base['grm']), 1e-12)
        self.assertLess(d.matrix_error(d.projector(result['columns']), d.projector(self.base['columns'])), 2e-6)

    def test_all_allele_reversal_combinations(self):
        for flip in itertools.product((False, True), repeat=3):
            with self.subTest(flip=flip):
                samples = {name: [2-g if f else g for g, f in zip(row, flip)]
                           for name, row in d.panel().items()}
                result = d.tool(samples)
                self.assertLess(d.matrix_error(result['grm'], self.base['grm']), 1e-12)
                self.assertEqual(d.exact_grm(samples), d.exact_grm(d.panel()))

    def test_sample_permutation_equivariance(self):
        samples = dict(reversed(list(d.panel().items())))
        result = d.tool(samples)
        restored = [list(reversed(row)) for row in reversed(result['grm'])]
        self.assertLess(d.matrix_error(restored, self.base['grm']), 1e-12)
        columns = [list(reversed(col)) for col in result['columns']]
        self.assertLess(d.matrix_error(d.projector(columns), d.projector(self.base['columns'])), 2e-6)

    def test_weight_grid_with_analytic_eigenpairs(self):
        # 16 real PLINK runs, not fitted expectations or reused golden output.
        for a, b in itertools.product(range(1, 5), repeat=2):
            with self.subTest(a=a, b=b):
                samples = d.panel(a, b)
                result = d.tool(samples)
                self.assertLess(d.matrix_error(result['grm'], d.exact_grm(samples)), 1e-12)
                for obs, want in zip(result['eigenvalues'], sorted((16*a/(a+b), 16*b/(a+b)), reverse=True)):
                    self.assertAlmostEqual(obs, want, delta=5e-5)
                self.assertLess(d.matrix_error(d.projector(result['columns']), d.projector([self.a, self.b])), 2e-6)
                if a != b:
                    axis = self.a if a > b else self.b
                    self.assertAlmostEqual(abs(d.dot(result['columns'][0], axis)), 1, places=10)

    def test_nonhalf_frequencies_against_exact_matrix(self):
        # Separate unbalanced fixture exercises centering and p(1-p), not only p=1/2.
        samples = {'x': [0, 0, 1], 'y': [0, 1, 0], 'z': [1, 1, 1],
                   'w': [2, 1, 0], 'v': [0, 0, 0]}
        result = d.tool(samples)
        self.assertEqual(d.frequencies(samples), [Fraction(3, 10), Fraction(3, 10), Fraction(1, 5)])
        self.assertLess(d.matrix_error(result['grm'], d.exact_grm(samples)), 1e-12)
        for lam, vector in zip(result['eigenvalues'], result['columns']):
            residual = max(abs(d.dot(row, vector)-lam*v) for row, v in zip(result['grm'], vector))
            self.assertLess(residual, 2e-5)

    def test_eigenvector_residuals_and_orthogonality(self):
        for result in (self.base, self.copied, self.tied):
            self.assertAlmostEqual(d.dot(*result['columns']), 0, places=6)
            for lam, vec in zip(result['eigenvalues'], result['columns']):
                self.assertLess(max(abs(d.dot(row, vec)-lam*v) for row, v in zip(result['grm'], vec)), 2e-5)

    def test_vcf_and_frequency_ids_align(self):
        vcf = [line.split('\t') for line in d.vcf_text(d.panel()).splitlines() if not line.startswith('#')]
        freq = [line.split('\t') for line in d.freq_text(d.panel()).splitlines()[1:]]
        self.assertEqual([r[2] for r in vcf], [r[1] for r in freq])
        self.assertTrue(all(r[-2:] == ['0.5', '32'] for r in freq))
        self.assertTrue(all(len(r) == 25 for r in vcf))

    def test_reject_bad_pattern_counts(self):
        for value in (0, -1, True, 1.5, 101, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                d.panel(value, 1)

    def test_reject_malformed_genotypes(self):
        for value in (-1, 3, None, True, 1.0, '1'):
            samples = d.panel()
            samples['s00'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                d.validate(samples)

    def test_reject_bad_shapes_and_identifiers(self):
        cases = [{}, {'a':[0, 1], 'b':[1, 0]}, {'bad id':[0, 1], 'b':[1, 0], 'c':[2, 1]},
                 {'a':[0], 'b':[1], 'c':[2]}, {'a':[0, 0], 'b':[1, 1, 1], 'c':[2, 2]}]
        for samples in cases:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                d.validate(samples)

    def test_reject_monomorphic_markers(self):
        for g in (0, 2):
            samples = {name: [g]+row[1:] for name, row in d.panel().items()}
            with self.assertRaises(ValueError):
                d.tool(samples)

    def test_checksum_guard(self):
        with self.assertRaises(ValueError):
            d.verify(b'not the binary', d.BINARY_SHA256)


if __name__ == '__main__':
    unittest.main()
