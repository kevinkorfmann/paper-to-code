import unittest

import numpy as np

from demo import GROUP, TRAIT, WITHIN, build_ts, statistic


class LinearModelTests(unittest.TestCase):
    def setUp(self):
        self.ts = build_ts()

    def test_known_genotypes(self):
        np.testing.assert_array_equal(self.ts.genotype_matrix(), [GROUP, WITHIN])

    def test_unadjusted_effect_sizes(self):
        np.testing.assert_allclose(statistic(self.ts), [100, 4], atol=1e-9)

    def test_group_tag_and_estimable_marker(self):
        g = self.ts.genotype_matrix()
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([np.ones(8), GROUP, g[0]])), 2)
        design = np.column_stack([np.ones(8), GROUP, g[1]])
        self.assertEqual(np.linalg.matrix_rank(design), 3)
        beta = np.linalg.lstsq(design, TRAIT, rcond=None)[0][-1]
        np.testing.assert_allclose(statistic(self.ts, GROUP[:, None]),
                                   [0, beta**2], atol=1e-9)

    def test_effect_sign_is_not_encoded(self):
        np.testing.assert_allclose(statistic(self.ts, GROUP[:, None], trait=-TRAIT),
                                   [0, 4], atol=1e-9)

    def test_span_normalization(self):
        np.testing.assert_allclose(statistic(self.ts, GROUP[:, None], span_normalise=True),
                                   [0, 0.4], atol=1e-9)
