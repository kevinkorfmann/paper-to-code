import unittest

import numpy as np

from example import estimate, make_example


class MeanDescendantsTests(unittest.TestCase):
    def setUp(self):
        self.ts, self.focal, self.sets = make_example()

    def test_local_descendants(self):
        self.assertEqual([set(t.samples(self.focal)) for t in self.ts.trees()],
                         [{0, 1}, {2}, set()])

    def test_conditional_counts(self):
        counts, span, _ = estimate(self.ts, self.focal, self.sets)
        self.assertEqual(span, 50)
        np.testing.assert_allclose(counts, [40 / 50, 30 / 50])

    def test_full_span_counts(self):
        _, _, counts = estimate(self.ts, self.focal, self.sets)
        np.testing.assert_allclose(counts, [40 / 100, 30 / 100])

    def test_reordered_groups(self):
        counts, span, _ = estimate(self.ts, self.focal, self.sets[::-1])
        self.assertEqual(span, 50)
        np.testing.assert_allclose(counts, [30 / 50, 40 / 50])

    def test_no_mutations(self):
        self.assertEqual(self.ts.num_sites, 0)
        counts, _, _ = estimate(self.ts, self.focal, self.sets)
        self.assertGreater(counts.sum(), 0)


if __name__ == "__main__":
    unittest.main()
