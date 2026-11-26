"""Branch f4 weights quartet spans by internal branch length.

Run: python demo.py
Tests: python demo.py --test
CPython 3.12.11, tskit 1.0.4, NumPy 2.5.5; no random input.
Reference: Ralph et al. (2020), https://doi.org/10.1534/genetics.120.303253
"""
import sys
import unittest

import numpy as np
import tskit

SETS = [[0], [1], [2], [3]]
ORDER = (0, 1, 2, 3)


def fixture(minority_internal=10, left_split=((0, 2), (1, 3)),
            right_split=((0, 3), (1, 2))):
    """Two rooted ultrametric quartets; left spans 70 of 100 bp."""
    tables = tskit.TableCollection(sequence_length=100)
    tables.time_units = "generations"
    for _ in SETS:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for left, right, split, length in [
        (0, 70, left_split, 2),
        (70, 100, right_split, minority_internal),
    ]:
        root = tables.nodes.add_row(time=1 + length / 2)
        for group in split:
            node = tables.nodes.add_row(time=1)
            tables.edges.add_row(left, right, root, node)
            for leaf in group:
                tables.edges.add_row(left, right, node, leaf)
    tables.sort()
    return tables.tree_sequence()


def f4(ts, order=ORDER, windows=None, mode="branch"):
    return ts.f4(SETS, indexes=order, windows=windows, mode=mode)


def paths_oracle(ts, order=ORDER):
    """Independent tree path identity for f4 on dated singleton quartets."""
    a, b, c, d = order
    return sum(
        tr.span * (tr.distance_between(a, d) + tr.distance_between(b, c)
                   - tr.distance_between(a, c) - tr.distance_between(b, d)) / 2
        for tr in ts.trees()
    ) / ts.sequence_length


def demo():
    ts = fixture()
    by_window = f4(ts, windows=[0, 70, 100])
    print(f"AC|BD 70 bp: {by_window[0]:+.2f} generations")
    print(f"AD|BC 30 bp: {by_window[1]:+.2f} generations")
    print(f"whole sequence: {f4(ts):+.2f} generations")
    print(f"same topology, equal internal lengths: {f4(fixture(2)):+.2f} generations")
    print(f"no mutation rows, site f4: {f4(ts, mode='site'):.2f}")


class Checks(unittest.TestCase):
    def test_windows(self):
        np.testing.assert_allclose(f4(fixture(), windows=[0, 70, 100]), [2, -10])

    def test_span_weighting_changes_sign(self):
        self.assertAlmostEqual(f4(fixture()), -1.6)
        self.assertAlmostEqual(f4(fixture(2)), 0.8)

    def test_path_oracle(self):
        for length in (1, 2, 8, 10, 20):
            for order in (ORDER, (1, 0, 2, 3), (0, 2, 1, 3)):
                with self.subTest(length=length, order=order):
                    ts = fixture(length)
                    self.assertAlmostEqual(f4(ts, order), paths_oracle(ts, order))

    def test_same_splits_after_changing_lengths(self):
        def clades(ts):
            return [tuple(sorted(tuple(sorted(tree.samples(u))) for u in
                                 tree.nodes() if tree.num_children(u) == 2))
                    for tree in ts.trees()]
        self.assertEqual(clades(fixture()), clades(fixture(2)))

    def test_matching_split_is_zero(self):
        split = ((0, 1), (2, 3))
        self.assertEqual(f4(fixture(left_split=split, right_split=split)), 0)

    def test_site_has_no_mutations(self):
        self.assertEqual(f4(fixture(), mode="site"), 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        unittest.main(argv=[sys.argv[0]])
    elif not sys.argv[1:]:
        demo()
    else:
        sys.exit("usage: python demo.py [--test]")
