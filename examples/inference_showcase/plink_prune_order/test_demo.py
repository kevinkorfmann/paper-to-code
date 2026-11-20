"""Exact and real-command controls for the original LD-chain fixture."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import unittest

from examples.inference_showcase.plink_prune_order import demo as d


class PruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = d.panel()
        cls.report = d.demo()

    def test_panel_counts(self):
        self.assertEqual(len(self.rows), 64)
        self.assertEqual({r: self.rows.count(r) for r in set(self.rows)},
                         {(0, 0, 0): 16, (0, 1, 2): 16, (2, 1, 0): 16, (2, 2, 2): 16})

    def test_exact_margins(self):
        self.assertEqual([sum(c) for c in zip(*self.rows)], [64]*3)

    def test_exact_r2(self):
        self.assertEqual(d.exact_r2(self.rows), [[1, Fraction(1, 2), 0],
                          [Fraction(1, 2), 1, Fraction(1, 2)], [0, Fraction(1, 2), 1]])

    def test_independent_pair_difference_oracle(self):
        # Sum over unordered sample pairs is n times centered covariance.
        cols = list(zip(*self.rows))
        for i, j in itertools.product(range(3), repeat=2):
            a, b = cols[i], cols[j]
            pairs = list(itertools.combinations(range(len(a)), 2))
            cross = sum((a[u]-a[v])*(b[u]-b[v]) for u, v in pairs)
            va = sum((a[u]-a[v])**2 for u, v in pairs)
            vb = sum((b[u]-b[v])**2 for u, v in pairs)
            self.assertEqual(Fraction(cross*cross, va*vb), d.exact_r2(self.rows)[i][j])

    def test_different_pair_scan_modes(self):
        self.assertEqual(self.report['order_1']['retained'], ['A', 'C'])
        self.assertEqual(self.report['order_2']['retained'], ['A'])

    def test_same_run_ld_report_is_unfiltered(self):
        for mode in (1, 2):
            self.assertEqual(self.report[f'order_{mode}']['r2_matrix'],
                             [[float(x) for x in row] for row in d.exact_r2(self.rows)])

    def test_exhaustive_maximum_subset(self):
        self.assertEqual(d.maximum_subsets(self.rows), [['A', 'C']])
        self.assertEqual(d.maximum_subsets(self.rows, Fraction(3, 5)), [['A', 'B', 'C']])

    def test_default_output_can_be_extended(self):
        # A and C are independent, so the one-marker result is not maximal either.
        keep = self.report['order_2']['retained']
        self.assertNotIn('C', keep)
        self.assertEqual(d.exact_r2(self.rows)[0][2], 0)

    def test_all_permutations_and_modes(self):
        expected = {('A','B','C'): (['A','C'], ['A']),
                    ('A','C','B'): (['A','C'], ['A','C']),
                    ('B','A','C'): (['B'], ['B']),
                    ('B','C','A'): (['B'], ['B']),
                    ('C','A','B'): (['C','A'], ['C','A']),
                    ('C','B','A'): (['C','A'], ['C'])}
        exact = d.exact_r2(self.rows)
        for order, answers in expected.items():
            for mode in (1, 2):
                with self.subTest(order=order, mode=mode):
                    out = d.tool(self.rows, order, mode)
                    self.assertEqual(out['retained'], answers[mode-1])
                    self.assertEqual(out['r2_matrix'],
                        [[float(exact['ABC'.index(i)]['ABC'.index(j)]) for j in order] for i in order])
                    self.assertTrue(all(exact['ABC'.index(i)]['ABC'.index(j)] <= Fraction(1, 5)
                                        for i, j in itertools.combinations(out['retained'], 2)))

    def test_strict_threshold_boundary(self):
        for threshold in (0.5, 0.6):
            for mode in (1, 2):
                with self.subTest(threshold=threshold, mode=mode):
                    self.assertEqual(d.tool(self.rows, mode=mode, threshold=threshold)['retained'], ['A','B','C'])
        for threshold in (0.1, 0.49):
            self.assertEqual(d.tool(self.rows, threshold=threshold)['retained'], ['A'])

    def test_sample_permutation(self):
        self.assertEqual(d.tool(list(reversed(self.rows))), self.report['order_2'])

    def test_replicating_count_table(self):
        self.assertEqual(d.tool(d.panel(20)), self.report['order_2'])

    def test_allele_reversal(self):
        rows = [(2-a, b, c) for a, b, c in self.rows]
        self.assertEqual(d.exact_r2(rows), d.exact_r2(self.rows))
        self.assertEqual(d.tool(rows), self.report['order_2'])

    def test_invalid_repeats(self):
        for value in (0, 12, 101, 16.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError): d.panel(value)

    def test_invalid_rows(self):
        for rows in ([], self.rows[:49], [(0, 0, 0)]*64,
                     [(0, -1, 2)]*64, [(0, True, 2)]*64, [(0, 1)]*64):
            with self.subTest(rows=rows[:1]), self.assertRaises(ValueError): d.exact_r2(rows)

    def test_invalid_order(self):
        for order in (('A','A','C'), ('A','B'), 'ABC', ('A','B','D')):
            with self.subTest(order=order), self.assertRaises(ValueError): d.vcf_text(self.rows, order)

    def test_invalid_controls(self):
        for mode in (0, 3, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError): d.tool(self.rows, mode=mode)
        for threshold in (0, 1, -1, True, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError): d.tool(self.rows, threshold=threshold)

    def test_checked_output(self):
        expected = json.loads(Path(d.__file__).with_name('sample_output.json').read_text())
        self.assertEqual(self.report, expected)


if __name__ == '__main__':
    unittest.main()
