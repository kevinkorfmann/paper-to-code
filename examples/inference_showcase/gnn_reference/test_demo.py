"""Independent exact tree-path oracle and deterministic controls."""
from fractions import Fraction
import itertools
import json
import math
import subprocess
import sys
import unittest

import numpy as np
import tskit

from examples.inference_showcase.gnn_reference import demo


def path_oracle(parent, focal, groups):
    """Pure-Python parent-chain intersections, no tskit traversal/statistic."""
    def ancestors(node):
        out = [node]
        while node in parent:
            node = parent[node]
            out.append(node)
        return out
    focal_path = ancestors(focal)
    refs = [(node, group) for group, nodes in enumerate(groups) for node in nodes if node != focal]
    mrca_depth = []
    for node, group in refs:
        ref_path = set(ancestors(node))
        common = [i for i, ancestor in enumerate(focal_path) if ancestor in ref_path]
        if common:
            mrca_depth.append((min(common), group))
    closest = min(depth for depth, _ in mrca_depth)
    counts = [sum(depth == closest and group == k for depth, group in mrca_depth) for k in range(2)]
    return [Fraction(count, sum(counts)) for count in counts]


PARENT1 = {**{node: 6 for node in range(5)}, 6: 7, 5: 7}
PARENT2 = {0: 8, 4: 8, 8: 6, 1: 6, 2: 6, 3: 6, 6: 7, 5: 7}


class TestGNN(unittest.TestCase):
    def setUp(self):
        self.ts = demo.build_tree_sequence()
        self.groups = [[1, 2, 3], [4, 5]]

    def test_main_result(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, [[1], [4, 5]]), [0.5, 0.5])
        np.testing.assert_array_equal(demo.gnn(self.ts, self.groups), [0.75, 0.25])

    def test_unselected_leaves_stay_in_tree(self):
        self.assertEqual(self.ts.num_samples, 6)
        self.assertEqual(self.ts.num_trees, 1)
        self.assertEqual(self.ts.num_sites, 0)
        self.assertEqual(dict((e.child, e.parent) for e in self.ts.edges()), PARENT1)

    def test_distant_B_does_not_dilute_nearest_clade(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, [[1, 2, 3], [4]]), demo.gnn(self.ts, self.groups))

    def test_removing_nearest_B_changes_target(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, [[1, 2, 3], [5]]), [1, 0])

    def test_focal_excludes_itself(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, [[0, 1, 2, 3], [4, 5]]), [0.75, 0.25])

    def test_reference_order(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, [[3, 1, 2], [5, 4]]), [0.75, 0.25])

    def test_group_swap(self):
        np.testing.assert_array_equal(demo.gnn(self.ts, self.groups[::-1]), [0.25, 0.75])

    def test_time_scale_invariance(self):
        for scale in (0.01, 2, 1000):
            np.testing.assert_array_equal(demo.gnn(demo.build_tree_sequence(time_scale=scale), self.groups), [0.75, 0.25])

    def test_coordinate_scale_invariance(self):
        for length in (0.125, 1, 2000):
            np.testing.assert_array_equal(demo.gnn(demo.build_tree_sequence(length=length), self.groups), [0.75, 0.25])

    def test_span_weighting(self):
        for split in (1, 25, 50, 75, 99):
            ts = demo.build_tree_sequence(split=split)
            a = Fraction(split, 100) * Fraction(3, 4)
            np.testing.assert_allclose(demo.gnn(ts, self.groups), [float(a), float(1-a)], atol=1e-15, rtol=0)

    def test_all_reference_assignments_exact_oracle(self):
        # Each nonfocal sample is omitted or assigned to A or B; both groups nonempty.
        count = 0
        ts2 = demo.build_tree_sequence(split=75)
        for labels in itertools.product((-1, 0, 1), repeat=5):
            groups = [[node for node, label in enumerate(labels, 1) if label == k] for k in range(2)]
            if not all(groups):
                continue
            count += 1
            q1 = path_oracle(PARENT1, 0, groups)
            q2 = path_oracle(PARENT2, 0, groups)
            np.testing.assert_allclose(demo.gnn(self.ts, groups), list(map(float, q1)), atol=1e-15, rtol=0)
            exact = [Fraction(3,4)*a+Fraction(1,4)*b for a,b in zip(q1,q2)]
            np.testing.assert_allclose(demo.gnn(ts2, groups), list(map(float, exact)), atol=1e-15, rtol=0)
        self.assertEqual(count, 180)

    def test_other_focal_nodes(self):
        for focal in range(6):
            expected = path_oracle(PARENT1, focal, self.groups)
            np.testing.assert_allclose(demo.gnn(self.ts, self.groups, focal), list(map(float, expected)), atol=1e-15, rtol=0)

    def test_invalid_groups(self):
        for groups in ([], [[1]], [[1], []], [[1], [1]], [[1, 1], [4]], [[1.0], [4]], [[True], [4]], [[6], [4]], [[-1], [4]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                demo.gnn(self.ts, groups)

    def test_invalid_focal(self):
        for focal in (-1, 6, 0.0, True):
            with self.assertRaises(ValueError):
                demo.gnn(self.ts, self.groups, focal)

    def test_invalid_geometry(self):
        for x in (0, -1, math.nan, math.inf):
            for key in ('length', 'time_scale'):
                with self.assertRaises(ValueError):
                    demo.build_tree_sequence(**{key:x})
        for split in (0, 100, 101, math.nan, math.inf):
            with self.assertRaises(ValueError):
                demo.build_tree_sequence(split=split)

    def test_disconnected_focal_rejected(self):
        tables = self.ts.dump_tables()
        edges = [(e.left,e.right,e.parent,e.child) for e in tables.edges if e.child != 0]
        tables.edges.clear()
        for edge in edges:
            tables.edges.add_row(*edge)
        tables.sort()
        with self.assertRaises(ValueError):
            demo.gnn(tables.tree_sequence(), self.groups)

    def test_no_mutations_needed(self):
        tables = self.ts.dump_tables()
        site = tables.sites.add_row(position=50, ancestral_state='0')
        tables.mutations.add_row(site=site, node=1, derived_state='1')
        np.testing.assert_array_equal(demo.gnn(tables.tree_sequence(), self.groups), demo.gnn(self.ts,self.groups))

    def test_sample_json(self):
        self.assertEqual(demo.report(), json.loads((demo.HERE/'sample_output.json').read_text()))

    def test_cli(self):
        proc = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.gnn_reference.demo', '--check-sample'], capture_output=True, text=True, check=True)
        self.assertEqual(proc.stdout, 'Checked sample matches\n')

    def test_versions(self):
        self.assertEqual(demo.report()['versions'], {'python':'3.12.11','tskit':'1.0.3','numpy':'2.5.4'})


if __name__ == '__main__':
    unittest.main()
