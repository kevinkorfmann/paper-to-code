import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from . import demo


class Tests(unittest.TestCase):
    def test_fixture(self):
        a, b = demo.fixture()
        self.assertEqual(demo.merge_values(a), [.75, .75, .875])
        self.assertEqual(demo.dissimilarities(a), demo.dissimilarities(b))
        self.assertEqual(demo.features(a), demo.features(b))
        self.assertEqual(demo.readout(a, [0, 1, 2, 3]), [1.125, 1.25, 1.625, 2.0])
        self.assertEqual(demo.readout(b, [0, 1, 2, 3]), [1., 1.375, 1.75, 1.875])

    def test_threshold_connectivity_oracle(self):
        # Independent flood-fill on every threshold, without MST/union-find.
        rng = random.Random(20261120)
        for n in range(2, 9):
            for _ in range(10):
                a = [[rng.randrange(1, 10) for j in range(n)] for i in range(n)]
                p = [[x / sum(row) for x in row] for row in a]
                d = [[1 - (p[i][j] + p[j][i]) / 2 for j in range(n)] for i in range(n)]
                thresholds = sorted(set(d[i][j] for i in range(n) for j in range(i)))
                expected, previous = [], n
                for t in thresholds:
                    unseen = set(range(n))
                    components = 0
                    while unseen:
                        frontier = [unseen.pop()]
                        components += 1
                        while frontier:
                            i = frontier.pop()
                            found = {j for j in unseen if d[i][j] <= t}
                            unseen -= found
                            frontier.extend(found)
                    expected.extend([t] * (previous - components))
                    previous = components
                self.assertEqual(demo.merge_values(a), expected)

    def test_features(self):
        f = demo.features(demo.fixture()[0])
        self.assertEqual(f['mean'], 19 / 24)
        self.assertAlmostEqual(f['std'], math.sqrt(1 / 288))
        for key, value in [('q10', .75), ('q25', .75), ('q50', .75),
                           ('q75', .8125), ('q90', .85), ('maximum', .875), ('total', 2.375)]:
            self.assertAlmostEqual(f[key], value)
        self.assertAlmostEqual(f['entropy'], -2 * (6/19)*math.log(6/19) - (7/19)*math.log(7/19))

    def test_uniform_and_length(self):
        for n in range(2, 20):
            f = demo.features([[1] * n for _ in range(n)])
            self.assertAlmostEqual(f['mean'], 1 - 1/n)
            self.assertAlmostEqual(f['total'], (n-1)**2/n)
            self.assertAlmostEqual(f['entropy'], math.log(n-1))
            self.assertAlmostEqual(1 + f['total']/f['mean'], n)

    def test_zero_death(self):
        f = demo.features([[0, 1], [1, 0]])
        self.assertTrue(all(v == 0 for v in f.values()))

    def test_identity_ties(self):
        self.assertEqual(demo.merge_values([[1,0,0],[0,1,0],[0,0,1]]), [1,1])

    def test_row_rescale(self):
        a, _ = demo.fixture()
        self.assertEqual(demo.features(a), demo.features([[x * (2**i) for x in row] for i, row in enumerate(a)]))

    def test_permutation(self):
        a, _ = demo.fixture()
        order = [3, 1, 0, 2]
        b = [[a[i][j] for j in order] for i in order]
        self.assertEqual(demo.features(a), demo.features(b))
        self.assertEqual(demo.readout(b, order), [demo.readout(a, [0,1,2,3])[i] for i in order])

    def test_invalid_matrix(self):
        for a in [[], [[1]], [[1,2],[3]], [[0,0],[1,1]], [[-1,2],[1,1]],
                  [[float('nan'),1],[1,1]], [[float('inf'),1],[1,1]]]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                demo.features(a)

    def test_invalid_values(self):
        a, _ = demo.fixture()
        for values in [[], [0,1,2,float('nan')], [0,1,2,float('inf')]]:
            with self.assertRaises(ValueError):
                demo.readout(a, values)

    def test_cli_sample(self):
        output = subprocess.check_output([sys.executable, '-m', 'examples.attention_topology.demo'])
        self.assertEqual(output, Path(__file__).with_name('sample_output.json').read_bytes())
        self.assertTrue(json.loads(output)['identical_features'])


if __name__ == '__main__':
    unittest.main()
