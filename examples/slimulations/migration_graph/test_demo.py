import os
import shutil
import unittest

from .demo import prediction, run


class MigrationGraphTests(unittest.TestCase):
    def test_exact_expected(self):
        exact = {
            "ring": [[1, 0, 0, 0], [.7, .15, 0, .15], [.535, .21, .045, .21]],
            "complete": [[1, 0, 0, 0], [.7, .1, .1, .1], [.52, .16, .16, .16]],
        }
        for graph, rows in exact.items():
            for obs, exp in zip(prediction(graph), rows, strict=True):
                for a, b in zip(obs, exp, strict=True):
                    self.assertAlmostEqual(a, b)
            for row in prediction(graph, 30):
                self.assertAlmostEqual(sum(row) / len(row), .25)

    def test_slim(self):
        executable = os.environ.get("SLIM", "slim")
        if shutil.which(executable) is None:
            self.skipTest("SLiM executable not on PATH; set SLIM")
        ring, complete = [run(graph, executable) for graph in ("ring", "complete")]
        self.assertEqual(ring[1]["observed"][2], 0)
        self.assertGreater(complete[1]["observed"][2], .08)
        self.assertGreater(ring[2]["observed"][2], .03)


if __name__ == "__main__":
    unittest.main()
