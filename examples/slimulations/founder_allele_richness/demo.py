"""Run and score a single neutral founding generation (SLiM 4.0.1)."""
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

P = (0.70, 0.20) + (0.01,) * 10
SEED = 20261126
REPS = 2048
M = 10


def theory(p=P, m=M):
    h = 1 - sum(x*x for x in p)
    return sum(1 - (1-x)**m for x in p), h * (1 - 1/m)


def run(reps=REPS):
    model = Path(__file__).with_name("model.slim")
    out = subprocess.check_output(
        ["slim", "-s", str(SEED), "-d", f"REPS={reps}", str(model)], text=True
    )
    rows = [tuple(map(int, row.split()[1:]))
            for row in out.splitlines() if row.startswith("WIN ")]
    if [r[0] for r in rows] != list(range(2, reps + 2)):
        raise ValueError("Missing, repeated, or out-of-order cohort")
    if any(not 1 <= k <= M or not M <= sq <= M*M for _, k, sq in rows):
        raise ValueError("Invalid founder summary")
    return [k for _, k, _ in rows], [1 - sq/(M*M) for _, _, sq in rows]


def output(observed):
    expected = theory()
    source = len(P), 1 - sum(x*x for x in P)
    return "\n".join(
        f"{name}: source {start:.5f}, founder mean {statistics.mean(values):.5f}, "
        f"exact {target:.5f}, MC SE {statistics.stdev(values)/len(values)**0.5:.5f}"
        for name, start, values, target in zip(("allele count", "gene diversity"),
                                              source, observed, expected)
    )


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run()

    def test_source(self):
        self.assertAlmostEqual(sum(P), 1)
        self.assertAlmostEqual(1 - sum(x*x for x in P), .469)

    def test_one_draw_or_one_state(self):
        for p, m in ((P, 1), ((1.0,), 10)):
            k, h = theory(p, m)
            self.assertAlmostEqual(k, 1)
            self.assertAlmostEqual(h, 0)

    def test_exact_small_multinomial(self):
        # Two draws from (3/4, 1/4): different with probability 3/8.
        k, h = theory((.75, .25), 2)
        self.assertAlmostEqual(k, 1 + 3/8)
        self.assertAlmostEqual(h, 3/16)

    def test_cohorts_and_bounds(self):
        k, h = self.results
        self.assertEqual(len(k), REPS)
        self.assertTrue(all(1 <= x <= M for x in k))
        self.assertTrue(all(0 <= x <= .9 for x in h))
        self.assertTrue(all((x == 1) == (y == 0) for x, y in zip(k, h)))

    def test_matches_expected(self):
        for values, target in zip(self.results, theory()):
            se = statistics.stdev(values)/len(values)**0.5
            self.assertLess(abs(statistics.mean(values) - target), 4*se)

    def test_checked_output(self):
        expected = Path(__file__).with_name("expected.txt").read_text().rstrip()
        self.assertEqual(output(self.results), expected)


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        unittest.main(argv=[sys.argv[0]])
    elif not sys.argv[1:]:
        print(output(run()))
    else:
        sys.exit("usage: python demo.py [--test]")
