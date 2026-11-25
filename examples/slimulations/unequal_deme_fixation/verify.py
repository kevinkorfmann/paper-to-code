"""Run seeded SLiM conditions; check totals, neutral theory, and replay."""
import argparse
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
SEEDS = (20261125, 20261126)
REPS = 384
EXPECTED = ((188, 196, 0), (55, 329, 0))


def migrate(state, rate):
    small, large = state
    return ((1 - rate) * small + rate * large,
            rate * small + (1 - rate) * large)


def run(slim, origin, reps=REPS):
    result = subprocess.run(
        [slim, "-s", str(SEEDS[origin]), "-d", f"ORIGIN={origin}",
         "-d", f"REPS={reps}", str(HERE / "model.slim")],
        check=True, text=True, capture_output=True, timeout=60,
    )
    row, = (line.split() for line in result.stdout.splitlines()
            if line.startswith("result "))
    _, returned_origin, fixed, lost, ongoing = row
    counts = tuple(map(int, (fixed, lost, ongoing)))
    if int(returned_origin) != origin or min(counts) < 0 or sum(counts) != reps:
        raise ValueError(f"Invalid output: {row}")
    return counts


class Checks(unittest.TestCase):
    def test_neutral_frequency_is_martingale_in_deme_average(self):
        for a in range(11):
            for b in range(41):
                before = (Fraction(a, 10), Fraction(b, 40))
                self.assertEqual(sum(migrate(before, Fraction(1, 5))), sum(before))

    def test_starting_pooled_frequency_matches(self):
        states = ((1, 0), (0, Fraction(1, 4)))
        self.assertEqual([Fraction(a + 4*b, 5) for a, b in states],
                         [Fraction(1, 5)] * 2)
        self.assertEqual([Fraction(a+b, 2) for a, b in states],
                         [Fraction(1, 2), Fraction(1, 8)])

    def test_pooled_frequency_is_not_martingale(self):
        a, b = migrate((1, 0), Fraction(1, 5))
        self.assertEqual((a + 4*b) / 5, Fraction(8, 25))

    def test_replay(self):
        self.assertEqual(OBSERVED, EXPECTED)

    def test_absorption_counts_match_binomial_expectations(self):
        for (fixed, _, ongoing), p in zip(OBSERVED, (0.5, 0.125), strict=True):
            self.assertEqual(ongoing, 0)
            se = math.sqrt(p * (1 - p) / REPS)
            self.assertLess(abs(fixed / REPS - p), 4 * se)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    version = subprocess.run([args.slim, "-v"], capture_output=True, text=True,
                             check=True).stdout.splitlines()[0].split()[2].rstrip(",")
    if version != "4.0.1":
        raise SystemExit(f"Expected SLiM 4.0.1 for replay; found {version}")
    print(f"SLiM {version}", flush=True)
    OBSERVED = tuple(run(args.slim, i) for i in range(2))
    for i, ((fixed, lost, ongoing), p) in enumerate(zip(OBSERVED, (0.5, 0.125), strict=True)):
        print(f"origin={i} seed={SEEDS[i]} fixed={fixed} lost={lost} "
              f"ongoing={ongoing} fixation={fixed/REPS:.6f} theory={p:.6f}",
              flush=True)
    if args.test:
        unittest.main(argv=["verify.py"], verbosity=2)
