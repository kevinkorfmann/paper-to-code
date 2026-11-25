"""Scalar SAGA key-head merging: match the logits, not just the keys.

Elata et al., "More Value per Key" (preprint, 3 October 2026), Sec. 4.1:
https://arxiv.org/html/2610.04753v1#S4.SS1

For one scalar calibration token equal to 1, a head with key k and query q
contributes (q * (merged_key - k))**2 to the logit reconstruction loss.
The scalar version of Eq. 15 uses q**2 as a weight. For four teacher heads,
there are exactly three perfect pairings. This toy enumerates them instead
of using a scalable matching algorithm. Value heads are not merged.
There is no RoPE, sparsity, model training, or timing benchmark.

CPython 3.14.5, standard library only, deterministic (no seed).
Run: python examples/saga_scalar_merge/demo.py
Test: python examples/saga_scalar_merge/demo.py --test

Expected stdout:
key-distance pairing: AB / CD, logit SSE 50.500000
logit-loss pairing: AC / BD, logit SSE 7.920792
same logit-loss pairing with plain averages: logit SSE 202.000000
"""
import argparse
from fractions import Fraction
from itertools import combinations
import unittest


HEADS = ((0, 10), (1, 10), (2, 1), (3, 1))  # (key, calibration query)
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def merged_key(a, b):
    ka, qa = a
    kb, qb = b
    wa, wb = qa**2, qb**2
    # Both zero: all keys give the same logits; choose pseudoinverse zero.
    return Fraction(wa * ka + wb * kb, wa + wb) if wa + wb else Fraction(0)


def logit_error(a, b, candidate):
    return sum((q * candidate - q * k)**2 for k, q in (a, b))


def pairing_loss(heads, pairing, *, weighted=True):
    total = Fraction(0)
    for i, j in pairing:
        a, b = heads[i], heads[j]
        key = merged_key(a, b) if weighted else Fraction(a[0] + b[0], 2)
        total += logit_error(a, b, key)
    return total


def key_distance(heads, pairing):
    return sum((heads[i][0] - heads[j][0])**2 for i, j in pairing)


def render():
    by_key = min(PAIRINGS, key=lambda p: key_distance(HEADS, p))
    by_logit = min(PAIRINGS, key=lambda p: pairing_loss(HEADS, p))
    label = lambda p: " / ".join("ABCD"[i] + "ABCD"[j] for i, j in p)
    return (
        f"key-distance pairing: {label(by_key)}, logit SSE {float(pairing_loss(HEADS, by_key)):.6f}\n"
        f"logit-loss pairing: {label(by_logit)}, logit SSE {float(pairing_loss(HEADS, by_logit)):.6f}\n"
        "same logit-loss pairing with plain averages: "
        f"logit SSE {float(pairing_loss(HEADS, by_logit, weighted=False)):.6f}"
    )


class TestScalarMerge(unittest.TestCase):
    def test_equal_query_magnitude_recovers_average(self):
        self.assertEqual(merged_key((0, -3), (4, 3)), 2)

    def test_no_queries_or_one_unseen_head(self):
        self.assertEqual(merged_key((7, 0), (9, 0)), 0)
        self.assertEqual(merged_key((7, 0), (9, -2)), 9)

    def test_fractional_merge_and_cost(self):
        self.assertEqual(merged_key(HEADS[0], HEADS[2]), Fraction(2, 101))
        self.assertEqual(pairing_loss(HEADS, PAIRINGS[1]), Fraction(800, 101))

    def test_minimum_against_perturbed_candidates(self):
        for a, b in combinations(HEADS, 2):
            key = merged_key(a, b)
            expected = logit_error(a, b, key)
            for n in range(-12, 13):
                delta = Fraction(n, 13)
                # Independent completion-of-square check.
                self.assertEqual(logit_error(a, b, key + delta) - expected,
                                 (a[1]**2 + b[1]**2) * delta**2)

    def test_pairing_objectives_choose_different_heads(self):
        self.assertEqual(min(PAIRINGS, key=lambda p: key_distance(HEADS, p)),
                         PAIRINGS[0])
        self.assertEqual(min(PAIRINGS, key=lambda p: pairing_loss(HEADS, p)),
                         PAIRINGS[1])

    def test_output_matches_documented_replay(self):
        self.assertEqual(render(), __doc__.split("Expected stdout:\n")[1].strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true")
    if parser.parse_args().test:
        unittest.main(argv=["demo.py"], verbosity=2)
    else:
        print(render())
