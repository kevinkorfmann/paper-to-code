import cmath
from itertools import product
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from . import demo


class HiSATest(unittest.TestCase):
    def test_frequencies(self):
        self.assertEqual(demo.frequencies(4, 16), (1.0, 0.25))
        self.assertEqual(demo.frequencies(2, 16), (1.0,))

    def test_strict_budget(self):
        self.assertEqual(demo.active_pairs(2, (1, .25), 2), (1,))
        self.assertEqual(demo.active_pairs(2, (1, .25), math.nextafter(2, math.inf)), (0, 1))
        self.assertEqual(demo.active_pairs(8, (1, .25), 2), ())

    def test_unwrapped(self):
        q, i, _, base, budget = demo.fixture()
        self.assertLess(abs(math.remainder(6, math.tau)), budget)
        self.assertEqual(demo.active_pairs(6, demo.frequencies(len(q), base), budget), (1,))

    def test_fixture_logits_and_no_pair_mask(self):
        out = demo.report()
        self.assertEqual([r["partial_logit"] for r in out["keys"]], [1., 0., 0., None])
        self.assertEqual([r["active_pair_indices"] for r in out["keys"]],
                         [[0, 1], [1], [1], []])

    def test_fixture_readout(self):
        q, i, keys, base, budget = demo.fixture()
        self.assertAlmostEqual(demo.attention(q, i, keys, base, budget), 6 / (math.e + 2))
        self.assertNotAlmostEqual(demo.attention(q, i, keys, base, budget), 106 / (math.e + 3))
        self.assertNotAlmostEqual(demo.attention(q, i, keys, base, budget), 0)

    def test_future_and_expired_keys_leave_readout_unchanged(self):
        q, i, keys, base, budget = demo.fixture()
        self.assertEqual(demo.attention(q, i, keys[:-1], base, budget),
                         demo.attention(q, i, keys + ((i+1, keys[0][1], 1e20),), base, budget))

    def test_pair_work_closed_form(self):
        self.assertEqual(demo.pair_work(8, (1., .25), math.pi / 2), (50, 72))
        self.assertEqual(demo.pair_work(8, (1., .25), math.inf), (72, 72))

    def test_pair_work_oracle(self):
        for n, d, base, budget in product(range(1, 11), (2, 4, 8), (4, 16), (0.5, 2, math.inf)):
            omega = demo.frequencies(d, base)
            counts = [0, 0]
            for i in range(n):
                for j in range(i + 1):
                    for rate in omega:
                        counts[0] += i < j + budget / rate
                        counts[1] += 1
            self.assertEqual(demo.pair_work(n, omega, budget), tuple(counts))

    def test_complex_absolute_position_oracle(self):
        # Independent implementation: rotate each vector at its absolute
        # position as complex numbers before taking the real inner product.
        rng = random.Random(20261122)
        for d, base, budget, distance in product((2,4,8), (4,16,256), (0.5, 2, math.inf), range(10)):
            q = [rng.uniform(-3,3) for _ in range(d)]
            k = [rng.uniform(-3,3) for _ in range(d)]
            omega = tuple(1 / base**(a/d) for a in range(0,d,2))
            j = rng.randrange(50)
            i = j + distance
            terms = []
            for idx, rate in enumerate(omega):
                if distance < budget / rate:
                    qr = complex(*q[2*idx:2*idx+2]) * cmath.exp(1j*i*rate)
                    kr = complex(*k[2*idx:2*idx+2]) * cmath.exp(1j*j*rate)
                    terms.append((qr * kr.conjugate()).real)
            actual = demo.score(q, k, distance, demo.frequencies(d, base), budget)
            if terms:
                self.assertAlmostEqual(actual, sum(terms)/math.sqrt(d), places=11)
            else:
                self.assertIsNone(actual)

    def test_invalid_controls(self):
        with self.assertRaises(ValueError):
            demo.frequencies(3, 16)
        with self.assertRaises(ValueError):
            demo.active_pairs(1, (1.0,), float("nan"))
        with self.assertRaises(ValueError):
            demo.active_pairs(-1, (1.0,), 2)
        with self.assertRaises(ValueError):
            demo.score((1, 2), (1,), 0, (1.0,), math.inf)
        with self.assertRaises(ValueError):
            demo.attention((1, 2), 1, (), 16, 2)
        with self.assertRaises(ValueError):
            demo.pair_work(0, (1.0,), 2)

    def test_checked_output_replay(self):
        out = subprocess.check_output([sys.executable, "-m", "examples.hisa_rope_pairs.demo"])
        self.assertEqual(out, Path(__file__).with_name("sample_output.json").read_bytes())
