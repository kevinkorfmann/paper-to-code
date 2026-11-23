import itertools
import math
import random
import unittest
from . import demo as d


class Tests(unittest.TestCase):
    def test_levels_and_saturation(self):
        self.assertEqual(d.quantize(0), 0)
        self.assertEqual(d.quantize(-100), -8)
        self.assertEqual(d.quantize(100), 7)
        for u in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                d.quantize(u)

    def test_analytic_inverse_grid(self):
        # Invert the bound and put each signed level inside its rounding bin.
        for x in range(-8, 8):
            y = min(7.0, max(-7.99, x))
            u = math.atanh((y + .5) / 7.5075) - math.atanh(.5 / 7.5075)
            self.assertEqual(d.quantize(u), x)

    def test_every_byte(self):
        for x, y in itertools.product(range(-8, 8), repeat=2):
            b = d.pack((x, y))
            self.assertEqual(b, bytes([x + 8 + 16*(y + 8)]))
            self.assertEqual(d.unpack(b, 2), (x, y))

    def test_odd_padding(self):
        self.assertEqual(d.pack((-8, 7, 0)), bytes.fromhex("f008"))
        self.assertEqual(d.unpack(bytes.fromhex("f008"), 3), (-8, 7, 0))

    def test_invalid_pack(self):
        for x in ((), (8,), (-9,), (True,), (0.0,)):
            with self.assertRaises(ValueError):
                d.pack(x)

    def test_invalid_payload(self):
        for b, n in ((b"", 2), (b"\x18", 1), (b"\x08", 3), (b"", 0)):
            with self.assertRaises(ValueError):
                d.unpack(b, n)

    def test_pool_identity_randomized(self):
        rng = random.Random(142)
        for width in (1, 2, 3, 8, 9):
            for n in (1, 2, 7, 40):
                z = [tuple(rng.randrange(-8, 8) for _ in range(width)) for _ in range(n)]
                k = [[rng.uniform(-1, 1) for _ in range(width)] for _ in range(3)]
                v = [[rng.uniform(-1, 1) for _ in range(width)] for _ in range(5)]
                q = [.4, -.7, .2]
                a = d.read(q, z, k, v, pool_first=False)
                b = d.read_packed(q, [d.pack(x) for x in z], width, k, v)
                for x, y in zip(a, b, strict=True):
                    self.assertAlmostEqual(x, y, places=13)

    def test_closed_form_two_tokens(self):
        # With scores 0, ln(3), weights are 1/4, 3/4.
        out = d.read_packed([math.log(3)], [d.pack((0,)), d.pack((1,))], 1, [[1]], [[4], [-8]])
        self.assertAlmostEqual(out[0], 3)
        self.assertAlmostEqual(out[1], -6)

    def test_stable_extreme_scores(self):
        out = d.read_packed([10000], [d.pack((-8,)), d.pack((7,))], 1, [[1]], [[1]])
        self.assertEqual(out, (7,))

    def test_visible_prefix(self):
        a = [d.pack((0,)), d.pack((1,))]
        self.assertEqual(d.read_packed([0], a[:1], 1, [[1]], [[1]]), (0,))
        self.assertEqual(d.read_packed([0], a, 1, [[1]], [[1]]), (.5,))

    def test_demo_distinguishes_errors(self):
        out = d.demo()
        self.assertEqual(out["decode_then_pool"], out["pool_then_decode"])
        self.assertGreater(out["quantization_max_abs_error"], 0)
        self.assertEqual(out["packed_payload_bytes"], 6)
        self.assertEqual(out["illustrative_fp16_kv_payload_bytes"], 24)


if __name__ == "__main__":
    unittest.main()
