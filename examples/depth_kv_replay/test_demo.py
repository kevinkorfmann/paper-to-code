import math
import random
import unittest
from . import demo as d


class Tests(unittest.TestCase):
    def test_provider_skips_readers_and_keeps_layers_separate(self):
        calls = []
        def writer(r, layer):
            calls.append((r, layer))
            if r % 2:
                self.fail("reader tried to project or write KV")
            return [(0, 10*r + layer), (999, 999)]
        rows, slots = d.replay([[0, 0]]*5, writer, 2, 1)
        self.assertEqual(calls, [(r, l) for r in (0, 2, 4) for l in (0, 1)])
        self.assertEqual(slots, 6)
        self.assertEqual([x["outputs"] for x in rows], [[0, 1]]*2+[[20, 21]]*2+[[40, 41]])
        self.assertEqual([x["writer"] for x in rows], [0, 0, 2, 2, 4])

    def test_queries_are_fresh_and_keys_are_not(self):
        rows, _ = d.replay([[1], [-1]], lambda r, l: [(-1, 0), (1, 1)], 2, 2)
        for row, q in zip(rows, (1, -1), strict=True):
            self.assertAlmostEqual(row["outputs"][0], 1/(1+math.exp(-2*q)))

    def test_routing_and_readout_vs_direct_oracle(self):
        rng = random.Random(24)
        for n in (1, 2, 7, 9):
            q = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(n)]
            kv = [[[tuple(rng.uniform(-2, 2) for _ in range(2)) for _ in range(5)]
                   for _ in range(3)] for _ in range(n)]
            for stride in (1, 2, 4, 12):
                rows, slots = d.replay(q, lambda r, l: kv[r][l], stride, 4)
                self.assertEqual(slots, 3*((n+stride-1)//stride))
                for r, row in enumerate(rows):
                    for layer, actual in enumerate(row["outputs"]):
                        history = kv[r-r%stride][layer][:4]
                        m = [math.exp(q[r][layer]*k) for k, _ in history]
                        expected = sum(w*v for w, (_, v) in zip(m, history))/sum(m)
                        self.assertAlmostEqual(actual, expected, places=13)

    def test_causal_suffix_not_read(self):
        a = d.scalar_attention(8, [(1, 2), (1, 4)], 2)
        b = d.scalar_attention(8, [(1, 2), (1, 4), (math.inf, math.nan)], 2)
        self.assertEqual(a, b)
        self.assertEqual(a, 3)

    def test_stable_large_logits(self):
        self.assertEqual(d.scalar_attention(10000, [(-5, 7), (5, 8)], 2), 8)

    def test_bad_shapes_and_controls(self):
        project = lambda *_: [(1, 2)]
        for q, s, v in (([], 1, 1), ([[]], 1, 1), ([[0], [0, 1]], 1, 1),
                        ([[0]], 0, 1), ([[0]], True, 1), ([[0]], 1, True),
                        ([[0]], 1, 2), ([[math.nan]], 1, 1)):
            with self.assertRaises(ValueError):
                d.replay(q, project, s, v)

    def test_isolated_forward_traces(self):
        self.assertEqual(d.replay([[0]], lambda *_: [(0, 3)], 5, 1)[0][0]["outputs"], [3])
        self.assertEqual(d.replay([[0]], lambda *_: [(0, 9)], 5, 1)[0][0]["outputs"], [9])

    def test_demo_mismatch_is_observable(self):
        cases = d.demo()["strides"]
        self.assertEqual([cases[s]["cache_groups"] for s in ("1", "2", "5")], [5, 3, 1])
        self.assertNotEqual(cases["1"]["trace"][1]["outputs"], cases["2"]["trace"][1]["outputs"])
        self.assertEqual(cases["1"]["trace"][2]["outputs"], cases["2"]["trace"][2]["outputs"])


if __name__ == "__main__":
    unittest.main()
