import itertools
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import random
import subprocess
import sys
import unittest
from .model import State, action, partial, merge, output
from .demo import results

class TestMerge(unittest.TestCase):
    def test_exact_two_groups(self):
        self.assertAlmostEqual(output(merge([partial([math.log(9)], [0]), partial([0], [10])])), 1)
    def test_equal_average_is_wrong(self):
        self.assertEqual(results()['wrong_equal_average_of_group_outputs'], 5)
    def test_skip_and_perturb(self):
        r = results()
        self.assertLess(r['absolute_error_perturb'], r['absolute_error_skip'])
    def test_decimal_reference(self):
        rng = random.Random(20261108)
        with localcontext() as ctx:
            ctx.prec = 60
            for n in range(1, 25):
                logits = [rng.uniform(-20, 20) for _ in range(n)]
                values = [rng.uniform(-10, 10) for _ in range(n)]
                weights = [Decimal.from_float(x).exp() for x in logits]
                expected = sum(w * Decimal.from_float(v) for w,v in zip(weights, values))/sum(weights)
                parts = [partial(logits[i:i+3], values[i:i+3]) for i in range(0,n,3)]
                self.assertAlmostEqual(output(merge(parts)), float(expected), places=12)
    def test_all_partition_boundaries(self):
        logits, values = [-3, 2, 0, 1, -1], [5, -2, 4, 8, 0]
        expected = output(partial(logits, values))
        for bits in itertools.product([0,1], repeat=4):
            bounds = [0] + [i+1 for i,b in enumerate(bits) if b] + [5]
            states = [partial(logits[a:b],values[a:b]) for a,b in zip(bounds,bounds[1:])]
            self.assertAlmostEqual(output(merge(states)),expected)
    def test_permutation(self):
        states = [partial([x],[v]) for x,v in zip([1,2,3],[4,5,6])]
        for p in itertools.permutations(states):
            self.assertAlmostEqual(output(merge(p)),output(merge(states)))
    def test_associativity(self):
        a,b,c = [partial([x],[x+1]) for x in [1,3,5]]
        self.assertAlmostEqual(output(merge([merge([a,b]),c])),output(merge([a,merge([b,c])])) )
    def test_shift(self):
        self.assertAlmostEqual(output(partial([1001,1002],[2,7])),output(partial([1,2],[2,7])))
    def test_extreme_logits(self):
        self.assertEqual(output(merge([partial([10000],[3]),partial([-10000],[9])])),3)
    def test_constant_values(self):
        self.assertAlmostEqual(output(partial([-10,5,9],[7,7,7])),7)
    def test_value_linearity(self):
        l,a,b=[1,2,3],[2,4,6],[1,-1,8]
        self.assertAlmostEqual(output(partial(l,[x+y for x,y in zip(a,b)])),output(partial(l,a))+output(partial(l,b)))
    def test_repeated_keys_change_mass(self):
        self.assertAlmostEqual(output(merge([partial([0],[0]),partial([0,0],[9,9])])),6)
    def test_invalid_inputs(self):
        for l,v in [([],[]),([1],[1,2]),([math.nan],[1]),([1],[math.inf]),([True],[0])]:
            with self.subTest(l=l,v=v), self.assertRaises(ValueError): partial(l,v)
    def test_invalid_states(self):
        for states in [[],[State(0,0,1)],[State(math.inf,1,1)],[State(0,1,math.nan)],[None]]:
            with self.subTest(states=states), self.assertRaises(ValueError): merge(states)
    def test_policy_example(self):
        self.assertEqual([action(6,j,2,4) for j in range(8)],['8-bit','skip','skip','4-bit','4-bit','8-bit','8-bit','masked'])
    def test_policy_protection(self):
        for i in range(30):
            self.assertEqual(action(i,0,1,1),'8-bit')
            self.assertEqual(action(i,i,1,1),'8-bit')
    def test_policy_boundaries(self):
        self.assertEqual(action(8,6,2,4),'4-bit')
        self.assertEqual(action(8,4,2,4),'skip')
    def test_no_low_precision_band(self):
        self.assertNotIn('4-bit',[action(9,j,3,3) for j in range(10)])
    def test_full_retention(self):
        self.assertNotIn('skip',[action(9,j,10,10) for j in range(10)])
    def test_policy_validation(self):
        for args in [(-1,0,1,1),(1,0,0,1),(1,0,2,1),(True,0,1,1),(1,0,1.0,2)]:
            with self.subTest(args=args),self.assertRaises(ValueError): action(*args)
    def test_sample(self):
        self.assertEqual(json.loads(Path(__file__).with_name('sample_output.json').read_text()),results())
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','examples.triattention_merge.demo','--check-sample'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__ == '__main__': unittest.main()
