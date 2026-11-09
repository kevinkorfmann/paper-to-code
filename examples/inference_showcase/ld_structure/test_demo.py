import itertools
import json
import math
import unittest
from fractions import Fraction

import allel
import numpy as np
from examples.inference_showcase.ld_structure import demo


class TestLDStructure(unittest.TestCase):
    def assertMatches(self, table):
        got = demo.estimate(table)
        oracle = demo.exact_pair_oracle(table)
        if oracle['r_squared'] is None:
            self.assertIsNone(got['r'])
            self.assertIsNone(got['r_squared'])
        else:
            expected = oracle['sign']*math.sqrt(float(oracle['r_squared']))
            self.assertAlmostEqual(got['r'], expected, delta=2e-6)
            self.assertAlmostEqual(got['r_squared'], float(oracle['r_squared']), delta=4e-6)

    def test_sample(self):
        self.assertEqual(demo.run(), json.loads((demo.HERE/'sample_output.json').read_text()))

    def test_versions(self):
        self.assertEqual(demo.run()['versions'], {'python':'3.12.11', 'scikit_allel':'1.3.13', 'numpy':'2.3.5'})

    def test_all_fixtures(self):
        for x in demo.fixtures().values(): self.assertMatches(x)

    def test_exact_within_independence(self):
        for name in ['group_A','group_B']:
            t=demo.fixtures()[name]
            self.assertTrue(np.array_equal(t*t.sum(), np.outer(t.sum(axis=1),t.sum(axis=0))))
            self.assertEqual(demo.exact_pair_oracle(t)['r_squared'],0)

    def test_exact_pooled(self):
        self.assertEqual(demo.exact_pair_oracle(demo.fixtures()['pooled']), {'sign':1,'r_squared':Fraction(81,289)})

    def test_binary_tables_exhaustive(self):
        tested=0
        for bits in itertools.product([0,1], repeat=9):
            if sum(bits)>=2:
                self.assertMatches(np.asarray(bits).reshape(3,3)); tested+=1
        self.assertEqual(tested,502)

    def test_integer_weight_tables(self):
        for a,b,c,d in itertools.product(range(3), repeat=4):
            if a+b+c+d>=2: self.assertMatches([[a,0,b],[0,0,0],[c,0,d]])

    def test_variant_swap(self):
        for t in demo.fixtures().values():
            a,b=demo.estimate(t),demo.estimate(np.asarray(t).T)
            self.assertEqual(a['r'],b['r'])

    def test_single_allele_reversal(self):
        t=demo.fixtures()['pooled']
        self.assertAlmostEqual(demo.estimate(t)['r'],-demo.estimate(t[:,::-1])['r'],places=6)
        self.assertEqual(demo.exact_pair_oracle(t)['r_squared'],demo.exact_pair_oracle(t[:,::-1])['r_squared'])

    def test_both_allele_reversal(self):
        t=demo.fixtures()['pooled']
        self.assertAlmostEqual(demo.estimate(t)['r'],demo.estimate(t[::-1,::-1])['r'],places=6)

    def test_uniform_replication(self):
        t=demo.fixtures()['pooled']
        self.assertEqual(demo.estimate(t)['r'],demo.estimate(2*t)['r'])

    def test_sample_order(self):
        g=demo.genotypes(demo.fixtures()['pooled'])
        self.assertEqual(float(allel.rogers_huff_r(g)),float(allel.rogers_huff_r(g[:,::-1].copy())))

    def test_constant_undefined(self):
        for t in [np.ones((3,3),dtype=int)*0, [[0,0,0],[1,2,3],[0,0,0]]]:
            if np.sum(t)>=2: self.assertIsNone(demo.estimate(t)['r'])

    def test_centering_on_known_groups(self):
        a,b=(demo.genotypes(demo.fixtures()[k]).astype(float) for k in ['group_A','group_B'])
        centered=np.concatenate([a-a.mean(axis=1,keepdims=True), b-b.mean(axis=1,keepdims=True)],axis=1)
        self.assertAlmostEqual(float(np.corrcoef(centered)[0,1]),0,places=12)

    def test_moment_decomposition(self):
        g=demo.genotypes(demo.fixtures()['pooled']).astype(float)
        self.assertAlmostEqual(float(np.var(g[0])),17/25)
        self.assertAlmostEqual(float(np.mean((g[0]-1)*(g[1]-1))),9/25)

    def test_copy_safety(self):
        t=demo.fixtures()['pooled']; old=t.copy(); demo.estimate(t)
        np.testing.assert_array_equal(t,old)

    def test_invalid_shape(self):
        for x in [[],[1,2,3],[[1,2],[3,4]],np.ones((3,3,1),dtype=int)]:
            with self.assertRaises(ValueError): demo.estimate(x)

    def test_invalid_type_and_range(self):
        for x in [np.ones((3,3),dtype=float),np.ones((3,3),dtype=bool),np.full((3,3),-1),np.full((3,3),1001),np.full((3,3),2**64-1,dtype=np.uint64)]:
            with self.assertRaises(ValueError): demo.estimate(x)

    def test_too_few(self):
        for n in [0,1]:
            with self.assertRaises(ValueError): demo.estimate([[n,0,0],[0,0,0],[0,0,0]])

    def test_json_is_strict(self):
        json.dumps(demo.run(),allow_nan=False)


if __name__=='__main__': unittest.main()
