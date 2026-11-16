"""Independent high-precision oracle and constructed controls; no RNG."""
import json
import math
from decimal import Decimal as D, localcontext
from pathlib import Path
import subprocess
import sys
import unittest
from .core import attend, select, softmax
from .demo import run


def oracle(q, keys, values, bias, indices):
    # Independent unshifted Decimal formula, not production softmax.
    with localcontext() as ctx:
        ctx.prec = 80
        scale = D(len(q)).sqrt()
        raw = [(sum(D(str(a))*D(str(b)) for a,b in zip(q,keys[j]))/scale
                + D(str(bias[j]))).exp() for j in indices]
        total = sum(raw)
        weights = [float(v/total) for v in raw]
        out = [float(sum(w*D(str(values[j][k])) for w,j in zip(raw,indices))/total)
               for k in range(len(values[0]))]
    return weights, out


class TestBias(unittest.TestCase):
    def setUp(self):
        self.q = [[0,1], [2,-1], [-3,2]]
        self.k = [[1,2], [-2,1], [0,-3]]
        self.v = [[1,-2], [4,3], [-5,2]]
        self.b = [2,-1,0]

    def assertClose(self, x, y):
        self.assertEqual(len(x),len(y))
        for a,b in zip(x,y): self.assertAlmostEqual(a,b,places=13)

    def test_decimal_oracle(self):
        for budget in (1,2,3):
            r = attend(self.q,self.k,self.v,self.b,budget)
            for q,w,y in zip(self.q,r['weights'],r['outputs']):
                ow,oy = oracle(q,self.k,self.v,self.b,r['indices'])
                self.assertClose(w,ow); self.assertClose(y,oy)

    def test_scalar_grid_oracle(self):
        for q in range(-8,9):
            for b in (-4,0,4):
                r=attend([[q]],[[1],[-1]],[[1],[-1]],[b,0])
                ow,oy=oracle([q],[[1],[-1]],[[1],[-1]],[b,0],[0,1])
                self.assertClose(r['weights'][0],ow); self.assertClose(r['outputs'][0],oy)

    def test_all_keys_is_dense(self):
        self.assertEqual(attend(self.q,self.k,self.v,self.b),attend(self.q,self.k,self.v,self.b,3))

    def test_bias_topk_not_query_topk(self):
        r=attend([[-8]],[[1],[-1]],[[1],[-1]],[4,0],1)
        dense=attend([[-8]],[[1],[-1]],[[1],[-1]],[4,0])
        self.assertEqual(r['indices'],[0]); self.assertGreater(dense['weights'][0][1],.999)

    def test_tie_break_and_order(self):
        self.assertEqual(select([0,2,2,1],2),[1,2])
        self.assertEqual(select([0,2,2,1],1),[1])
        self.assertEqual(select([2,0,3],2),[0,2])

    def test_bias_is_after_scaling(self):
        row=attend([[0,0]],[[1,1],[2,2]],[[0],[1]],[2,0])['weights'][0]
        self.assertAlmostEqual(row[0],1/(1+math.exp(-2)))

    def test_zero_queries_bias_only(self):
        r=attend([[0,0]],self.k,self.v,self.b)
        self.assertClose(r['weights'][0],softmax(self.b))

    def test_zero_bias_plain_attention(self):
        r=attend(self.q,self.k,self.v,[0]*3)
        for q,w in zip(self.q,r['weights']):
            raw=[math.exp(sum(a*b for a,b in zip(q,k))/math.sqrt(2)) for k in self.k]
            self.assertClose(w,[x/sum(raw) for x in raw])

    def test_multiplicative_gate_identity(self):
        plain=attend(self.q,self.k,self.v,[0]*3)
        biased=attend(self.q,self.k,self.v,self.b)
        for a,w in zip(plain['weights'],biased['weights']):
            raw=[x*math.exp(b) for x,b in zip(a,self.b)]
            self.assertClose(w,[x/sum(raw) for x in raw])

    def test_bias_common_shift(self):
        a=attend(self.q,self.k,self.v,self.b,2)
        b=attend(self.q,self.k,self.v,[x+1000 for x in self.b],2)
        self.assertEqual(a['indices'],b['indices'])
        for x,y in zip(a['outputs'],b['outputs']): self.assertClose(x,y)

    def test_row_shift(self):
        self.assertClose(softmax([1,2,-3]),softmax([1001,1002,997]))

    def test_extreme_logits(self):
        self.assertEqual(softmax([1000,-1000]),[1,0])
        self.assertClose(softmax([-1000,-1000]),[.5,.5])

    def test_renormalization(self):
        full=attend(self.q,self.k,self.v,self.b)
        sub=attend(self.q,self.k,self.v,self.b,2)
        for f,s in zip(full['weights'],sub['weights']):
            z=sum(f[i] for i in sub['indices'])
            self.assertClose(s,[f[i]/z for i in sub['indices']])
            self.assertAlmostEqual(sum(s),1)

    def test_scalar_error_identity(self):
        # y-y_kept = (1-retained_mass)*(y_dropped-y_kept).
        full=attend(self.q,self.k,self.v,self.b)
        sub=attend(self.q,self.k,self.v,self.b,2)
        for row,y,yp in zip(full['weights'],full['outputs'],sub['outputs']):
            mass=sum(row[j] for j in sub['indices'])
            for d in range(2):
                self.assertAlmostEqual(y[d]-yp[d],(1-mass)*(self.v[1][d]-yp[d]),places=13)

    def test_constant_values(self):
        for n in (1,2,3):
            r=attend(self.q,self.k,[[7,-4]]*3,self.b,n)
            for y in r['outputs']: self.assertClose(y,[7,-4])

    def test_convex_hull(self):
        for budget in (1,2,3):
            for y in attend(self.q,self.k,self.v,self.b,budget)['outputs']:
                for d in range(2):
                    self.assertGreaterEqual(y[d],min(v[d] for v in self.v))
                    self.assertLessEqual(y[d],max(v[d] for v in self.v))

    def test_permutation_without_ties(self):
        a=attend(self.q,self.k,self.v,self.b,2)
        p=[2,0,1]
        b=attend(self.q,[self.k[i] for i in p],[self.v[i] for i in p],[self.b[i] for i in p],2)
        for x,y in zip(a['outputs'],b['outputs']): self.assertClose(x,y)

    def test_all_queries_retained(self):
        r=attend(self.q,self.k,self.v,self.b,1)
        self.assertEqual(len(r['outputs']),len(self.q))
        self.assertEqual(len(r['weights'][0]),1)

    def test_demo_rare_query_and_mean(self):
        r=run()
        self.assertGreater(r['mean_retained_mass'],.97)
        self.assertLess(r['min_retained_mass'],.00001)
        self.assertGreater(r['max_absolute_output_error'],1.99)
        self.assertEqual(r['all_keys_max_error'],0)
        self.assertEqual(r['constant_values_max_error'],0)

    def test_invalid_budget(self):
        for n in (0,4,-1,True,1.5):
            with self.assertRaises(ValueError): select(self.b,n)

    def test_invalid_bias(self):
        for b in ([],[math.nan],[math.inf],[True]):
            with self.assertRaises(ValueError): select(b,1)

    def test_invalid_shapes(self):
        cases=[([],self.k,self.v,self.b),([[]],self.k,self.v,self.b),
               ([[1],[1,2]],self.k,self.v,self.b),([[1]],self.k,self.v,self.b),
               (self.q,self.k,self.v[:1],self.b),(self.q,self.k,self.v,[1])]
        for args in cases:
            with self.assertRaises(ValueError): attend(*args)

    def test_nonfinite_inputs(self):
        for bad in (math.nan,math.inf,-math.inf,True):
            with self.assertRaises(ValueError): attend([[bad]],[[1]],[[0]],[0])
            with self.assertRaises(ValueError): attend([[1]],[[bad]],[[0]],[0])
            with self.assertRaises(ValueError): attend([[1]],[[1]],[[bad]],[0])

    def test_invalid_softmax(self):
        for v in ([],[math.inf],[math.nan]):
            with self.assertRaises(ValueError): softmax(v)

    def test_sample(self):
        self.assertEqual(run(),json.loads(Path(__file__).with_name('sample_output.json').read_text()))

    def test_cli(self):
        actual=subprocess.check_output([sys.executable,'-m','examples.bias_pruning.demo'],text=True)
        self.assertEqual(actual,Path(__file__).with_name('sample_output.json').read_text())
