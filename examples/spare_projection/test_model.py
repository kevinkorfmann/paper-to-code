import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from fractions import Fraction as F
from .model import basis, compact_step, project, read, reconstruct, update
from .demo import output, results


def oracle(c, p, k, v, a, b, eps=F(1, 100000000), cap=F(16), compensate=True):
    # Exact rational matrix recurrence: C [a I - a b g khat khat^T] + b g v khat^T.
    z = [sum(p[i][j] * k[i] for i in range(len(k))) for j in range(len(p[0]))]
    rho = sum(x*x for x in z)
    g = min(1 / max(rho, eps), cap) if compensate else F(1)
    transition = [[a * ((i == j) - b*g*z[i]*z[j]) for j in range(len(z))] for i in range(len(z))]
    return [[sum(row[i]*transition[i][j] for i in range(len(z))) + b*g*value*z[j]
             for j in range(len(z))] for row, value in zip(c, v)]


class TestSpare(unittest.TestCase):
    def assertMatrix(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(len(x), len(y))
            for u, v in zip(x, y):
                self.assertAlmostEqual(u, float(v), places=11)

    def test_exact_rational_sequence(self):
        rng = random.Random(20261113)
        for p in [[[F(1)], [F(0)]], [[F(3,5)], [F(4,5)]], [[F(1),F(0)],[F(0),F(1)]]]:
            c = [[F(0)]*len(p[0]) for _ in range(2)]
            got = [[float(x) for x in row] for row in c]
            pf = [[float(x) for x in row] for row in p]
            for _ in range(60):
                k = rng.choice([[F(3,5),F(4,5)],[F(4,5),F(-3,5)],[F(0),F(1)]])
                v = [F(rng.randrange(-4,5),3),F(rng.randrange(-4,5),3)]
                a,b = F(rng.randrange(5),4),F(rng.randrange(5),4)
                cap,eps = rng.choice([F(1),F(2),F(16)]),rng.choice([F(1,100),F(1,2)])
                c=oracle(c,p,k,v,a,b,eps,cap)
                got,_=compact_step(got,pf,list(map(float,k)),list(map(float,v)),float(a),float(b),epsilon=float(eps),cap=float(cap))
                self.assertMatrix(got,c)

    def test_single_write_compensation(self):
        c,info=compact_step([[0]],[[1],[0]],[.6,.8],[2],1,.75)
        self.assertAlmostEqual(read(c,[.6])[0],1.5)
        self.assertAlmostEqual(info['effective_write'],.75)

    def test_uncompensated(self):
        c,_=compact_step([[0]],[[1],[0]],[.6,.8],[2],1,.75,compensate=False)
        self.assertAlmostEqual(read(c,[.6])[0],.54)

    def test_cap(self):
        c,info=compact_step([[0]],[[1],[0]],[.6,.8],[1],1,1,cap=2)
        self.assertAlmostEqual(info['gamma'],2)
        self.assertAlmostEqual(read(c,[.6])[0],.72)

    def test_floor(self):
        c,_=compact_step([[0]],[[1],[0]],[.6,.8],[1],1,1,epsilon=.5)
        self.assertAlmostEqual(read(c,[.6])[0],.72)

    def test_zero_projection(self):
        c,info=compact_step([[4]],[[1],[0]],[0,1],[1],.5,1)
        self.assertEqual(c,[[2]])
        self.assertEqual(info['effective_write'],0)

    def test_effective_strength_bound(self):
        for x in [0,.001,.1,.6,.999,1]:
            for cap in [1,2,100]:
                for eps in [1e-8,.2,1]:
                    _,d=compact_step([[0]],[[1],[0]],[x,math.sqrt(1-x*x)],[1],.8,.9,epsilon=eps,cap=cap)
                    self.assertLessEqual(d['effective_write'],.9+1e-14)
                    self.assertGreaterEqual(d['effective_write'],0)

    def test_key_read_identity(self):
        for alpha in [0,.7,1]:
            for beta in [0,.3,1]:
                old=2.5*.6
                c,d=compact_step([[2.5]],[[1],[0]],[.6,.8],[-2],alpha,beta,cap=2)
                expected=(1-d['effective_write'])*alpha*old+d['effective_write']*(-2)
                self.assertAlmostEqual(read(c,[.6])[0],expected)

    def test_invariant_subspace_sequence(self):
        p=[[.6],[.8],[0]]
        c=[[2],[-1]]
        dense=reconstruct(c,p)
        for sign in [1,-1,1,1,-1]:
            k=[sign*.6,sign*.8,0]
            v=[3,-2]
            c,_=compact_step(c,p,k,v,.9,.7)
            dense=update(dense,k,v,.9,.7)
            self.assertMatrix(reconstruct(c,p),dense)
            for q in [[1,2,3],[-2,4,-9]]:
                self.assertMatrix([read(c,project(p,q))],[read(dense,q)])

    def test_full_rank_rotation(self):
        p=[[.6,-.8],[.8,.6]]
        c=[[0,0]]; dense=[[0,0]]
        for k,v in [([1,0],[2]),([.6,.8],[-3]),([0,1],[1])]:
            c,_=compact_step(c,p,k,v,.8,.6)
            dense=update(dense,k,v,.8,.6)
            self.assertMatrix(reconstruct(c,p),dense)

    def test_collision_counterexample(self):
        r=results()['orthogonal_key_collision']
        self.assertMatrix([r['dense_reads']],[[1,-1]])
        self.assertMatrix([r['compact_reads']],[[-.75,-1]])

    def test_basis_sign_invariance(self):
        a,_=compact_step([[0]],[[1],[0]],[.6,.8],[1],.9,.7)
        b,_=compact_step([[0]],[[-1],[0]],[.6,.8],[1],.9,.7)
        self.assertMatrix(reconstruct(a,[[1],[0]]),reconstruct(b,[[-1],[0]]))

    def test_value_linearity(self):
        p=[[1],[0]]
        def run(v):
            c=[[0]]
            for k,x in zip([[.6,.8],[.8,-.6]],v):
                c,_=compact_step(c,p,k,[x],.9,.7)
            return c[0][0]
        self.assertAlmostEqual(run([3,1]),run([1,2])+run([2,-1]))

    def test_causal_prefix(self):
        p=[[1],[0]]; c=[[0]]
        c,_=compact_step(c,p,[.6,.8],[1],.9,.7)
        snapshot=[row[:] for row in c]
        compact_step(c,p,[.8,-.6],[-5],.9,.7)
        self.assertEqual(c,snapshot)

    def test_beta_zero(self):
        c,_=compact_step([[4]],[[1],[0]],[.6,.8],[8],.5,0)
        self.assertEqual(c,[[2]])

    def test_alpha_zero(self):
        c,_=compact_step([[4]],[[1],[0]],[.6,.8],[2],0,.3)
        self.assertAlmostEqual(read(c,[.6])[0],.6)

    def test_nonunit_key_rejected(self):
        with self.assertRaises(ValueError): compact_step([[0]],[[1],[0]],[.3,.4],[1],1,1)

    def test_invalid_basis(self):
        for p in [[],[[1,0]],[[1],[1]],[[1,0],[0,0]],[[float('nan')]]]:
            with self.assertRaises(ValueError): basis(p)

    def test_invalid_gates(self):
        for x in [-.1,1.1,float('nan'),float('inf'),True]:
            for name in ['alpha','beta']:
                args={'alpha':.5,'beta':.5}; args[name]=x
                with self.assertRaises(ValueError): compact_step([[0]],[[1],[0]],[1,0],[1],**args)

    def test_invalid_compensation(self):
        for kw in [{'epsilon':0},{'epsilon':2},{'epsilon':float('nan')},{'cap':.5},{'cap':float('inf')},{'compensate':1}]:
            with self.assertRaises(ValueError): compact_step([[0]],[[1],[0]],[1,0],[1],1,1,**kw)

    def test_bad_dimensions(self):
        for f in [lambda:read([[1,2]],[1]),lambda:project([[1],[0]],[1]),lambda:update([[0]],[1],[1,2],1,1),lambda:reconstruct([[1,2]],[[1],[0]])]:
            with self.assertRaises(ValueError): f()

    def test_nonfinite_state(self):
        with self.assertRaises(ValueError): compact_step([[float('inf')]],[[1],[0]],[1,0],[1],1,1)

    def test_sample(self):
        self.assertEqual(output(),Path(__file__).with_name('sample_output.json').read_text())
        self.assertEqual(json.loads(output())['discarded_key_read'],0)

    def test_cli(self):
        p=subprocess.run([sys.executable,'-m','examples.spare_projection.demo','--check-sample'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('verified',p.stdout)


if __name__=='__main__': unittest.main()
