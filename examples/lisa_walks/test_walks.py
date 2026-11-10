"""Exact rational path enumeration is independent of feature-space recurrence."""
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from fractions import Fraction as F
from .core import diffuse, dense_transition, weights
from .demo import result


def oracle(q,k,v,beta,terms,epsilon=0):
    n=len(q)
    affinity=[[sum(F(x)*F(y) for x,y in zip(qi,kj)) for kj in k] for qi in q]
    s=[[x/(sum(row)+F(epsilon)) for x in row] for row in affinity]
    w=[F(beta)**h for h in range(terms)]
    total=sum(w)
    out=[]
    for start in range(n):
        values=[]
        for col in range(len(v[0])):
            answer=F(0)
            for h in range(terms):
                for tail in itertools.product(range(n),repeat=h):
                    path=(start,)+tail
                    prob=F(1)
                    for a,b in zip(path,path[1:]): prob*=s[a][b]
                    answer+=w[h]*prob*F(v[path[-1]][col])/total
            values.append(float(answer))
        out.append(values)
    return out


class WalkTests(unittest.TestCase):
    q=[[1,0,0],[0,1,0],[0,0,1]]
    k=[[0,0,0],[1,0,0],[0,1,1]]
    v=[[0],[0],[1]]
    def close(self,a,b):
        self.assertEqual(len(a),len(b))
        for ra,rb in zip(a,b):
            self.assertEqual(len(ra),len(rb))
            for x,y in zip(ra,rb): self.assertAlmostEqual(x,y,places=12)
    def test_chain_exact(self):
        self.close(diffuse(self.q,self.k,self.v),[[1/7],[3/7],[1]])
    def test_transition_orientation(self):
        self.assertEqual(dense_transition(self.q,self.k),[[0,1,0],[0,0,1],[0,0,1]])
    def test_zero_hop(self):
        self.assertEqual(diffuse(self.q,self.k,self.v,terms=1),self.v)
    def test_zero_beta(self):
        self.assertEqual(diffuse(self.q,self.k,self.v,beta=0),self.v)
    def test_conditioned_not_forced(self):
        self.assertEqual(weights(.5,3),[4/7,2/7,1/7])
        self.assertNotEqual(diffuse(self.q,self.k,self.v)[0][0],.25)
    def test_near_one(self):
        w=weights(math.nextafter(1.,0.),10)
        self.assertAlmostEqual(sum(w),1)
        for x in w: self.assertAlmostEqual(x,.1)
    def test_long_walk_limit(self):
        self.assertAlmostEqual(diffuse(self.q,self.k,self.v,terms=100)[0][0],.25)
    def test_identity(self):
        self.assertEqual(diffuse(self.q,self.q,self.v),self.v)
    def test_constants_preserved(self):
        q=[[1,2],[3,4],[1,0]]; k=[[1,0],[2,4],[1,2]]
        self.close(diffuse(q,k,[[2,-3]]*3,terms=5),[[2,-3]]*3)
    def test_epsilon_is_substochastic(self):
        self.close(diffuse(self.q,self.k,[[1]]*3,epsilon=1),[[.75]]*3)
        self.assertTrue(all(sum(r)<1 for r in dense_transition(self.q,self.k,epsilon=1)))
    def test_exact_oracle_many_small_factors(self):
        # All 81 positive 2x1 factor pairs, 3 betas, 4 lengths.
        vectors=list(itertools.product([1,2,3],repeat=2))
        for qa,ka,b,t in itertools.product(vectors,vectors,[F(0),F(1,2),F(3,4)],[1,2,3,4]):
            q=[[x] for x in qa]; k=[[x] for x in ka]; v=[[2,-1],[-3,4]]
            self.close(diffuse(q,k,v,float(b),t),oracle(q,k,v,b,t))
    def test_seeded_multifeature_oracle(self):
        rng=random.Random(20261110)
        for _ in range(40):
            q=[[rng.randrange(1,5) for _ in range(2)] for _ in range(3)]
            k=[[rng.randrange(1,5) for _ in range(2)] for _ in range(3)]
            v=[[rng.randrange(-3,4) for _ in range(2)] for _ in range(3)]
            self.close(diffuse(q,k,v,.75,4),oracle(q,k,v,F(3,4),4))
    def test_epsilon_oracle(self):
        self.close(diffuse(self.q,self.k,self.v,.5,4,.25),oracle(self.q,self.k,self.v,F(1,2),4,F(1,4)))
    def test_token_permutation(self):
        p=[2,0,1]; original=diffuse(self.q,self.k,self.v)
        self.close(diffuse([self.q[i] for i in p],[self.k[i] for i in p],[self.v[i] for i in p]),[original[i] for i in p])
    def test_query_row_scaling(self):
        self.close(diffuse([[v*scale for v in row] for row,scale in zip(self.q,[3,2,4])],self.k,self.v),diffuse(self.q,self.k,self.v))
    def test_global_key_scaling(self):
        self.close(diffuse(self.q,[[v*7 for v in row] for row in self.k],self.v),diffuse(self.q,self.k,self.v))
    def test_value_linearity(self):
        a=diffuse(self.q,self.k,self.v)
        self.close(diffuse(self.q,self.k,[[2*x+3 for x in row] for row in self.v]),[[2*x+3 for x in row] for row in a])
    def test_convex_hull(self):
        q=[[1,3],[2,1],[4,5]]; k=[[1,2],[3,2],[2,3]]
        out=diffuse(q,k,[[-2],[5],[1]],.8,5)
        self.assertTrue(all(-2<=r[0]<=5 for r in out))
    def test_noncausal_by_design(self):
        a=diffuse(self.q,self.k,self.v)
        b=diffuse(self.q,self.k,[[0],[0],[100]])
        self.assertGreater(b[0][0],a[0][0])
    def test_invalid_beta_and_terms(self):
        for b in [-1,1,math.inf,math.nan,True,'x',10**1000]:
            with self.assertRaises(ValueError): weights(b,3)
        for t in [0,-1,1.2,True]:
            with self.assertRaises(ValueError): weights(.5,t)
    def test_invalid_matrices(self):
        for q in [[],[[1],[1,2]],[[math.nan]],[[True]],[[-1]]]:
            with self.assertRaises(ValueError): diffuse(q,[[1]],[[1]])
        with self.assertRaises(ValueError): diffuse([[1]],[[1],[2]],[[1]])
        with self.assertRaises(ValueError): diffuse([[1,2]],[[1]],[[1]])
        with self.assertRaises(ValueError): diffuse([[1]],[[1]],[[math.inf]])
    def test_zero_mass(self):
        with self.assertRaises(ValueError): diffuse([[0]],[[1]],[[1]])
        # Stabilization allows zero rows, but all nonzero-hop mass disappears.
        self.close(diffuse([[0]],[[1]],[[1]],epsilon=1),[[4/7]])
    def test_invalid_epsilon(self):
        for e in [-1,math.nan,math.inf,True,'x',10**1000]:
            with self.assertRaises(ValueError): diffuse(self.q,self.k,self.v,epsilon=e)
    def test_overflow_rejected(self):
        with self.assertRaises(ValueError): diffuse([[1e308]],[[1e308]],[[1]])
        with self.assertRaises(ValueError): diffuse([[1]],[[10**1000]],[[1]])
    def test_input_not_mutated(self):
        before=json.dumps([self.q,self.k,self.v])
        diffuse(self.q,self.k,self.v)
        self.assertEqual(before,json.dumps([self.q,self.k,self.v]))
    def test_sample(self):
        self.assertEqual(json.loads(Path(__file__).with_name('sample_output.json').read_text()),result())
    def test_cli(self):
        proc=subprocess.run([sys.executable,'-m','examples.lisa_walks.demo','--check-sample'],check=True,capture_output=True,text=True)
        self.assertEqual(proc.stdout,'Sample output matches.\n')


if __name__=='__main__': unittest.main()
