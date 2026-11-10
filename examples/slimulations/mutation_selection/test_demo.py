"""Independent rational parent/mutation enumeration and simulator controls."""
from fractions import Fraction as F
import itertools
import math
import shutil
import unittest
from . import demo as d


def enumerate_row(n, k, w, u, v):
    # Independently enumerate parent identity then state transition for each child.
    outcomes = []
    total = k*w+n-k
    for parent in range(n):
        if parent < k:
            outcomes += [(1, w/total*(1-v)), (0, w/total*v)]
        else:
            outcomes += [(1, u/total), (0, (1-u)/total)]
    row = [F(0)]*(n+1)
    for children in itertools.product(outcomes, repeat=n):
        row[sum(x[0] for x in children)] += math.prod(x[1] for x in children)
    return row


class Mathematics(unittest.TestCase):
    def test_parent_enumeration(self):
        for n in (2,3):
            for k in range(n+1):
                for w,u,v in [(F(4,5),F(1,5),F(1,10)),(F(1),F(0),F(0)),(F(1,2),F(1),F(1))]:
                    ref = enumerate_row(n,k,w,u,v)
                    got = d.binomial_row(n,d.offspring_probability(k/n,float(w),float(u),float(v)))
                    for x,y in zip(got,ref): self.assertAlmostEqual(x,float(y),places=13)
    def test_two_generations_exact(self):
        n=3; w,u,v=F(4,5),F(1,5),F(1,10)
        matrix=[enumerate_row(n,k,w,u,v) for k in range(n+1)]
        prob=matrix[1]
        prob=[sum(prob[i]*matrix[i][j] for i in range(n+1)) for j in range(n+1)]
        mean=sum(F(i,n)*prob[i] for i in range(n+1))
        self.assertAlmostEqual(d.oracle(n,1,2,float(w),float(u),float(v))[-1]['mean_frequency'],float(mean))
    def test_neutral_closed_form(self):
        for k in (0,3,10):
            got=d.oracle(10,k,12,1,.1,.2)
            for t,row in enumerate(got):
                self.assertAlmostEqual(row['mean_frequency'],1/3+(k/10-1/3)*.7**t)
    def test_no_forward_absorption(self):
        self.assertTrue(all(x['absence_probability']==1 for x in d.oracle(u=0)))
    def test_no_backward_absorption(self):
        self.assertTrue(all(x['all_A_probability']==1 for x in d.oracle(k=100,v=0)))
    def test_mutation_breaks_absorption(self):
        self.assertAlmostEqual(d.offspring_probability(0,.8,.02,.01),.02)
        self.assertAlmostEqual(d.offspring_probability(1,.8,.02,.01),.99)
    def test_equilibrium(self):
        for w in (.1,.5,.8,1.):
            p=d.equilibrium(w,.02,.01)
            self.assertAlmostEqual(d.offspring_probability(p,w,.02,.01),p)
        self.assertAlmostEqual(d.equilibrium(1,.02,.01),2/3)
    def test_no_back_fixed_point_u_over_s(self):
        for s,u in [(.2,.02),(.5,.03)]:
            p=u/s
            self.assertAlmostEqual(d.offspring_probability(p,1-s,u,0),p)
    def test_stronger_selection(self):
        self.assertLess(d.equilibrium(.5),d.equilibrium(.8))
    def test_finite_mean_not_deterministic(self):
        finite=d.oracle(n=10,k=5,steps=2)[-1]['mean_frequency']
        self.assertNotAlmostEqual(finite,d.deterministic(n=10,k=5,steps=2)[-1],places=5)
    def test_oracle_mass_and_variance(self):
        for r in d.oracle(n=10,steps=4):
            self.assertGreaterEqual(r['variance_frequency'],0)
            self.assertLessEqual(r['absence_probability']+r['all_A_probability'],1+1e-14)
    def test_extreme_probabilities(self):
        self.assertEqual(d.offspring_probability(1,.1,0,0),1)
        self.assertEqual(d.offspring_probability(0,.1,1,0),1)
        for p in (0,.3,1):
            self.assertEqual(d.offspring_probability(p,.5,1,0),1)
            self.assertEqual(d.offspring_probability(p,.5,0,1),0)
    def test_validation(self):
        for kw in [dict(n=True),dict(n=201),dict(k=-1),dict(steps=81),dict(w=0),dict(u=-1),dict(v=float('nan'))]:
            with self.assertRaises(ValueError): d.oracle(**kw)
        for p in (-1,2,float('inf'),True):
            with self.assertRaises(ValueError): d.offspring_probability(p,.8,.02,.01)
        for uv in [(0,.1),(.1,0),(.8,.8)]:
            with self.assertRaises(ValueError): d.equilibrium(.8,*uv)
    def test_parser(self):
        self.assertEqual(d.parse_output('header\nDATA,0,0,0\nDATA,1,0,1',2,0,1,1),[[0],[1]])
        for text in ['DATA,0,0,0','DATA,0,0,0\nDATA,1,0,3','DATA,0,0,0\nDATA,1,0,1\nDATA,1,0,1','DATA,0,0,1\nDATA,1,0,1','DATA,0,0,0,0']:
            with self.assertRaises(ValueError): d.parse_output(text,2,0,1,1)


@unittest.skipUnless(shutil.which('slim'),'SLiM not on PATH')
class Simulator(unittest.TestCase):
    def test_no_forward(self):
        self.assertTrue(all(k==0 for row in d.run(n=10,steps=3,u=0,replicates=8) for k in row))
    def test_no_backward(self):
        self.assertTrue(all(k==10 for row in d.run(n=10,k=10,steps=3,v=0,replicates=8) for k in row))
    def test_flip_each_clone(self):
        rows=d.run(n=10,k=0,steps=4,u=1,v=1,replicates=8)
        self.assertEqual(rows,[[0]*8,[10]*8,[0]*8,[10]*8,[0]*8])
    def test_mutation_one_step_binomial(self):
        rows=d.run(n=20,k=0,steps=1,u=.3,v=.2,replicates=512)
        mean=sum(rows[1])/512/20
        self.assertLess(abs(mean-.3),6*math.sqrt(.3*.7/20/512))
    def test_seed_replay(self):
        args=dict(n=10,k=3,steps=3,replicates=8)
        self.assertEqual(d.run(**args),d.run(**args))


if __name__=='__main__': unittest.main()
