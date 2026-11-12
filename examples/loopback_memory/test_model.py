"""Independent rational and Decimal oracles plus trajectory and CLI controls."""
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import unittest
from .model import allocate, prior, read, rollout
from .demo import result


class ModelTests(unittest.TestCase):
    def close(self,a,b):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b): self.assertAlmostEqual(x,float(y),places=11)

    def test_boundary_schedule(self):
        expected=[[1],[.5,.5],[.5,.5,1],[.25,.25,.5,.5]]
        for n,e in enumerate(expected,1): self.close([math.exp(x) for x in prior(n,2,.5)[0]],e)

    def test_independent_repeated_transport(self):
        for K in range(1,6):
            for lam in [F(1,7), F(1,2), F(9,10),F(1)]:
                bank=[]
                for n in range(1,31):
                    bank.append(F(1))
                    if n%K==0: bank=[x*lam for x in bank]
                    logs,B=prior(n,K,float(lam))
                    self.close([math.exp(x) for x in logs],bank)
                    self.assertAlmostEqual(B,float(sum(bank)),places=11)

    def test_complete_iteration_mass(self):
        for K in [1,2,7]:
            for R in [1,2,13,100]:
                for lam in [.2,.5,.99,1.]:
                    expected=K*sum(lam**j for j in range(1,R+1))
                    self.assertAlmostEqual(prior(K*R,K,lam)[1],expected,places=10)

    def test_mass_bound(self):
        for R in [1,3,20,100]: self.assertLess(prior(3*R,3,.7)[1],3*.7/.3)

    def test_no_decay(self): self.close(allocate([0]*20,3,1),[1]*20)

    def test_equal_scores_restore_prior(self):
        for a in [-700.,0.,700.]:
            self.close(allocate([a]*7,2,.5),[math.exp(x) for x in prior(7,2,.5)[0]])

    def test_exact_allocation(self):
        self.close(allocate([math.log(6),0,0,0],2,.5),[F(9,11),F(3,22),F(3,11),F(3,11)])

    def test_mass_preserved(self):
        for n in range(1,40):
            c=allocate([math.sin(i)*19 for i in range(n)],3,.3)
            self.assertTrue(all(x>=0 for x in c))
            self.assertAlmostEqual(sum(c),prior(n,3,.3)[1],places=12)

    def test_shift_invariance(self):
        s=[-3.,2.,7.,-1.]
        self.close(allocate(s,2,.7),allocate([x+100 for x in s],2,.7))

    def test_extreme_score(self): self.close(allocate([1000,-1000],2,.5),[1,0])

    def test_underflow_prior_can_be_recovered(self):
        # log prior must be retained even when its exponent rounds to zero.
        scores=[0.]*1100
        scores[0]=1100*math.log(2)
        c=allocate(scores,1,.5)
        self.assertGreater(c[0],.49)
        self.assertAlmostEqual(sum(c),1.,places=12)

    def test_decimal_reader_oracle(self):
        h=[.7,-.3,1.2]; u=[[1.,-.2,.3],[-.8,2.,.1],[.4,.5,-1.]]; q=[.2,-1.1,.7]
        out,c=read(h,u,q,2,.6,1e-5)
        with localcontext() as ctx:
            ctx.prec=80
            D=lambda x: Decimal(str(x))
            d=D(3); roots=d.sqrt()
            masses=[D('.6'),D('.6'),D(1)]
            raw=[]
            for row,m in zip(u,masses):
                norm=(sum(D(x)**2 for x in row)/d+D('0.00001')).sqrt()
                z=sum(D(a)*D(b)/norm/roots for a,b in zip(row,q))
                raw.append(m*z.exp())
            coeff=[sum(masses)*x/sum(raw) for x in raw]
            expected=[D(h[j])+sum(a*D(row[j]) for a,row in zip(coeff,u)) for j in range(3)]
        self.close(c,coeff); self.close(out,expected)

    def test_identity_zero_updates(self):
        self.close(read([3,-2],[[0,0]]*5,[9,-7],2,.5)[0],[3,-2])

    def test_zero_queries_recurrence(self):
        blocks=[lambda h:[h[0]*1.1+1],lambda h:[h[0]-.4]]
        for K in [1,2]:
            bs=blocks[:K]; state=2.; expected=[]
            for i in range(13):
                state=bs[i%K]([state])[0]
                if (i+1)%K==0: state=2+.6*(state-2)
                expected.append(state)
            states,_,_=rollout([2],bs,[[0]]*13,.6)
            self.close([x[0] for x in states],expected)

    def test_ordinary_composition(self):
        bs=[lambda h:[h[0]+1],lambda h:[h[0]*2]]
        states,_,_=rollout([3],bs,[[0]]*4,1)
        self.close([x[0] for x in states],[4,8,9,18])

    def test_replacement_not_accumulation(self):
        s,_,_=rollout([10],[lambda h:[h[0]+1]],[[0]]*4,.5)
        self.close([x[0] for x in s],[10.5,10.75,10.875,10.9375])

    def test_bounded_mass_unbounded_trajectory(self):
        s,_,_=rollout([1],[lambda h:[3*h[0]]],[[0]]*8,.5)
        self.close([x[0] for x in s],[2*(1.5**n)-1 for n in range(1,9)])
        self.assertLess(prior(8,1,.5)[1],1)

    def test_trajectory_updates_not_frozen_reference(self):
        blocks=[lambda h:[2*h[0],h[1]+1]]
        qs=[[1,-1],[-2,3],[1,1]]
        states,updates,_=rollout([1,2],blocks,qs,.8)
        self.close(updates[1],[states[0][0],1])
        self.close(updates[2],[states[1][0],1])
        self.assertNotAlmostEqual(updates[2][0],4.)

    def test_prefix_independence(self):
        bs=[lambda h:[1+h[0],h[1]-.2]]; qs=[[.2,1],[-.3,1],[4,-3]]
        full,_,_=rollout([2,3],bs,qs,.7)
        short,_,_=rollout([2,3],bs,qs[:2],.7)
        self.assertEqual(full[:2],short)

    def test_cumulative_coefficients(self):
        for weights in [[.2,.3,.5],[0,0,1],[1,0,0]]:
            effective=[sum(weights[i:]) for i in range(3)]
            self.assertEqual(effective[0],1)
            self.assertGreaterEqual(effective[0],effective[1])
            self.assertGreaterEqual(effective[1],effective[2])

    def test_small_and_large_updates(self):
        out,_=read([0],[[1e200]],[1],1,.5)
        self.assertEqual(out,[5e199])
        out,_=read([0],[[1e-200]],[1],1,.5)
        self.assertEqual(out,[5e-201])

    def test_invalid_schedule(self):
        for args in [(0,2,.5),(1,0,.5),(1,True,.5),(1,2,0),(1,2,1.1),(1,2,float('nan'))]:
            with self.subTest(args=args), self.assertRaises(ValueError): prior(*args)

    def test_invalid_reader(self):
        for args in [([1],[],[1],1,.5),([1],[[1,2]],[1],1,.5),([1],[[1]],[float('inf')],1,.5),([1],[[1]],[1],1,.5,0)]:
            with self.subTest(args=args), self.assertRaises(ValueError): read(*args)
        with self.assertRaises(ValueError): allocate([],1,.5)

    def test_sample_and_cli(self):
        expected=Path(__file__).with_name('sample_output.json').read_text()
        self.assertEqual(json.dumps(result(),indent=2,allow_nan=False)+'\n',expected)
        r=subprocess.run([sys.executable,'-m','examples.loopback_memory.demo'],text=True,capture_output=True,check=True)
        self.assertEqual(r.stdout,expected)
        subprocess.run([sys.executable,'-m','examples.loopback_memory.demo','--check-sample'],check=True,capture_output=True)

if __name__=='__main__': unittest.main()
