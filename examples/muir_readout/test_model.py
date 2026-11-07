"""Independent rational, finite-ensemble and finite-difference checks."""
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from .model import fit_scale,lag_survival,loss,matched_gain,mixture_gain,readout
from .demo import results

class Tests(unittest.TestCase):
    c=(.5,.5); m=(.1,1); a=(.5,.95)
    def test_exact_optimum(self):
        c=(F(1,2),F(1,2)); m=(F(1,10),F(1)); a=(F(1,2),F(19,20))
        z=sum(x*y*y/w for x,y,w in zip(c,a,m)); expected=[y/w/(1+z) for y,w in zip(a,m)]
        for x,y in zip(matched_gain(self.c,self.m,self.a),expected): self.assertAlmostEqual(x,float(y),14)
    def test_exact_ensemble(self):
        # q_i = +/- .5 gives c_i=.25 exactly; independently enumerate signal
        # and each row's centered interference. No moment-closure approximation.
        c=(.25,)*4; m=(1.,4.,9.,16.); a=(.9,.7,.5,.2); g=(1.,.5,-.3,.1)
        errors=[]
        for bits in product((-1,1),repeat=9):
            q=[.5*x for x in bits[:4]]; v=bits[4]
            state=[[a[i]*q[i]*v+math.sqrt(m[i])*bits[5+i]] for i in range(4)]
            errors.append((readout(g,q,state)[0]-v)**2)
        self.assertAlmostEqual(sum(errors)/len(errors),loss(g,c,m,a)['mse'],12)
    def test_stationary_gradient(self):
        g=matched_gain(self.c,self.m,self.a)
        for i in range(2):
            hi=list(g);lo=list(g);hi[i]+=1e-5;lo[i]-=1e-5
            self.assertAlmostEqual((loss(hi,self.c,self.m,self.a)['mse']-loss(lo,self.c,self.m,self.a)['mse'])/2e-5,0,9)
    def test_global_quadratic_gap(self):
        g=matched_gain(self.c,self.m,self.a); delta=(.3,-.2); h=tuple(x+y for x,y in zip(g,delta))
        expected=sum(x*x*c*m for x,c,m in zip(delta,self.c,self.m))+sum(x*c*a for x,c,a in zip(delta,self.c,self.a))**2
        self.assertAlmostEqual(loss(h,self.c,self.m,self.a)['mse']-loss(g,self.c,self.m,self.a)['mse'],expected,14)
    def test_unit_signal(self): self.assertAlmostEqual(loss(matched_gain(self.c,self.m,self.a,unit_signal=True),self.c,self.m,self.a)['signal_gain'],1)
    def test_unit_signal_not_mse(self):
        g=matched_gain(self.c,self.m,self.a); h=matched_gain(self.c,self.m,self.a,unit_signal=True)
        self.assertLess(loss(g,self.c,self.m,self.a)['mse'],loss(h,self.c,self.m,self.a)['mse'])
    def test_fit_scale_matches_direction(self):
        d=[a/m for a,m in zip(self.a,self.m)]
        for x,y in zip(fit_scale(d,self.c,self.m,self.a),matched_gain(self.c,self.m,self.a)):self.assertAlmostEqual(x,y)
    def test_scale_invariance(self):
        for x,y in zip(fit_scale((1,2),self.c,self.m,self.a),fit_scale((10,20),self.c,self.m,self.a)):self.assertAlmostEqual(x,y)
    def test_lag_switch(self):
        g=matched_gain(self.c,self.m,lag_survival((.5,.95),1));h=matched_gain(self.c,self.m,lag_survival((.5,.95),10))
        self.assertGreater(g[0],g[1]); self.assertLess(h[0],h[1])
    def test_no_signal(self):self.assertEqual(matched_gain(self.c,self.m,(0,0)),(0,0))
    def test_no_unit_signal(self):
        with self.assertRaises(ValueError):matched_gain(self.c,self.m,(0,0),unit_signal=True)
    def test_readout(self):self.assertEqual(readout((2,3),(.5,2),((1,2),(3,4))),(19,26))
    def test_dimension_rejection(self):
        for fn in (lambda:matched_gain((1,),self.m,self.a),lambda:loss((1,),self.c,self.m,self.a),lambda:readout((1,),(1,),((1,),(2,))),lambda:fit_scale((1,),self.c,self.m,self.a)):
            with self.assertRaises(ValueError):fn()
    def test_invalid_moments(self):
        for c,m,a in (((),(),()),((1,1),self.m,self.a),(self.c,(0,1),self.a),(self.c,self.m,(-1,.5)),(self.c,self.m,(2,.5)),(self.c,(float('nan'),1),self.a)):
            with self.assertRaises(ValueError): matched_gain(c,m,a)
    def test_tiny_unit_signal_rejected(self):
        with self.assertRaises(ValueError):matched_gain((1,),(1,),(1e-160,),unit_signal=True)
    def test_tiny_noise_rejected(self):
        with self.assertRaises(ValueError):matched_gain((1,),(1e-320,),(1,))
    def test_large_direction_rescaled(self):
        self.assertEqual(fit_scale((1e200,),(1,),(1,),(1,)),(.5,))
    def test_channel_permutation(self):
        g=matched_gain(self.c,self.m,self.a)
        self.assertEqual(g[::-1],matched_gain(self.c[::-1],self.m[::-1],self.a[::-1]))
    def test_reviewer_oracle(self):
        g=matched_gain((.25,.75),(1,3),(.5,.75))
        self.assertAlmostEqual(g[0],32/77);self.assertAlmostEqual(g[1],16/77)
        self.assertAlmostEqual(loss(g,(.25,.75),(1,3),(.5,.75))['mse'],64/77)
        h=mixture_gain((.25,.75),(1,3),[(.5,.75),(.125,27/64)],(.4,.6))
        self.assertAlmostEqual(h[0],178712/727775);self.assertAlmostEqual(h[1],121408/727775)
    def test_variance(self):
        for s in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):matched_gain(self.c,self.m,self.a,s)
    def test_lag_validation(self):
        for lag in (0,-1,1.5,True):
            with self.assertRaises(ValueError):lag_survival((.5,),lag)
        for r in (0,1,2,float('nan')):
            with self.assertRaises(ValueError):lag_survival((r,),1)
    def test_mixture_single(self):
        for x,y in zip(mixture_gain(self.c,self.m,[self.a],[1]),matched_gain(self.c,self.m,self.a)):self.assertAlmostEqual(x,y)
    def test_mixture_exact_2x2(self):
        c=[F(1,2)]*2;m=[F(1,10),F(1)];aa=[[F(1,2),F(19,20)],[F(1,100),F(3,5)]];p=[F(1,3),F(2,3)]
        b=[[x*y for x,y in zip(c,a)] for a in aa]
        h=[[sum(w*z[i]*z[j] for w,z in zip(p,b))+(c[i]*m[i] if i==j else 0) for j in range(2)] for i in range(2)]
        v=[sum(w*z[i] for w,z in zip(p,b)) for i in range(2)];det=h[0][0]*h[1][1]-h[0][1]**2
        expected=[(v[0]*h[1][1]-v[1]*h[0][1])/det,(v[1]*h[0][0]-v[0]*h[1][0])/det]
        for x,y in zip(mixture_gain(c,m,aa,p),expected):self.assertAlmostEqual(x,float(y),13)
    def test_mixture_gradient(self):
        aa=[self.a,(.01,.6)];g=mixture_gain(self.c,self.m,aa,(.5,.5))
        for i in range(2):
            derivative=sum((sum(g[j]*self.c[j]*a[j] for j in range(2))-1)*self.c[i]*a[i] for a in aa)+2*self.c[i]*self.m[i]*g[i]
            self.assertAlmostEqual(derivative,0,14)
    def test_mixture_jensen_gap(self):
        aa=[self.a,(.01,.6)];avg=tuple((x+y)/2 for x,y in zip(*aa));g=(1.2,.7)
        actual=sum(loss(g,self.c,self.m,a)['mse']/2 for a in aa)-loss(g,self.c,self.m,avg)['mse']
        expected=sum(g[i]*self.c[i]*(aa[0][i]-aa[1][i])/2 for i in range(2))**2
        self.assertAlmostEqual(actual,expected,14)
    def test_mixture_validation(self):
        for aa,p in (([],[]),([self.a],[.5]),([self.a,self.a],[1]),([self.a,self.a],[0,1])):
            with self.assertRaises(ValueError):mixture_gain(self.c,self.m,aa,p)
    def test_demo_order(self):
        x=results();self.assertLess(x['adaptive_mixture_mse'],x['mixture']['mixture_optimum']['mse']);self.assertLess(x['mixture']['mixture_optimum']['mse'],x['mixture']['mean_survival_substitution']['mse'])
    def test_sample(self):self.assertEqual(results(),json.loads(Path(__file__).with_name('sample_output.json').read_text()))
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','examples.muir_readout.demo','--check-sample'],capture_output=True,text=True,check=True)
        self.assertEqual(r.stdout,'sample output matches\n')

if __name__=='__main__':unittest.main()
