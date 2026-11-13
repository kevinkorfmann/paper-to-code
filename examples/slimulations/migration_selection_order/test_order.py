"""Independent rational enumeration and actual SLiM integration checks."""
import json
import math
from fractions import Fraction as F
from functools import lru_cache
import subprocess
import sys
import unittest
from . import demo as d


def exact(n,k,m,w,order):
    # Enumerate both count outcomes; use raw integer fitness totals, not d.select.
    prob1=F(w*k,w*k+n-k) if order=='SM' else (1-m)*F(k,n)
    mean=F(0); sq=F(0)
    for j in range(n+1):
        pj=math.comb(n,j)*prob1**j*(1-prob1)**(n-j)
        prob2=(1-m)*F(j,n) if order=='SM' else F(w*j,w*j+n-j)
        for z in range(n+1):
            pz=math.comb(n,z)*prob2**z*(1-prob2)**(n-z)
            mean+=pj*pz*F(z,n); sq+=pj*pz*F(z*z,n*n)
    return mean,sq-mean*mean


class Theory(unittest.TestCase):
    def test_rational_enumeration(self):
        for n in (2,3,5):
            for k in range(n+1):
                for order in ('SM','MS'):
                    for m,w in ((F(0),F(1)),(F(1,3),F(2)),(F(1),F(3,2))):
                        mean,var=exact(n,k,m,w,order)
                        result=d.prediction(n,k,float(m),float(w),order)
                        self.assertAlmostEqual(result['expected_final_mean'],float(mean),places=13)
                        self.assertAlmostEqual(result['expected_final_variance'],float(var),places=13)
    def test_opposite_direction(self):
        self.assertLess(d.prediction(order='SM')['expected_final_mean'],.5)
        self.assertGreater(d.prediction(order='MS')['expected_final_mean'],.5)
    def test_deterministic_composition(self):
        self.assertAlmostEqual(d.prediction(order='SM')['deterministic_final_frequency'],7/15)
        self.assertAlmostEqual(d.prediction(order='MS')['deterministic_final_frequency'],14/27)
    def test_selection_last_jensen(self):
        x=d.prediction(order='MS'); self.assertLess(x['expected_final_mean'],x['deterministic_final_frequency'])
    def test_selection_first_linear_last(self):
        x=d.prediction(order='SM'); self.assertAlmostEqual(x['expected_final_mean'],x['deterministic_final_frequency'])
    def test_neutral_means_not_variances(self):
        a=d.prediction(weight=1,order='SM'); b=d.prediction(weight=1,order='MS')
        self.assertAlmostEqual(a['expected_final_mean'],.35); self.assertAlmostEqual(b['expected_final_mean'],.35)
        self.assertNotAlmostEqual(a['expected_final_variance'],b['expected_final_variance'])
    def test_no_migration_still_stage_sampling(self):
        a=d.prediction(migration=0,order='SM'); b=d.prediction(migration=0,order='MS')
        self.assertLess(b['expected_final_mean'],a['expected_final_mean'])
    def test_absent_boundary(self):
        for order in ('SM','MS'):
            self.assertEqual(d.prediction(k=0,order=order)['expected_final_mean'],0)
    def test_fixed_no_migration(self):
        for order in ('SM','MS'):
            self.assertEqual(d.prediction(k=100,migration=0,order=order)['expected_final_mean'],1)
    def test_complete_migration(self):
        for order in ('SM','MS'):
            self.assertEqual(d.prediction(migration=1,order=order)['expected_final_variance'],0)
    def test_binomial_normalization(self):
        for q in (0,.01,.3,.5,.99,1):
            probs=d.binomial(100,q); self.assertAlmostEqual(sum(probs),1)
            self.assertAlmostEqual(sum(i*p for i,p in enumerate(probs)),100*q)
    def test_invalid_arguments(self):
        for kw in ({'n':True},{'n':1},{'k':101},{'k':-1},{'migration':float('nan')},
                   {'migration':1.1},{'weight':0},{'weight':float('inf')},{'order':'bad'}):
            with self.assertRaises((ValueError,TypeError)): d.prediction(**kw)
    def test_parser_valid(self):
        self.assertEqual(d.parse('DATA,0,0,1\nDATA,1,0,2\nDATA,2,0,0',2,1,1),[[1],[2],[0]])
    def test_parser_invalid(self):
        base='DATA,0,0,1\nDATA,1,0,2\nDATA,2,0,0'
        for text in (base+'\nDATA,1,0,2',base.replace('DATA,2','DATA,3'),base.replace(',2\n',',3\n'),
                     base.replace('DATA,0,0,1','DATA,0,0,0'),base.rsplit('\n',1)[0],base+'\nDATA,1'):
            with self.assertRaises(ValueError): d.parse(text,2,1,1)
    def test_reject_bad_summary(self):
        with self.assertRaises(ValueError): d.summarize([[1],[2],[0]],2,1,.3,2,'SM')


@lru_cache(maxsize=1)
def observed(): return d.summary()


class Simulator(unittest.TestCase):
    def test_pinned_versions(self):
        self.assertEqual(observed()['python'],'3.14.5')
        self.assertIn('4.0.1',observed()['slim'])
    def test_saved_output(self):
        self.assertEqual(observed(),json.loads((d.HERE/'sample_summary.json').read_text()))
    def test_opposite_observed_direction(self):
        t=observed()['treatments']; self.assertLess(t['selection_then_migration']['observed_final_mean'],.5)
        self.assertGreater(t['migration_then_selection']['observed_final_mean'],.5)
    def test_all_variances(self):
        for t in observed()['treatments'].values():
            for stage in ('first','final'):
                # Broad predeclared moment check in addition to seven-SE mean checks.
                exp=t[f'expected_{stage}_variance']; obs=t[f'observed_{stage}_variance']
                self.assertLessEqual(abs(obs-exp),.2*exp+1e-12)
    def test_boundary_runs(self):
        for name in ('complete_migration','absent_A'):
            self.assertEqual(observed()['treatments'][name]['observed_final_mean'],0)
    def test_seed_repeat(self):
        args=dict(n=10,k=4,replicates=8,seed=202611139)
        self.assertEqual(d.run(**args),d.run(**args))
    def test_fixed_control_run(self):
        rows=d.run(n=10,k=10,migration=0,order='MS',replicates=8)
        self.assertTrue(all(row==[10]*8 for row in rows))
    def test_invalid_slim_constants(self):
        for definition in ('M=2.0','W=0.0','ORDER=4','N=1','REPLICATES=0'):
            result=subprocess.run(['slim','-d',definition,str(d.HERE/'model.slim')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
    def test_cli(self):
        result=subprocess.run([sys.executable,'-m','examples.slimulations.migration_selection_order.demo','--check-sample'],capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'sample output matches')

if __name__=='__main__': unittest.main()
