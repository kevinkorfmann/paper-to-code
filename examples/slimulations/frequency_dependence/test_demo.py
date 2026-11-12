import itertools
import json
import math
from fractions import Fraction as F
import subprocess
import unittest
from unittest.mock import patch
from examples.slimulations.frequency_dependence import demo as d

class Mathematics(unittest.TestCase):
    def test_independent_parent_and_mendelian_enumeration(self):
        for counts in ((2,1,1),(1,2,1),(1,1,2),(0,4,0),(4,0,0),(0,0,4)):
            for s in (F(0),F(1,3),F(4,5)):
                n=sum(counts); p=F(counts[1]+2*counts[2],2*n); a=s*(1-2*p)
                parents=[(k,1+a*(k-1)) for k,c in enumerate(counts) for _ in range(c)]
                total=sum(w for _,w in parents); genotype=[F(0)]*3
                for (i,wi),(j,wj) in itertools.product(parents,repeat=2):
                    for x,y in itertools.product((0,1),repeat=2):
                        pi=F(i,2) if x else 1-F(i,2); pj=F(j,2) if y else 1-F(j,2)
                        genotype[x+y]+=wi*wj/total**2*pi*pj
                pred=d.prediction(counts,float(s))
                for actual,expected in zip(pred['expected_genotype_frequencies'],genotype): self.assertAlmostEqual(actual,float(expected))
                self.assertAlmostEqual(pred['expected_frequency'],float(genotype[1]/2+genotype[2]))
    def test_exact_rare_value(self):
        self.assertAlmostEqual(d.prediction((81,18,1),.8)['expected_frequency'],133/610)
    def test_allele_relabeling(self):
        for aa in range(6):
            for het in range(6-aa):
                c=(aa,het,5-aa-het)
                self.assertAlmostEqual(d.prediction(c,.8)['expected_frequency']+d.prediction(c[::-1],.8)['expected_frequency'],1)
    def test_neutral(self):
        for c in ((81,18,1),(40,20,40),(5,0,95)):
            p=d.prediction(c,0); self.assertEqual(p['expected_frequency'],p['initial_frequency'])
    def test_equilibrium(self):
        for c in ((25,50,25),(0,100,0),(50,0,50)):
            self.assertEqual(d.prediction(c,.8)['expected_frequency'],.5)
    def test_heterozygote_never_best(self):
        for c in ((81,18,1),(25,50,25),(1,18,81)):
            a,h,b=d.prediction(c,.8)['fitness_aa_Aa_AA']; self.assertLessEqual(min(a,b),h); self.assertGreaterEqual(max(a,b),h)
    def test_selected_change_direction(self):
        for c in ((81,18,1),(49,42,9),(9,42,49),(1,18,81)):
            p=d.prediction(c,.8); self.assertGreater((p['expected_frequency']-p['initial_frequency'])*(.5-p['initial_frequency']),0)
    def test_finite_binomial_moments(self):
        q=F(133,610); n=4
        probs=[math.comb(2*n,k)*q**k*(1-q)**(2*n-k) for k in range(2*n+1)]
        self.assertEqual(sum(probs),1)
        self.assertEqual(sum(F(k,2*n)*v for k,v in enumerate(probs)),q)
        self.assertEqual(sum((F(k,2*n)-q)**2*v for k,v in enumerate(probs)),q*(1-q)/(2*n))

class Contracts(unittest.TestCase):
    def test_invalid_parameters_before_execution(self):
        with patch.object(d,'check_slim_version') as m:
            for kw in ({'counts':(True,18,1)},{'counts':(-1,0,3)},{'counts':(1,0,0)},{'counts':(1001,0,0)},
                       {'strength':1},{'strength':-1},{'strength':float('nan')},{'strength':float('inf')},{'strength':True},
                       {'replicates':0},{'replicates':1.5},{'seed':0}):
                with self.subTest(kw=kw),self.assertRaises(ValueError): d.run(**kw)
            m.assert_not_called()
    def test_valid_parser(self): self.assertEqual(d.parse('INIT,0,2,1,1\nDATA,0,1,2,1',(2,1,1),1),[[1,2,1]])
    def test_bad_parser(self):
        good='INIT,0,2,1,1\nDATA,0,1,2,1'
        for bad in ('','DATA,0,1,2,1',good+'\nDATA,0,1,2,1',good.replace('INIT,0','INIT,1'),
                    good.replace('2,1,1','1,1,2'),good.replace('DATA,0,1,2,1','DATA,0,-1,4,1'),
                    good.replace('DATA,0,1,2,1','DATA,0,1,2'),good.replace('DATA,0,1,2,1','DATA,0,1,2,2')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):d.parse(bad,(2,1,1),1)
    def test_bad_observations_rejected(self):
        for rows in ([],[[81,18,1]],[[1,0,0]]*2):
            with self.assertRaises(ValueError):d.summarize(rows,(81,18,1),.8)
        with self.assertRaises(AssertionError):d.summarize([[0,0,100]]*1024,(81,18,1),.8)
    def test_wrong_version(self):
        with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,'SLiM version 5.2','')),self.assertRaises(ValueError): d.run()

class Integration(unittest.TestCase):
    def test_replay_seed(self):self.assertEqual(d.run(replicates=5),d.run(replicates=5))
    def test_absorbing_boundaries(self):
        for c in ((100,0,0),(0,0,100)):
            self.assertEqual(d.run(c,replicates=5),[list(c)]*5)
    def test_all_heterozygotes_segregate(self):
        rows=d.run((0,100,0),replicates=512)
        d.summarize(rows,(0,100,0),.8)
        self.assertTrue(all(sum(c[j] for c in rows)>0 for j in range(3)))
    def test_non_hwe_founders(self):
        for c in ((60,20,20),(20,20,60)):
            d.summarize(d.run(c,replicates=512),c,.8)
    def test_eidos_validation(self):
        for v in ('AA=-1','HET=0.5','HOM=2000','S=1.0','REPLICATES=0'):
            p=subprocess.run(['slim','-d',v,str(d.HERE/'model.slim')],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(p.returncode,0);self.assertIn('invalid',p.stderr)
    def test_actual_version(self):self.assertEqual(d.check_slim_version('slim'),'4.0.1')
    def test_sample_replay(self):self.assertEqual(json.loads(json.dumps(d.summary())),json.loads((d.HERE/'sample_summary.json').read_text()))
if __name__=='__main__':unittest.main()
