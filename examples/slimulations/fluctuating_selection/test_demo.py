import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
import subprocess
import unittest
from examples.slimulations.fluctuating_selection import demo as d


class Mathematics(unittest.TestCase):
    def test_two_generation_rational_path_enumeration(self):
        # Enumerate every ordered offspring vector, independently of binomial PMFs.
        for n in (2, 3, 4):
            for k in range(n+1):
                p = F(2*k, n+k)
                dist = [F(0)]*(n+1)
                for children in itertools.product((0, 1), repeat=n):
                    j = sum(children)
                    first = p**j*(1-p)**(n-j)
                    q = F(j, 2*n-j)
                    for grandchildren in itertools.product((0, 1), repeat=n):
                        z = sum(grandchildren)
                        dist[z] += first*q**z*(1-q)**(n-z)
                expected = sum(F(j,n)*v for j,v in enumerate(dist))
                result = d.oracle(n,k,2,(2,0.5))[-1]
                self.assertAlmostEqual(result['mean_frequency'], float(expected), places=13)
                self.assertAlmostEqual(result['loss_probability'], float(dist[0]), places=13)
                self.assertAlmostEqual(result['fixation_probability'], float(dist[-1]), places=13)

    def test_transition_rows(self):
        for n in (2, 17, 100, 200):
            for p in (0, 0.0001, 0.1, 0.5, 0.9999, 1):
                row = d.binomial_row(n,p)
                self.assertAlmostEqual(sum(row),1,places=12)
                self.assertAlmostEqual(sum(j*v for j,v in enumerate(row)),n*p,places=10)

    def test_one_step_expectation(self):
        for k in range(11):
            self.assertAlmostEqual(d.oracle(10,k,1,(1.8,0.4))[1]['mean_frequency'], 1.8*k/(10+0.8*k))

    def test_odds_multiply(self):
        out = d.deterministic(weights=(1.8,0.4))
        for g in range(0,13,2):
            odds = 0.72**(g//2)
            self.assertAlmostEqual(out[g], odds/(1+odds))

    def test_deterministic_endpoints(self):
        for w in (0.1,0.4,1,1.8,3):
            for k in (0,100):
                self.assertEqual(d.deterministic(k,100,40,(w,w)),[k/100]*41)

    def test_arithmetic_mean_is_misleading(self):
        self.assertGreater(sum((1.8,0.4))/2,1)
        self.assertLess(d.deterministic()[-1],0.5)
        self.assertGreater(d.deterministic(weights=(1.1,1.1))[-1],0.5)

    def test_order_invariance_only_deterministic(self):
        self.assertAlmostEqual(d.deterministic()[-1],d.deterministic(weights=(0.4,1.8))[-1])
        self.assertGreater(abs(d.oracle(20,10,12)[-1]['mean_frequency']-d.oracle(20,10,12,(0.4,1.8))[-1]['mean_frequency']),1e-5)

    def test_geometric_constant(self):
        self.assertAlmostEqual(d.deterministic()[-1],d.deterministic(weights=(math.sqrt(.72),)*2)[-1])

    def test_neutral_martingale_and_variance(self):
        n,k=20,7
        for g, val in enumerate(d.oracle(n,k,12,(1,1))):
            self.assertAlmostEqual(val['mean_frequency'],k/n)
            self.assertAlmostEqual(val['variance_frequency'],(k/n)*(1-k/n)*(1-(1-1/n)**g))

    def test_absorbing_endpoints(self):
        for k in (0,20):
            for row in d.oracle(20,k,12):
                self.assertAlmostEqual(row['mean_frequency'],k/20)
                self.assertAlmostEqual(row['variance_frequency'],0)

    def test_invalid_inputs(self):
        for args in [(1,0,1,(1,1)),(10,-1,1,(1,1)),(10,11,1,(1,1)),(10,1,0,(1,1)),(10,1,41,(1,1)),(10,1,1,(float('nan'),1)),(10,1,1,(0,1)),(10,1,1,(1,)),(True,0,1,(1,1))]:
            with self.subTest(args=args), self.assertRaises((ValueError,TypeError)):
                d.oracle(*args)


class Parsing(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(d.parse_output('banner\nDATA,0,0,1\nDATA,1,0,2\n',2,1,1,1),[[1],[2]])

    def test_reject_bad_rows(self):
        for text in ['DATA,0,0,1\n','DATA,0,0,1\nDATA,1,0,2\nDATA,1,0,2',
                     'DATA,0,0,0\nDATA,1,0,2', 'DATA,0,0,1\nDATA,1,0,3',
                     'DATA,0,0,1\nDATA,1,1,2', 'DATA,0,0,1\nDATA,1,0',
                     'DATA,0,0,1\nDATA,1,0,1.5']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                d.parse_output(text,2,1,1,1)


class Integration(unittest.TestCase):
    def test_schedule_and_binomial_oracle(self):
        for weights in [(1.8,0.4),(0.4,1.8),(1,1)]:
            rows=d.run(n=20,k=10,steps=4,weights=weights,replicates=512,seed=202611099)
            expected=d.oracle(20,10,4,weights)
            for g, counts in enumerate(rows):
                mean=sum(counts)/(20*len(counts))
                se=math.sqrt(expected[g]['variance_frequency']/len(counts))
                self.assertLessEqual(abs(mean-expected[g]['mean_frequency']),6*se+1e-10)

    def test_replay(self):
        args=dict(n=10,k=5,steps=3,replicates=4,seed=1234)
        self.assertEqual(d.run(**args), d.run(**args))

    def test_clonal_absorption(self):
        for k in (0,10):
            self.assertEqual(d.run(n=10,k=k,steps=3,replicates=4),[[k]*4]*4)

    def test_eidos_rejects_bad_input(self):
        for definition in ('N=2.5','K=-1','REPLICATES=0','STEPS=0','W1=0.0'):
            proc=subprocess.run(['slim','-s','123','-d',definition,str(d.HERE/'model.slim')],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)

    def test_sample_runtime_and_controls(self):
        saved=json.loads((d.HERE/'sample_summary.json').read_text())
        self.assertEqual(saved['runtime']['python'],'3.14.5')
        ts=saved['treatments']
        self.assertLess(ts['alternating']['trajectory'][-1]['mean_frequency'],.2)
        self.assertGreater(ts['constant_arithmetic']['trajectory'][-1]['mean_frequency'],.7)
        self.assertAlmostEqual(ts['neutral']['trajectory'][-1]['mean_frequency'],.5,delta=.04)

    def test_complete_sample_replay(self):
        saved=json.loads((d.HERE/'sample_summary.json').read_text())
        self.assertEqual(d.experiment(),saved)


if __name__=='__main__':
    unittest.main()
