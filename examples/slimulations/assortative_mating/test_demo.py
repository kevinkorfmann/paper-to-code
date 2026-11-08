import itertools
import json
import math
from fractions import Fraction as F
import subprocess
import unittest
from examples.slimulations.assortative_mating.demo import HERE, oracle, parse_output, run, summarize, validate


def exact(n, q):
    # Enumerate ordered offspring genotype outcomes, independent of moment formula.
    probs = [q/2, 1-q, q/2]
    sums = {k: F(0) for k in ('p','p2','h','h2','d','d2','mass')}
    for dosages in itertools.product(range(3), repeat=n):
        w = math.prod(probs[d] for d in dosages)
        p = F(sum(dosages), 2*n)
        h = F(dosages.count(1), n)
        d = 2*p*(1-p)
        for k, v in [('p',p),('p2',p*p),('h',h),('h2',h*h),('d',d),('d2',d*d),('mass',1)]:
            sums[k] += w*v
    return sums


class Theory(unittest.TestCase):
    def test_exact_enumeration(self):
        for n in (4,6):
            for q in (F(0),F(1,10),F(1,2),F(9,10),F(1)):
                e = exact(n,q)
                self.assertEqual(e['mass'], 1)
                o = oracle(n,float(q))
                for key, letter in [('frequency','p'),('observed_heterozygosity','h'),('gene_diversity','d')]:
                    self.assertAlmostEqual(o['mean_'+key],float(e[letter]),places=14)
                    self.assertAlmostEqual(o['variance_'+key],float(e[letter+'2']-e[letter]**2),places=14)
    def test_random_mating(self):
        self.assertEqual(oracle(200,.5)['genotype_probabilities_aa_Aa_AA'],[.25,.5,.25])
    def test_deterministic_opposites(self):
        o=oracle(200,0)
        self.assertEqual(o['mean_observed_heterozygosity'],1)
        self.assertEqual(o['variance_frequency'],0)
    def test_same_only(self):
        self.assertEqual(oracle(200,1)['mean_observed_heterozygosity'],0)
    def test_gene_diversity_is_not_heterozygosity(self):
        self.assertGreater(oracle(200,.9)['mean_gene_diversity'],.49)
        self.assertAlmostEqual(oracle(200,.9)['mean_observed_heterozygosity'],.1)
    def test_invalid_n(self):
        for n in (True,3,5,1002,4.0):
            with self.assertRaises(ValueError): validate(n,.5)
    def test_invalid_q(self):
        for q in (True,-.1,1.1,float('nan'),float('inf'),'0.5'):
            with self.assertRaises(ValueError): validate(4,q)
    def test_invalid_runtime(self):
        for kw in ({'replicates':0},{'replicates':True},{'seed':0},{'seed':1.2}):
            with self.assertRaises(ValueError): run(**kw)


class Parsing(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_output('noise\nINIT,0,2,0,2\nDATA,0,1,2,1\n',4,1),[[1,2,1]])
    def test_malformed(self):
        for text in ('','INIT,0,2,0,2','INIT,0,1,2,1\nDATA,0,1,2,1',
                     'INIT,0,2,0,2\nDATA,0,-1,4,1','INIT,0,2,0,2\nDATA,0,1,2,2',
                     'INIT,0,2,0,2\nDATA,0,1,2,1\nDATA,0,1,2,1',
                     'INIT,0,2,0,2\nDATA,1,1,2,1','INIT,0,2,0,2\nDATA,0,1,3'):
            with self.assertRaises(ValueError): parse_output(text,4,1)
    def test_summary_rejects_bad_counts(self):
        for rows in ([],[[0,4]],[[1.,2,1]],[[0,0,0]],[[True,2,1]]):
            with self.assertRaises(ValueError): summarize(rows,4,.5)
    def test_summary_detects_wrong_model(self):
        with self.assertRaises(AssertionError): summarize([[0,4,0]]*100,4,1)
    def test_label_symmetry(self):
        a=summarize([[1,2,1],[0,2,2],[2,2,0]],4,.5)
        b=summarize([[1,2,1],[2,2,0],[0,2,2]],4,.5)
        self.assertEqual(a,b)


class Simulation(unittest.TestCase):
    def test_opposite_control(self):
        self.assertEqual(run(0,n=4,replicates=32,seed=1),[[0,4,0]]*32)
    def test_same_control(self):
        rows=run(1,n=4,replicates=64,seed=2)
        self.assertTrue(all(row[1]==0 for row in rows))
    def test_seed_replay(self):
        self.assertEqual(run(.3,n=8,replicates=16,seed=3),run(.3,n=8,replicates=16,seed=3))
    def test_all_treatments(self):
        for i,q in enumerate((.1,.5,.9)):
            summarize(run(q,n=100,replicates=256,seed=100+i),100,q)
    def test_eidos_rejects_parameters(self):
        for assignment in ('N=3','Q=-0.1','Q=1.1','REPLICATES=0'):
            p=subprocess.run(['slim','-s','1','-d',assignment,str(HERE/'model.slim')],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(p.returncode,0)
    def test_saved_output(self):
        saved=json.loads((HERE/'sample_summary.json').read_text())
        self.assertEqual(saved['python_version'],'3.14.5')
        self.assertEqual(saved['slim_version'],'4.0.1')
        self.assertEqual(len(saved['treatments']),5)
        self.assertEqual(saved['treatments']['opposite_only']['mean_observed_heterozygosity'],1)

if __name__=='__main__': unittest.main()
