"""Independent rational and real-SLiM checks for habitat fitness normalization."""
from fractions import Fraction as F
import json
import math
import statistics
import subprocess
import sys
import unittest
from unittest.mock import patch
from examples.slimulations.soft_selection import demo as d


def enumerated(mode, start, scale=1):
    """Enumerate each founder and both homologs with exact rational mass."""
    habitats = []
    for h in (0,1):
        individuals = []
        for k,count in enumerate(d.STARTS[start]):
            alleles = ((0,0), (0,1), (1,1))[k]
            w = F(1) if mode == 'neutral' else F((1+k)*scale if h == 0 else 3-k)
            individuals += [(alleles,w)]*count
        mass = sum(w for _,w in individuals)
        if mode == 'soft':
            individuals = [(a,w/mass) for a,w in individuals]
        habitats.append(individuals)
    all_parents = habitats[0]+habitats[1]
    total = sum(w for _,w in all_parents)
    a = sum(w*sum(alleles)/2 for alleles,w in all_parents)/total
    h0 = sum(w for _,w in habitats[0])/total
    # Enumerate ordered gamete alleles, independently of the implementation.
    genotypes = [F(0),F(0),F(0)]
    for first,p1 in [(0,1-a),(1,a)]:
        for second,p2 in [(0,1-a),(1,a)]:
            genotypes[first+second] += p1*p2
    return a,h0,genotypes


class AnalyticalTests(unittest.TestCase):
    def test_exact_rare_fractions(self):
        q,h0,_ = enumerated('soft',1)
        self.assertEqual(q,F(103,455))
        self.assertEqual(h0,F(1,2))
        self.assertAlmostEqual(d.oracle()['allele_frequency'],float(q))

    def test_independent_founder_and_mendelian_enumeration(self):
        for mode in d.MODES:
            for start in d.STARTS:
                for scale in (1,2,3):
                    with self.subTest(mode=mode,start=start,scale=scale):
                        q,h0,gen = enumerated(mode,start,scale)
                        obs = d.oracle(mode,start,scale)
                        self.assertAlmostEqual(obs['allele_frequency'],float(q))
                        self.assertAlmostEqual(obs['habitat0_parent_probability'],float(h0))
                        for x,y in zip(obs['genotype_probabilities_aa_Aa_AA'],gen):
                            self.assertAlmostEqual(x,float(y))

    def test_hard_cancels_at_equal_scale(self):
        for start,p in [(1,.2),(2,.5),(3,.8)]:
            self.assertAlmostEqual(d.oracle('hard',start)['allele_frequency'],p)

    def test_soft_rare_and_common_directions(self):
        self.assertGreater(d.oracle('soft',1)['allele_frequency'],.2)
        self.assertLess(d.oracle('soft',3)['allele_frequency'],.8)

    def test_soft_allele_exchange_symmetry(self):
        self.assertAlmostEqual(d.oracle('soft',1)['allele_frequency']+d.oracle('soft',3)['allele_frequency'],1)

    def test_balanced_zero_shift(self):
        for mode in d.MODES:
            self.assertAlmostEqual(d.oracle(mode,2)['allele_frequency'],.5)

    def test_habitat_scaling_soft_invariance(self):
        for scale in (.125,.5,2,10):
            self.assertAlmostEqual(d.oracle('soft',1,scale)['allele_frequency'],float(F(103,455)))
            self.assertEqual(d.oracle('soft',1,scale)['habitat0_parent_probability'],.5)

    def test_hard_scaling_changes_global_selection(self):
        self.assertAlmostEqual(d.oracle('hard',1,2)['allele_frequency'],float(F(31,135)))
        self.assertAlmostEqual(d.oracle('hard',1,2)['habitat0_parent_probability'],float(F(14,27)))
        self.assertNotEqual(d.oracle('hard',2,2)['allele_frequency'],.5)

    def test_neutral_ignores_scale(self):
        for scale in (.1,1,2,10):
            self.assertEqual(d.oracle('neutral',1,scale)['allele_frequency'],.2)

    def test_invalid_oracle_inputs(self):
        for params in [('bad',1,1),('soft',0,1),('soft',True,1),('soft',1,0),('soft',1,-1),('soft',1,0.001),('soft',1,11),('soft',1,math.nan),('soft',1,math.inf),('soft',1,True)]:
            with self.subTest(params=params),self.assertRaises(ValueError):
                d.oracle(*params)

    def test_runner_validation_before_subprocess(self):
        with patch.object(d,'check_slim_version') as version:
            for args in [{'n':201},{'n':True},{'n':200.0},{'replicates':0},{'replicates':2.5},{'seed':0},{'seed':True}]:
                with self.subTest(args=args),self.assertRaises(ValueError):
                    d.run(**args)
            version.assert_not_called()

    def test_single_replicate_standard_error_none(self):
        out=d.summarize([(120,70,10,200)],200,d.oracle())
        self.assertIsNone(out['empirical_standard_error_of_mean_frequency'])

    def test_summary_rejects_false_prediction(self):
        with self.assertRaises(AssertionError):
            d.summarize([(0,0,200,400)]*512,200,d.oracle())


class ParserTests(unittest.TestCase):
    TEXT='INIT,0,0,64,32,4\nINIT,0,1,64,32,4\nWEIGHT,0,0.5\nDATA,0,120,70,10,200\n'
    def parse(self,text):
        return d.parse_output(text,200,1,'soft',1,1)

    def test_good_and_ignored_noise(self):
        self.assertEqual(self.parse('SLiM version 4.0.1\n'+self.TEXT),[(120,70,10,200)])

    def test_missing_any_required_row(self):
        lines=self.TEXT.splitlines(True)
        for i in range(len(lines)):
            with self.subTest(i=i),self.assertRaises(ValueError):
                self.parse(''.join(lines[:i]+lines[i+1:]))

    def test_duplicate_each_row_type(self):
        for row in self.TEXT.splitlines(True):
            with self.subTest(row=row),self.assertRaises(ValueError):
                self.parse(self.TEXT+row)

    def test_malformed_or_inconsistent_rows(self):
        for old,new in [('64,32,4','63,33,4'),('INIT,0,1','INIT,0,2'),('WEIGHT,0,0.5','WEIGHT,0,nan'),('WEIGHT,0,0.5','WEIGHT,0,0.6'),('120,70,10,200','120,70,11,200'),('120,70,10,200','120,70,10,401'),('120,70,10,200','-1,191,10,200'),('DATA,0,','DATA,1,'),('120,70,10,200','120,70,10'),('INIT,0,0,64,32,4','INIT,0,0,64,32')]:
            with self.subTest(new=new),self.assertRaises(ValueError):
                self.parse(self.TEXT.replace(old,new))


class SLiMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.check_slim_version('slim')
        cls.report=d.summary()

    def test_all_eight_treatments_and_expected_output(self):
        self.assertEqual(set(self.report['treatments']),set(d.TREATMENTS))
        self.assertEqual(self.report,json.loads((d.HERE/'sample_summary.json').read_text()))

    def test_scaled_soft_same_seed_exact_counts(self):
        self.assertEqual(d.run(scale=1,replicates=64),d.run(scale=2,replicates=64))

    def test_repeat_seed_exact_counts(self):
        self.assertEqual(d.run(replicates=8,seed=88),d.run(replicates=8,seed=88))

    def test_different_seed_changes_counts(self):
        self.assertNotEqual(d.run(replicates=8,seed=88),d.run(replicates=8,seed=89))

    def test_binomial_finite_population_variance(self):
        rows=d.run(replicates=1024,seed=7001)
        frequencies=[(r[1]+2*r[2])/400 for r in rows]
        q=d.oracle()['allele_frequency']
        predicted=q*(1-q)/400
        self.assertLess(abs(statistics.variance(frequencies)/predicted-1),.2)

    def test_habitat_contribution_is_not_fixed(self):
        rows=d.run(replicates=32)
        self.assertGreater(len({row[3] for row in rows}),1)
        self.assertTrue(any(row[3] != 200 for row in rows))

    def test_larger_population(self):
        rows=d.run(n=400,replicates=64,seed=7002)
        d.summarize(rows,400,d.oracle())
        self.assertTrue(all(sum(row[:3]) == 400 for row in rows))

    def test_backend_rejects_invalid_inputs(self):
        for arg in ['N=201','REPLICATES=0','MODE=3','START=0','SCALE=0.0','SCALE=11.0','SCALE=0.001']:
            result=subprocess.run(['slim','-s','1','-d',arg,str(d.HERE/'model.slim')],capture_output=True,text=True,timeout=15)
            with self.subTest(arg=arg):
                self.assertNotEqual(result.returncode,0)
                self.assertIn('invalid',result.stderr)

    def test_cli_exact_replay(self):
        result=subprocess.run([sys.executable,'-m','examples.slimulations.soft_selection.demo','--check-sample'],capture_output=True,text=True,check=True,timeout=120)
        self.assertIn('reproduced exactly',result.stdout)


if __name__ == '__main__':
    unittest.main()
