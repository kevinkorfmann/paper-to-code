"""Analytical references are independent of the implementation's formula."""
from fractions import Fraction as F
from itertools import product
import json
import math
import subprocess
import unittest
from unittest.mock import patch
from examples.slimulations.epistasis_ld import demo


def enumerated_gametes(s, r):
    haplotypes = list(product((0,1), repeat=2))
    masses = {h:F(0) for h in haplotypes}
    total = F(0)
    for first, second in product(haplotypes, repeat=2):
        w = 1+s*(first[0]+second[0]-1)*(first[1]+second[1]-1)
        total += w
        for left, right in ((first,second),(second,first)):
            masses[left] += w*(1-r)/2
            masses[(left[0],right[1])] += w*r/2
    return [masses[h]/total for h in haplotypes]


def exhaustive_moments(q, copies):
    mean = F(0)
    second = F(0)
    # Enumerate ordered gamete outcomes, not the falling-factorial formula.
    for outcome in product(range(4), repeat=copies):
        probability = math.prod(q[h] for h in outcome)
        counts = [outcome.count(h) for h in range(4)]
        ld = F(counts[0]*counts[3]-counts[1]*counts[2], copies**2)
        mean += probability*ld
        second += probability*ld*ld
    return float(mean), float(second-mean*mean)


def valid_output():
    return 'INIT,0,'+','.join(['1']*16)+'\nDATA,0,8,8,8,8\n'


class AnalyticalTests(unittest.TestCase):
    def test_independent_meiosis_enumeration(self):
        for s in [F(-1),F(-4,5),F(0),F(1,3),F(4,5),F(1)]:
            for r in [F(0),F(1,7),F(1,2)]:
                expected = enumerated_gametes(s,r)
                actual = demo.haplotype_probabilities(float(s),float(r))
                for p,q in zip(actual,expected): self.assertAlmostEqual(p,float(q))

    def test_recombination_invariance_is_first_generation(self):
        for r in [0,.1,.25,.5]:
            self.assertEqual(demo.haplotype_probabilities(.8,r),(.3,.2,.2,.3))

    def test_marginal_frequencies_remain_half(self):
        for s in [-1,-.8,0,.8,1]:
            q=demo.haplotype_probabilities(s,.5)
            self.assertEqual(q[2]+q[3],.5)
            self.assertEqual(q[1]+q[3],.5)

    def test_neutral_control(self):
        self.assertEqual(demo.haplotype_probabilities(0,.5),(.25,)*4)

    def test_negative_interaction_reverses_ld(self):
        positive=demo.haplotype_probabilities(.8,.5)
        negative=demo.haplotype_probabilities(-.8,.5)
        self.assertEqual(negative,(positive[1],positive[0],positive[3],positive[2]))

    def test_exhaustive_sampling_moments(self):
        for q in [(F(1,4),)*4,(F(3,10),F(1,5),F(1,5),F(3,10)),(F(1,10),F(2,10),F(3,10),F(4,10))]:
            for copies in range(2,5):
                mean,var=exhaustive_moments(q,copies)
                actual=demo.ld_moments([float(x) for x in q],copies)
                self.assertAlmostEqual(actual['mean_sample_ld'],mean,places=14)
                self.assertAlmostEqual(actual['variance_sample_ld'],var,places=14)

    def test_independent_balanced_variance_formula(self):
        for m in [2,3,4,32,512,1000000]:
            for s in [-1,-.8,0,.8,1]:
                d=F(str(s))/16
                expected=F(m-1,16*m**3)*(m-16*(m-2)*d*d)
                actual=demo.ld_moments(demo.haplotype_probabilities(s,.5),m)
                self.assertAlmostEqual(actual['variance_sample_ld'],float(expected),places=14)

    def test_finite_sample_bias(self):
        result=demo.ld_moments([.3,.2,.2,.3],512)
        self.assertEqual(result['population_ld'],.05)
        self.assertEqual(result['mean_sample_ld'],.04990234375)

    def test_fixed_haplotype_zero_variance(self):
        for index in range(4):
            q=[0]*4
            q[index]=1
            self.assertEqual(demo.ld_moments(q,10)['variance_sample_ld'],0)

    def test_invalid_parameters(self):
        for kw in [{'s':True},{'s':1.01},{'s':float('nan')},{'r':-.1},{'r':.51},{'r':math.inf},{'n':17},{'n':True},{'replicates':0},{'seed':0}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError): demo.run(**kw)

    def test_invalid_moment_inputs(self):
        for q,m in [([],2),([.1]*4,2),([0,0,-1,2],2),([0,0,0,math.nan],2),([.25]*4,1),([.25]*4,True)]:
            with self.assertRaises(ValueError): demo.ld_moments(q,m)

    def test_parse_valid_output(self):
        self.assertEqual(demo.parse_output('// SLiM header\n'+valid_output(),16,1),[(8,8,8,8)])

    def test_parse_missing_or_duplicate_rows(self):
        good=valid_output()
        for bad in ['',good.splitlines()[0],good.splitlines()[1],good+good,good+'DATA,0,8,8,8,8\n']:
            with self.assertRaises(ValueError): demo.parse_output(bad,16,1)

    def test_parse_invalid_counts_and_ids(self):
        good=valid_output()
        for bad in [good.replace('DATA,0','DATA,1'),good.replace('INIT,0','INIT,-1'),good.replace('8,8,8,8','8,8,8,7'),good.replace('8,8,8,8','-1,9,12,12'),good.replace('INIT,0,1','INIT,0,2'),good.replace('8,8,8,8','1.5,2.5,3,25'),good.replace('8,8,8,8','8,8,8')]:
            with self.assertRaises(ValueError): demo.parse_output(bad,16,1)

    def test_distribution_diagnostic_rejects_wrong_output(self):
        with self.assertRaises(AssertionError): demo.summarize_rows([(512,0,0,0)]*512,.8,.5,256)

    def test_version_failure_prevents_execution(self):
        with patch.object(demo,'check_slim_version',side_effect=ValueError('wrong version')), patch.object(demo.subprocess,'run') as execute:
            with self.assertRaises(ValueError): demo.run()
            execute.assert_not_called()

    def test_subprocess_failure_is_not_replaced(self):
        with patch.object(demo,'check_slim_version'), patch.object(demo.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['slim'])):
            with self.assertRaises(subprocess.CalledProcessError): demo.run()


class IntegrationTests(unittest.TestCase):
    def test_checked_output(self):
        self.assertEqual(demo.summary(),json.loads((demo.HERE/'sample_summary.json').read_text()))

    def test_second_seed_small_population(self):
        for s,r in demo.TREATMENTS.values():
            rows=demo.run(s,r,n=16,replicates=1024,seed=1944115)
            demo.summarize_rows(rows,s,r,16)

    def test_intermediate_recombination(self):
        rows=demo.run(.8,.17,n=32,replicates=512,seed=501191)
        demo.summarize_rows(rows,.8,.17,32)

    def test_zero_fitness_boundary(self):
        for s in [-1,1]:
            rows=demo.run(s,.5,n=32,replicates=512,seed=771025)
            demo.summarize_rows(rows,s,.5,32)

    def test_replay(self):
        first=demo.run(replicates=3)
        self.assertEqual(first,demo.run(replicates=3))

    def test_direct_script_rejects_invalid_parameters(self):
        for argument in ['S=1.1','R=0.6','N=17','REPLICATES=0']:
            result=subprocess.run(['slim','-d',argument,str(demo.HERE/'model.slim')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

    def test_cli_check(self):
        import sys
        result=subprocess.run([sys.executable,'-m','examples.slimulations.epistasis_ld.demo','--check-sample'],capture_output=True,text=True,check=True)
        self.assertIn('reproduced exactly',result.stdout)


if __name__ == '__main__': unittest.main()
