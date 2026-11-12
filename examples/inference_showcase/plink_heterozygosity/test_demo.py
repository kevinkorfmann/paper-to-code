"""Exact independent enumeration and integration checks for PLINK F."""
from fractions import Fraction as Q
import itertools
import json
import math
import subprocess
import sys
import unittest
from . import demo as d

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = d.run()

    def test_balanced(self):
        self.assertEqual(self.result['cases']['balanced_reference']['exact_F'], '0')

    def test_rare(self):
        self.assertEqual(self.result['cases']['rare_ALT_reference']['exact_F'], '-16/9')

    def test_allele_symmetry(self):
        self.assertEqual(self.result['cases']['rare_ALT_reference'], self.result['cases']['rare_REF_reference'])

    def test_heterozygote_dropout(self):
        self.assertEqual(self.result['cases']['mask_heterozygotes']['plink']['F'], 1)

    def test_homozygote_dropout(self):
        self.assertEqual(self.result['cases']['mask_homozygotes']['plink']['F'], -1)

    def test_missing_counts(self):
        for name in ('mask_heterozygotes','mask_homozygotes'):
            self.assertEqual(self.result['cases'][name]['plink']['OBS_CT'], 10)
            self.assertEqual(self.result['cases'][name]['plink']['E(HET)'], 5)

    def test_copies_no_new_information(self):
        a=self.result['cases']['balanced_reference']['plink']
        b=self.result['cases']['copy_all_sites']['plink']
        self.assertEqual(a['F'],b['F'])
        for key in ('OBS_CT','O(HET)','E(HET)','O(HOM)','E(HOM)'):
            self.assertEqual(2*a[key],b[key])

    def test_counts_conserve(self):
        for case in self.result['cases'].values():
            a=case['plink']
            self.assertAlmostEqual(a['O(HET)']+a['O(HOM)'],a['OBS_CT'])
            self.assertAlmostEqual(a['E(HET)']+a['E(HOM)'],a['OBS_CT'])

    def test_enumerated_HWE_expectation_zero(self):
        # Enumerate two independently drawn alleles, not the production formula.
        for p in (Q(1,10),Q(1,3),Q(1,2),Q(4,5)):
            expected=Q(0)
            for a,b in itertools.product((0,1),repeat=2):
                weight=(p if a else 1-p)*(p if b else 1-p)
                expected+=weight*d.oracle([a+b],[p])['F']
            self.assertEqual(expected,0)

    def test_mixed_frequencies_ratio_of_sums(self):
        gs=[1,0]; ps=[Q(1,2),Q(1,10)]
        actual=d.tool(gs,ps)
        # Expected heterozygosity 1/2 + 9/50 = 17/25.
        self.assertAlmostEqual(actual['F'],float(-Q(8,17)),places=5)
        mean_ratios=(d.oracle([1],[ps[0]])['F']+d.oracle([0],[ps[1]])['F'])/2
        self.assertNotEqual(d.oracle(gs,ps)['F'],mean_ratios)

    def test_tool_parameter_grid(self):
        for p in (.1,.3,.5):
            for k in (0,1,3,6):
                gs=[1]*k+[0]*(6-k)
                actual=d.tool(gs,[p]*6)
                self.assertAlmostEqual(actual['F'],float(d.oracle(gs,[p]*6)['F']),delta=5e-5)

    def test_marker_permutation(self):
        gs=[0,1,None,2,1];ps=[.1,.2,.3,.4,.5]
        self.assertEqual(d.tool(gs,ps),d.tool(gs[::-1],ps[::-1]))

    def test_genotype_allele_flip(self):
        gs=[0,1,None,2,1];ps=[.1,.2,.3,.4,.5]
        a=d.tool(gs,ps);b=d.tool([None if g is None else 2-g for g in gs],[1-p for p in ps])
        self.assertEqual(a,b)

    def test_missing_equals_remove(self):
        self.assertEqual(d.tool([0,None,1],[.1,.2,.3]),d.tool([0,1],[.1,.3]))

    def test_invalid_calls(self):
        for bad in ([],[None],[True],[3],[-1],[1.0]):
            with self.assertRaises(ValueError): d.inputs(bad,[.5]*len(bad))

    def test_invalid_frequencies(self):
        for p in (0,1,-.1,1.1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): d.inputs([0],[p])

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError): d.inputs([0,1],[.5])

    def test_vcf_and_frequency_alignment(self):
        v,f=d.inputs([0,None,2],[.1,.2,.3])
        self.assertEqual(v.splitlines()[-2].split()[-1],'./.')
        self.assertEqual([x.split()[2] for x in v.splitlines()[4:]],[x.split()[1] for x in f.splitlines()[1:]])

    def test_fixture(self):
        fixture=json.loads((d.HERE/'synthetic_input.json').read_text())
        self.assertEqual(fixture['ALT_dosages'],[0,1,1,2]*5)

    def test_saved_output(self):
        self.assertEqual(self.result,json.loads((d.HERE/'sample_summary.json').read_text()))

    def test_cli(self):
        p=subprocess.run([sys.executable,'-m',d.__name__,'--check-sample'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout),self.result)

if __name__ == '__main__': unittest.main()
