import json
import math
import subprocess
import unittest
from examples.slimulations.sexual_antagonism import demo


class AnalyticalTests(unittest.TestCase):
    def test_neutral(self):
        self.assertEqual(demo.oracle('neutral')['genotype_probabilities_aa_Aa_AA'], [.25,.5,.25])

    def test_antagonism(self):
        result = demo.oracle('antagonistic')
        self.assertEqual(result['male_gamete_A'], .25)
        self.assertEqual(result['female_gamete_A'], .75)
        self.assertEqual(result['genotype_probabilities_aa_Aa_AA'], [.1875,.625,.1875])
        self.assertEqual(result['heterozygote_excess'], .125)

    def test_concordance(self):
        self.assertEqual(demo.oracle('concordant')['genotype_probabilities_aa_Aa_AA'], [.0625,.375,.5625])

    def test_scale_invariance_and_complement(self):
        self.assertEqual(demo.gamete_probability([1e308]*3), .5)
        for w in [(1,2,7),(.1,.3,.9),(1,1,1)]:
            self.assertAlmostEqual(demo.gamete_probability(w), demo.gamete_probability([10*x for x in w]))
            self.assertAlmostEqual(demo.gamete_probability(w)+demo.gamete_probability(w[::-1]),1)

    def test_general_excess_identity(self):
        for pm in [0,.2,.5,.9,1]:
            for pf in [0,.1,.5,.8,1]:
                p=(pm+pf)/2
                self.assertAlmostEqual(pm*(1-pf)+(1-pm)*pf-2*p*(1-p), (pm-pf)**2/2)

    def test_invalid_weights(self):
        for w in [[], [1,2], [1,2,0], [1,2,-1], [1,2,math.nan], [1,2,True]]:
            with self.assertRaises(ValueError): demo.gamete_probability(w)

    def test_invalid_inputs(self):
        for kw in [{'mode':'x'},{'n':9},{'n':True},{'replicates':0},{'seed':0}]:
            with self.assertRaises(ValueError): demo.run(**({'mode':'neutral'}|kw))

    def test_parser_corruption(self):
        good='INIT,0,M,1,2,1\nINIT,0,F,1,2,1\nDATA,0,2,4,2\n'
        self.assertEqual(demo.parse_output(good,8,1), [(2,4,2)])
        for bad in ['', good+ 'DATA,0,2,4,2\n', good.replace('1,2,1','0,3,1'),good.replace('2,4,2','2,-1,7'),good.replace('DATA,0','DATA,1'),good.replace('M','X')]:
            with self.assertRaises(ValueError): demo.parse_output(bad,8,1)


class IntegrationTests(unittest.TestCase):
    def test_checked_output_and_exact_probabilities(self):
        self.assertEqual(demo.summary(), json.loads((demo.HERE/'sample_summary.json').read_text()))

    def test_second_seed_small_population(self):
        for mode in demo.MODES:
            data = demo.run(mode,n=8,replicates=1024,seed=190923)
            probabilities = demo.oracle(mode)['genotype_probabilities_aa_Aa_AA']
            for g, p in enumerate(probabilities):
                observed=sum(row[g] for row in data)/(8*len(data))
                self.assertLess(abs(observed-p),6*math.sqrt(p*(1-p)/(8*len(data))))

    def test_reproducibility(self):
        self.assertEqual(demo.run('antagonistic',replicates=3),demo.run('antagonistic',replicates=3))

    def test_direct_script_rejects_bad_mode(self):
        result=subprocess.run(['slim','-d','MODE=4',str(demo.HERE/'model.slim')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)


if __name__ == '__main__': unittest.main()
