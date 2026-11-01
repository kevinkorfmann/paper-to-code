"""Real external-tool inference, exhaustive HMM controls and synthetic-data sensitivity."""

from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import itertools
import json
import math
import platform
import subprocess
import sys
import unittest

from examples.inference_showcase.bcftools_roh import demo,install,reference


def exhaustive(pls,frequencies,switch=.02):
    """Enumerate every hidden state sequence; no recurrence or production helpers."""
    with localcontext() as context:
        context.prec=70
        emissions=[]
        for row,f in zip(pls,frequencies):
            f=D(str(f));likelihood=[D(10)**(-D(p)/10) for p in row]
            hw=sum(((1-f) if a==0 else f)*((1-f) if b==0 else f)*likelihood[a+b]
                   for a,b in itertools.product((0,1),repeat=2))
            az=(1-f)*likelihood[0]+f*likelihood[2]
            emissions.append((hw,az))
        transition=D(str(switch));paths=[]
        for path in itertools.product((0,1),repeat=len(pls)):
            mass=D('.5')*math.prod(emissions[i][s] for i,s in enumerate(path))
            mass*=math.prod(transition if a!=b else 1-transition for a,b in zip(path,path[1:]))
            paths.append((path,mass))
        normalizer=sum(mass for _,mass in paths)
        posterior=[sum(mass for path,mass in paths if path[i]==1)/normalizer for i in range(len(pls))]
        return list(max(paths,key=lambda pair:pair[1])[0]),list(map(float,posterior))


class RohTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(demo.FIXTURE.read_text())
        cls.report=demo.report()
        cls.tiny=[[30,0,30],[0,30,60],[0,30,60],[30,0,30]]

    def test_exact_emissions_from_independent_allele_enumeration(self):
        for f in (F(1,2),F(1,10),F(9,10)):
            for row in ([0,20,40],[20,0,20],[40,20,0]):
                likelihood=[F(1,10**(p//10)) for p in row]
                hw=sum((f if a else 1-f)*(f if b else 1-f)*likelihood[a+b]
                       for a,b in itertools.product((0,1),repeat=2))
                az=(1-f)*likelihood[0]+f*likelihood[2]
                got=reference.emissions(row,float(f))
                self.assertAlmostEqual(got[0],float(hw),places=14)
                self.assertAlmostEqual(got[1],float(az),places=14)

    def test_reference_matches_exhaustive_decimal_paths_and_marginals(self):
        for rows,frequencies in [(self.tiny,[.5]*4),
                                 ([[0,10,20],[20,10,0],[10,0,10],[0,10,20],[0,20,40],[10,0,10]],[.2,.7,.3,.4,.5,.6])]:
            path,posterior=exhaustive(rows,frequencies)
            actual=reference.infer(rows,frequencies)
            self.assertEqual(actual['viterbi'],path)
            for a,b in zip(actual['posterior_AZ'],posterior):self.assertAlmostEqual(a,b,places=13)

    def test_actual_tool_matches_exhaustive_tiny_oracle(self):
        path,_=exhaustive(self.tiny,[.5]*4)
        actual=demo.run_tool({'tiny':self.tiny},[.5]*4)['tiny']
        self.assertEqual(actual['states'],path)

    def test_all_1164_reported_states_match_independent_viterbi(self):
        rows=[*self.report['main_constructed_PLs'].values(),*self.report['weaker_PLs_same_best_genotypes'].values(),
              *self.report['frequency_sensitivity_same_PLs']]
        self.assertEqual(sum(x['markers'] for x in rows)+4,1164)
        self.assertTrue(all(x['independent_viterbi_disagreements']==0 for x in rows))

    def test_tool_detects_tract_with_one_marker_boundary_error(self):
        row=self.report['main_constructed_PLs']['tract']
        self.assertEqual(row['designed_AZ_markers'],60)
        self.assertEqual(row['inferred_AZ_markers'],61)
        self.assertEqual(row['state_disagreements_with_design'],1)
        self.assertEqual((row['regions'][0]['start'],row['regions'][0]['end']),(60,120))

    def test_outbred_and_fully_autozygous_controls(self):
        a=self.report['main_constructed_PLs']
        self.assertEqual(a['outbred']['inferred_AZ_markers'],0)
        self.assertEqual(a['all_autozygous']['inferred_AZ_markers'],180)

    def test_likelihood_weakening_preserves_best_genotypes(self):
        for sample in self.data['samples'].values():
            for g,row in zip(sample['genotypes'],sample['PL']):
                self.assertEqual(min(range(3),key=lambda i:row[i]),g)
                weak=[x//10 for x in row]
                self.assertEqual(min(range(3),key=lambda i:weak[i]),g)
        self.assertEqual(self.report['weaker_PLs_same_best_genotypes']['tract']['inferred_AZ_markers'],0)

    def test_frequency_sensitivity_holds_likelihoods_fixed(self):
        a,b=self.report['frequency_sensitivity_same_PLs']
        self.assertEqual(a['inferred_AZ_markers'],24);self.assertEqual(b['inferred_AZ_markers'],0)
        self.assertGreater(a['homref_emission_ratio_AZ_over_HW'],1.99)
        self.assertLess(b['homref_emission_ratio_AZ_over_HW'],1.02)

    def test_quality_diagnostic_is_explicit_not_mislabeled_posterior(self):
        q=self.report['quality_diagnostic']
        self.assertEqual(q['states'],[0]*4)
        self.assertEqual(q['tool_reported_quality'],[3.,35.2,25.4,25.4])
        _,p=exhaustive(self.tiny,[.5]*4)
        for actual,expected in zip(q['independent_same_site_posterior_AZ'],p):self.assertAlmostEqual(actual,expected,places=13)
        self.assertGreater(abs(q['tool_reported_quality'][0]-q['independent_same_site_phred_error'][0]),30)
        self.assertAlmostEqual(q['independent_same_site_phred_error'][0],q['independent_same_site_phred_error'][-1],places=12)

    def test_quality_offset_observation_at_printed_precision(self):
        q=self.report['quality_diagnostic']
        self.assertEqual(q['tool_reported_quality'][1:],
                         [round(x,1) for x in q['independent_same_site_phred_error'][:-1]])

    def test_fixture_is_exactly_seed_reproducible(self):
        self.assertEqual(self.data,demo.generate_fixture())
        self.assertEqual(self.data['seed'],20261101)

    def test_fixture_state_construction_and_diploid_genotypes(self):
        for name,item in self.data['samples'].items():
            self.assertEqual(len(item['PL']),180)
            for state,g in zip(item['truth_states'],item['genotypes']):
                self.assertIn(g,(0,1,2))
                if state:self.assertIn(g,(0,2))
        self.assertEqual([i+1 for i,s in enumerate(self.data['samples']['tract']['truth_states']) if s],list(range(61,121)))

    def test_inference_input_omits_truth_and_genotype_calls(self):
        text=demo.vcf_text({n:v['PL'] for n,v in self.data['samples'].items()},self.data['ALT_frequencies'])
        self.assertNotIn('truth',text);self.assertNotIn('FORMAT=<ID=GT',text)
        records=[line.split('\t') for line in text.splitlines() if not line.startswith('#')]
        self.assertEqual(len(records),180);self.assertTrue(all(row[8]=='PL' for row in records))

    def test_likelihood_scale_invariance_actual_tool_and_reference(self):
        shifted=[[x+10 for x in row] for row in self.tiny]
        a=demo.run_tool({'tiny':self.tiny},[.5]*4)['tiny']
        b=demo.run_tool({'tiny':shifted},[.5]*4)['tiny']
        self.assertEqual(a,b)
        for x,y in zip(reference.infer(self.tiny,[.5]*4)['posterior_AZ'],reference.infer(shifted,[.5]*4)['posterior_AZ']):
            self.assertAlmostEqual(x,y,places=14)

    def test_allele_flip_invariance_actual_tool(self):
        rows=[[0,20,40],[20,0,20],[40,20,0],[0,10,20],[0,10,20]];f=[.2,.4,.8,.3,.6]
        a=demo.run_tool({'flip':rows},f)['flip']
        b=demo.run_tool({'flip':[list(reversed(r)) for r in rows]},[1-x for x in f])['flip']
        self.assertEqual(a,b)

    def test_two_state_hmm_has_valid_marginals_and_reversal_symmetry(self):
        rows=self.data['samples']['tract']['PL'];f=self.data['ALT_frequencies']
        result=reference.infer(rows,f);rev=reference.infer(list(reversed(rows)),list(reversed(f)))
        self.assertTrue(all(0<=p<=1 for p in result['posterior_AZ']))
        for a,b in zip(result['posterior_AZ'],reversed(rev['posterior_AZ'])):self.assertAlmostEqual(a,b,places=13)

    def test_flat_likelihood_reference_preserves_symmetric_prior(self):
        result=reference.infer([[0,0,0]]*10,[.2]*10)
        for p in result['posterior_AZ']:self.assertAlmostEqual(p,.5,places=14)
        with self.assertRaises(ValueError):demo.vcf_text({'flat':[[0,0,0]]},[.2])

    def test_single_site_emission_bayes_identity(self):
        for f in (.1,.5,.9):
            hw,az=reference.emissions([0,20,40],f)
            p=reference.infer([[0,20,40]],[f])['posterior_AZ'][0]
            self.assertAlmostEqual(p,az/(az+hw),places=14)

    def test_version_pins_and_local_binary(self):
        self.assertEqual(install.check_version(install.BINARY),{'bcftools':'1.23.1','htslib':'1.24'})
        self.assertEqual(platform.python_version(),'3.14.5')
        stamp=json.loads((install.CACHE/'build.json').read_text())
        self.assertEqual(stamp['binary_sha256'],install.digest(install.BINARY))
        self.assertEqual(stamp['archives'],{k:v[1] for k,v in install.RELEASES.items()})

    def test_invalid_emissions_and_transition_inputs(self):
        for row in ([0,1],[0,1,201],[0,-1,2],[True,0,3],[0,1.,2]):
            with self.assertRaises(ValueError):reference.emissions(row,.5)
        for f in (0,1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):reference.emissions([0,10,20],f)
        for switch in (0,.5,True,float('nan')):
            with self.assertRaises(ValueError):reference.infer(self.tiny,[.5]*4,switch)
        for rows,f in [([] ,[]),(self.tiny,[.5])]:
            with self.assertRaises(ValueError):reference.infer(rows,f)

    def test_invalid_synthetic_input_format(self):
        for rows,f in [({},[.5]),({'bad name':[[0,10,20]]},[.5]),({'tiny':self.tiny},[.5]),({'tiny':[[0,10,20]]},[])]:
            with self.assertRaises(ValueError):demo.vcf_text(rows,f)

    def test_parser_rejects_missing_duplicate_wrong_version_and_regions(self):
        text='# This file was produced by: bcftools roh(1.23.1+htslib-1.24)\nST\ttiny\t1\t1\t0\t3.0\n'
        self.assertEqual(demo.parse_output(text,['tiny'],1)['tiny']['states'],[0])
        for bad in ['',text.replace('1.23.1','1.22'),text+text.splitlines()[-1]+'\n',
                    text.replace('\t0\t3.0','\t2\t3.0'),text.replace('\t3.0','\tnan'),
                    text+'RG\ttiny\t1\t1\t1\t1\t1\t3.0\n']:
            with self.assertRaises(ValueError):demo.parse_output(bad,['tiny'],1)

    def test_tool_reproducibility(self):
        args=({'tiny':self.tiny},[.5]*4)
        self.assertEqual(demo.run_tool(*args),demo.run_tool(*args))

    def test_checked_sample_output(self):
        self.assertEqual(demo.render(self.report),demo.SAMPLE.read_text())

    def test_cli_sample_and_unknown_argument(self):
        command=[sys.executable,'-m','examples.inference_showcase.bcftools_roh.demo']
        result=subprocess.run(command+['--check-sample'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout,demo.SAMPLE.read_text())
        bad=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(bad.returncode,2)


if __name__=='__main__':unittest.main()
