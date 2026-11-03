"""Independent rational/enumeration oracles plus tests of the actual ANGSD CLI."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import tempfile
import unittest

from examples.inference_showcase.angsd_frequency import model
from examples.inference_showcase.angsd_frequency.demo import run_demo
from examples.inference_showcase.angsd_frequency.install import BINARY, check_version, digest, RELEASES

HERE = Path(__file__).parent


def enumerate_likelihood(rows, p):
    """Independent latent-genotype enumeration, using exact rational arithmetic."""
    weights = ((1-p)**2, 2*p*(1-p), p*p)
    total = F(0)
    for genotypes in product(range(3), repeat=len(rows)):
        term = F(1)
        for row, genotype in zip(rows, genotypes):
            term *= row[genotype] * weights[genotype]
        total += term
    return total


class LikelihoodTests(unittest.TestCase):
    def test_exact_closed_form(self):
        self.assertAlmostEqual(model.repeated_heterozygote_mle(.8, .2), .2)
        self.assertEqual(model.repeated_heterozygote_mle(.2, .2), .5)

    def test_rational_derivative_and_global_maximum(self):
        # q(p)=a+2(1-a)p+(a+b-2)p^2 has negative second derivative.
        for a, b in [(F(4,5), F(1,5)), (F(1,5), F(1,5)), (F(1,5), F(4,5))]:
            p = (1-a)/(2-a-b)
            self.assertEqual(2*(1-a)+2*(a+b-2)*p, 0)
            self.assertLess(2*(a+b-2), 0)
            self.assertAlmostEqual(model.repeated_heterozygote_mle(float(a),float(b)), float(p))

    def test_independent_latent_genotype_enumeration(self):
        rows = [(F(4,5), F(1), F(1,5)), (F(1,10), F(3,5), F(1)),
                (F(1), F(1,2), F(1,10)), (F(1,2), F(1,2), F(1,2))]
        for n in range(1,5):
            for p in [F(0), F(1,10), F(1,5), F(1,2), F(4,5), F(1)]:
                expected = enumerate_likelihood(rows[:n], p)
                actual = model.log_likelihood([[float(x) for x in row] for row in rows[:n]], float(p))
                self.assertAlmostEqual(actual, math.log(float(expected)), places=13)

    def test_eighty_digit_log_likelihood_gap(self):
        with localcontext() as context:
            context.prec = 80
            # q(.2)=.84, q(.5)=.75; all 20 rows are identical.
            expected = 20 * (Decimal(28)/Decimal(25)).ln()
        rows = [(0.8,1,0.2)]*20
        actual = model.log_likelihood(rows,.2)-model.log_likelihood(rows,.5)
        self.assertAlmostEqual(actual,float(expected),places=13)

    def test_likelihood_not_posterior(self):
        rows=[(.8,1,.2)]*20
        self.assertAlmostEqual(model.normalized_likelihood_dosage(rows),.35)
        self.assertNotAlmostEqual(model.normalized_likelihood_dosage(rows),.2)

    def test_hard_calls_do_not_change(self):
        self.assertEqual(model.hard_call_frequency([(.8,1,.2)]*20),.5)
        self.assertEqual(model.hard_call_frequency([(.2,1,.2)]*20),.5)

    def test_scale_changes_log_likelihood_only_by_constant(self):
        rows=[(.8,1,.2),(.2,1,.2)]
        scaled=[tuple(x*c for x in row) for row,c in zip(rows,[.1,7])]
        for p in [0,.1,.5,.9,1]:
            self.assertAlmostEqual(model.log_likelihood(scaled,p)-model.log_likelihood(rows,p),math.log(.7))

    def test_allele_reversal(self):
        rows=[(.8,1,.2),(.2,1,.3)]
        for p in [0,.2,.6,1]:
            self.assertAlmostEqual(model.log_likelihood(rows,p),model.log_likelihood([r[::-1] for r in rows],1-p))

    def test_no_information_flat(self):
        for p in [0,.001,.2,.5,.999,1]:
            self.assertAlmostEqual(model.log_likelihood([(1,1,1)]*20,p),0)

    def test_extra_flat_rows_add_no_information(self):
        rows=[(.8,1,.2)]*10
        for p in [0,.2,.5,1]:
            self.assertAlmostEqual(model.log_likelihood(rows,p),model.log_likelihood(rows+[(1,1,1)]*10,p))

    def test_invalid_likelihoods(self):
        for rows in [[],[(1,2)],[(1,2,3,4)],[(0,1,1)],[(-1,1,1)],[(math.nan,1,1)],[(math.inf,1,1)]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):model.validate(rows)

    def test_invalid_frequency(self):
        for p in [-.1,1.1,math.nan,math.inf]:
            with self.assertRaises(ValueError):model.log_likelihood([(1,2,3)],p)

    def test_invalid_closed_form(self):
        for a,b in [(0,.2),(1,.2),(.2,1),(-1,2),(math.nan,.2)]:
            with self.assertRaises(ValueError):model.repeated_heterozygote_mle(a,b)

    def test_tied_hard_calls_rejected(self):
        with self.assertRaises(ValueError):model.hard_call_frequency([(1,1,1)])

    def test_invalid_sample_count(self):
        for n in [0,1,3,-2,2.5,True]:
            with self.assertRaises(ValueError):model.constructed_sites(n)

    def test_vcf_roundtrip_gl_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'in.vcf'
            model.write_vcf([('site',[(.8,1,.2)])],path)
            fields=path.read_text().splitlines()[-1].split('\t')
            self.assertEqual(fields[3:5],['A','C'])
            self.assertEqual(fields[8],'GL')
            for actual,expected in zip(fields[9].split(','),[.8,1,.2]):
                self.assertAlmostEqual(10**float(actual),expected,places=11)

    def test_vcf_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'in.vcf'
            cases=[[],[('x',[(1,2,3)]),('x',[(1,2,3)])],
                   [('bad name',[(1,2,3)])],
                   [('x',[(1,2,3)]),('y',[(1,2,3)]*2)]]
            for sites in cases:
                with self.assertRaises(ValueError):model.write_vcf(sites,path)


class AngsdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites=model.constructed_sites()
        cls.results=model.run_angsd(cls.sites)

    def test_pinned_version(self):
        self.assertEqual(check_version(),{'angsd':'0.941','htslib':'1.24'})

    def test_installed_binary_checksum(self):
        record=json.loads((BINARY.parents[2]/'build.json').read_text())
        self.assertEqual(record['binary_sha256'],digest(BINARY))
        self.assertEqual(record['archives'],{k:v[1] for k,v in RELEASES.items()})

    def test_symmetric_matches_exact(self):
        self.assertAlmostEqual(self.results['symmetric']['alt_frequency'],.5,delta=6e-6)

    def test_asymmetric_matches_exact_with_stated_tolerance(self):
        self.assertAlmostEqual(self.results['asymmetric']['alt_frequency'],.2,delta=5e-5)

    def test_allele_reversal_not_folded(self):
        self.assertAlmostEqual(self.results['alleles_reversed']['alt_frequency'],.8,delta=5e-5)
        self.assertAlmostEqual(self.results['alleles_reversed']['alt_frequency']+
                               self.results['asymmetric']['alt_frequency'],1,delta=1e-4)

    def test_rescaled_likelihoods_same_inference(self):
        self.assertAlmostEqual(self.results['rescaled']['alt_frequency'],
                               self.results['asymmetric']['alt_frequency'],delta=2e-6)

    def test_uninformative_site_omitted(self):
        self.assertNotIn('all_uninformative',self.results)

    def test_uninformative_subjects_not_counted(self):
        self.assertEqual(self.results['half_uninformative']['informative_diploids'],10)
        self.assertAlmostEqual(self.results['half_uninformative']['alt_frequency'],.2,delta=5e-5)

    def test_boundary_reference(self):
        self.assertEqual(self.results['all_reference']['alt_frequency'],0)

    def test_boundary_alternate(self):
        self.assertEqual(self.results['all_alternate']['alt_frequency'],1)

    def test_all_55_small_genotype_tables(self):
        count=0
        for n in range(1,6):
            sites=[];expected={}
            for n0 in range(n+1):
                for n1 in range(n-n0+1):
                    n2=n-n0-n1
                    name=f'n{n}_ref{n0}_het{n1}'
                    rows=[(1,1e-12,1e-12)]*n0+[(1e-12,1,1e-12)]*n1+[(1e-12,1e-12,1)]*n2
                    sites.append((name,rows));expected[name]=(n1+2*n2)/(2*n)
            results=model.run_angsd(sites)
            self.assertEqual(set(results),set(expected))
            for name,p in expected.items():
                self.assertAlmostEqual(results[name]['alt_frequency'],p,delta=1e-6)
                count+=1
        self.assertEqual(count,55)

    def test_individual_order(self):
        rows=[(.8,1,.2)]*10+[(.2,1,.2)]*10
        actual=model.run_angsd([('forward',rows),('reverse',rows[::-1])])
        self.assertAlmostEqual(actual['forward']['alt_frequency'],actual['reverse']['alt_frequency'],delta=2e-6)

    def test_repeatable(self):
        self.assertEqual(model.run_angsd(self.sites),self.results)

    def test_checked_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'in.vcf'
            model.write_vcf(self.sites,path)
            self.assertEqual(path.read_bytes(),(HERE/'synthetic_input.vcf').read_bytes())

    def test_checked_output(self):
        self.assertEqual(run_demo(),json.loads((HERE/'sample_output.json').read_text()))


if __name__=='__main__':
    unittest.main()
