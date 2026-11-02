"""Independent finite-population enumeration, numerical checks and actual SLiM runs."""
from collections import defaultdict
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.bottleneck_recovery import demo


def enumerated_row(k,n,next_n):
    p=F(k,2*n)
    row=[F(0)]*(2*next_n+1)
    for draw in itertools.product((0,1),repeat=2*next_n):
        row[sum(draw)]+=math.prod(p if allele else 1-p for allele in draw)
    return row


class BottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results={name:demo.run_counts(sizes,seed=demo.SEED+i)
                     for i,(name,sizes) in enumerate(demo.SCHEDULES.items())}
        cls.report=demo.build_report(results=cls.results)

    def test_binomial_against_ordered_gamete_enumeration(self):
        for k in range(5):
            exact=enumerated_row(k,2,3)
            got=demo.binomial_row(6,k/4)
            for a,b in zip(got,exact): self.assertAlmostEqual(a,float(b),places=14)

    def test_binomial_against_80_digit_direct_formula(self):
        with localcontext() as context:
            context.prec=80
            for p in ('0.005','0.1','0.5','0.995'):
                got=demo.binomial_row(400,float(p));q=D(p)
                expected=[D(math.comb(400,k))*q**k*(1-q)**(400-k) for k in range(401)]
                for a,b in zip(got,expected):
                    self.assertTrue(math.isclose(a,float(b),rel_tol=2e-12,abs_tol=1e-16))

    def test_transition_mass_mean_and_variance(self):
        for old,new in [(2,3),(5,100),(100,5)]:
            for k,row in enumerate(demo.transition(old,new)):
                p=k/(2*old)
                self.assertAlmostEqual(math.fsum(row),1,places=14)
                mean=math.fsum(j*x for j,x in enumerate(row))
                variance=math.fsum((j-2*new*p)**2*x for j,x in enumerate(row))
                self.assertAlmostEqual(mean,2*new*p,places=12)
                self.assertAlmostEqual(variance,2*new*p*(1-p),places=11)

    def test_tiny_variable_size_path_against_exact_rational_enumeration(self):
        sizes=[2,3,2];dist={1:F(1)};old=2
        got=demo.expected_path(2,sizes,1)
        for t,n in enumerate([2]+sizes):
            if t:
                next_dist=defaultdict(F)
                for k,mass in dist.items():
                    for j,prob in enumerate(enumerated_row(k,old,n)): next_dist[j]+=mass*prob
                dist=next_dist;old=n
            expected={'mean_p':sum(m*F(k,2*n) for k,m in dist.items()),
                      'mean_gene_diversity':sum(m*2*F(k,2*n)*(1-F(k,2*n)) for k,m in dist.items()),
                      'loss_probability':dist.get(0,F(0)),'fixation_probability':dist.get(2*n,F(0))}
            for label,value in expected.items(): self.assertAlmostEqual(got[t][label],float(value),places=14)

    def test_diversity_product_matches_transition_distribution(self):
        for group in self.report['treatments']:
            exact=F(9,50)
            path=demo.expected_path(100,group['offspring_size_schedule'],20)
            for t,row in enumerate(path):
                if t: exact*=1-F(1,2*group['offspring_size_schedule'][t-1])
                self.assertAlmostEqual(row['mean_gene_diversity'],float(exact),places=13)
                self.assertAlmostEqual(row['diversity_product_formula'],float(exact),places=13)

    def test_constant_size_reduction_and_martingale_mean(self):
        for t,row in enumerate(demo.expected_path(100,[100]*40,20)):
            self.assertAlmostEqual(row['mean_gene_diversity'],.18*.995**t,places=13)
        for group in self.report['treatments']:
            for x in group['checkpoints']: self.assertAlmostEqual(x['finite_model']['mean_p'],.1,places=13)

    def test_same_size_multiset_same_diversity_different_loss(self):
        a,b=self.report['treatments'][1:]
        self.assertEqual(sorted(a['offspring_size_schedule']),sorted(b['offspring_size_schedule']))
        x,y=[g['checkpoints'][-1]['finite_model'] for g in (a,b)]
        self.assertAlmostEqual(x['mean_gene_diversity'],y['mean_gene_diversity'],places=13)
        self.assertGreater(y['loss_probability']-x['loss_probability'],.01)

    def test_exact_ensemble_diversity_declines_and_loss_increases(self):
        for sizes in demo.SCHEDULES.values():
            path=demo.expected_path(100,sizes,20)
            for a,b in zip(path,path[1:]):
                self.assertLessEqual(b['mean_gene_diversity'],a['mean_gene_diversity']+1e-14)
                self.assertGreaterEqual(b['loss_probability']+1e-14,a['loss_probability'])
                self.assertAlmostEqual(b['probability_mass'],1,places=13)

    def test_exact_absorbing_states(self):
        for copies in (0,200):
            for x in demo.expected_path(100,[5,100,5,100],copies):
                self.assertEqual(x['mean_gene_diversity'],0)
                self.assertEqual(x['mean_p'],copies/200)
                self.assertEqual(x['loss_probability'],int(copies==0))

    def test_first_bottleneck_loss_has_simple_analytic_oracle(self):
        row=demo.expected_path(100,[5],20)[1]
        self.assertAlmostEqual(row['loss_probability'],.9**10,places=14)
        self.assertAlmostEqual(row['mean_gene_diversity'],.18*.9,places=14)

    def test_realized_sizes_and_all_generations_and_replicates(self):
        for name,rows in self.results.items():
            self.assertEqual(set(rows),set(range(41)))
            self.assertTrue(all(len(x)==1024 for x in rows.values()))
            for t,values in rows.items():
                n=100 if t==0 else demo.SCHEDULES[name][t-1]
                self.assertTrue(all(0<=k<=2*n for k,_ in values))
            self.assertTrue(all(k==20 for k,_ in rows[0]))

    def test_actual_means_match_theory_with_monte_carlo_error(self):
        for group in self.report['treatments']:
            for row in group['checkpoints'][1:]:
                for label,se in [('mean_p','mean_p_monte_carlo_se'),('mean_gene_diversity','gene_diversity_monte_carlo_se')]:
                    self.assertLess(abs(row['observed'][label]-row['finite_model'][label]),5*row['observed'][se])

    def test_actual_absorption_counts_match_finite_distribution(self):
        for group in self.report['treatments']:
            for row in group['checkpoints'][1:]:
                for observed,predicted in [('loss_fraction','loss_probability'),('fixed_replicates','fixation_probability')]:
                    p=row['finite_model'][predicted]
                    actual=row['observed'][observed]
                    if observed=='fixed_replicates': actual/=1024
                    # Numerical sanity envelope, not a confidence claim or preregistered test.
                    self.assertLessEqual(abs(actual-p),5*math.sqrt(p*(1-p)/1024)+1/1024)

    def test_recovery_cannot_restore_lost_alleles_in_actual_replicates(self):
        rows=self.results['early_bottleneck']
        lost=[r for r,(k,_) in enumerate(rows[5]) if k==0]
        self.assertGreater(len(lost),0)
        self.assertTrue(all(rows[t][r][0]==0 for r in lost for t in range(6,41)))
        self.assertEqual(self.report['treatments'][1]['alleles_lost_by_generation_5_still_lost_at_40'],len(lost))

    def test_constant_size_control_retains_more_diversity(self):
        a,b=[g['checkpoints'][-1]['observed'] for g in self.report['treatments'][:2]]
        self.assertGreater(a['mean_gene_diversity']-b['mean_gene_diversity'],
                           5*math.hypot(a['gene_diversity_monte_carlo_se'],b['gene_diversity_monte_carlo_se']))
        self.assertGreater(b['loss_fraction']-a['loss_fraction'],
                           5*math.hypot(a['loss_fraction_monte_carlo_se'],b['loss_fraction_monte_carlo_se']))

    def test_observed_heterozygosity_and_gene_diversity_have_distinct_expectations(self):
        values=[het/5 for _,het in self.results['early_bottleneck'][1]]
        self.assertLess(abs(statistics.mean(values)-.18),5*statistics.stdev(values)/math.sqrt(1024))
        row=self.report['treatments'][1]['checkpoints'][1]
        self.assertAlmostEqual(row['finite_model']['mean_gene_diversity'],.162,places=14)

    def test_allele_label_symmetry_in_finite_model(self):
        a=demo.expected_path(8,[2,7,3],3);b=demo.expected_path(8,[2,7,3],13)
        for x,y in zip(a,b):
            self.assertAlmostEqual(x['mean_gene_diversity'],y['mean_gene_diversity'],places=14)
            self.assertAlmostEqual(x['loss_probability'],y['fixation_probability'],places=14)

    def test_all_actual_absorbed_replicates_remain_absorbed(self):
        for name,sizes in demo.SCHEDULES.items():
            for t in range(40):
                n=100 if t==0 else sizes[t-1]
                for (k,_),(next_k,_) in zip(self.results[name][t],self.results[name][t+1]):
                    if k==0: self.assertEqual(next_k,0)
                    if k==2*n: self.assertEqual(next_k,2*sizes[t])

    def test_exact_seed_repeat(self):
        kwargs=dict(n0=4,initial_copies=1,replicates=3,seed=123)
        self.assertEqual(demo.run_counts([2,5,4],**kwargs),demo.run_counts([2,5,4],**kwargs))

    def test_actual_lost_and_fixed_founder_controls(self):
        for copies in (0,8):
            rows=demo.run_counts([2,5,4],n0=4,initial_copies=copies,replicates=4,seed=124)
            for t,n in enumerate([4,2,5,4]):
                self.assertEqual(rows[t],[(0 if copies==0 else 2*n,0)]*4)

    def test_invalid_python_parameters(self):
        for sizes in ([],[0],[201],[True],[2.5],[2]*101):
            with self.assertRaises(ValueError): demo.run_counts(sizes)
        for kw in ({'n0':1},{'initial_copies':201},{'replicates':0},{'seed':0},{'seed':True}):
            with self.assertRaises(ValueError): demo.run_counts([100],**kw)
        for p in (-.1,1.1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): demo.binomial_row(4,p)

    def test_invalid_eidos_parameters(self):
        for define in ('SIZES=c(1,100)','INITIAL_COPIES=201','REPLICATES=0'):
            result=subprocess.run(['slim','-d',define,str(demo.SCRIPT)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('invalid',result.stderr)

    def test_parser_rejects_corrupt_count_size_version_and_missing_rows(self):
        result=subprocess.run(['slim','-s','123','-d','N0=4','-d','INITIAL_COPIES=1',
                               '-d','REPLICATES=1','-d','SIZES=c(2)',str(demo.SCRIPT)],
                              capture_output=True,text=True,check=True,timeout=10)
        text=result.stdout
        parsed=demo.parse_output(text,4,[2],1,1);self.assertEqual(parsed[0],[(1,1)])
        data=next(x for x in text.splitlines() if x.startswith('DATA,1,'))
        bads=['',text+data+'\n',text.replace(data,''),text.replace('DATA,1,0,2,','DATA,1,0,3,'),
              text.replace('DATA,0,0,4,1,1','DATA,0,0,4,1,0'),text.replace('FIT,1','FIT,2'),
              text.replace('SLiM version 4.0.1','SLiM version 5.0')]
        for bad in bads:
            with self.assertRaises(ValueError): demo.parse_output(bad,4,[2],1,1)

    def test_parser_rejects_false_recovery(self):
        text='Eidos version 3.0.1\nSLiM version 4.0.1\nFIT,1\nINIT,0,4,0,0\nDATA,0,0,4,0,0\nDATA,1,0,2,1,1\n'
        with self.assertRaises(ValueError): demo.parse_output(text,4,[2],0,1)

    def test_versions(self):
        self.assertEqual(platform.python_version(),'3.14.5')
        self.assertEqual(demo.check_slim_version('slim'),'4.0.1')

    def test_single_replicate_uncertainty_is_unavailable(self):
        row=demo.summarize([(1,1)],2)
        self.assertIsNone(row['mean_p_monte_carlo_se'])
        self.assertIsNone(row['gene_diversity_monte_carlo_se'])
        self.assertIsNone(row['loss_fraction_monte_carlo_se'])

    def test_checked_sample_output(self):
        self.assertEqual(demo.render(self.report),demo.SAMPLE.read_text())

    def test_cli_and_invalid_flag(self):
        command=[sys.executable,'-m','examples.slimulations.bottleneck_recovery.demo']
        p=subprocess.run(command+['--replicates','2'],capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['replicates_per_treatment'],2)
        bad=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(bad.returncode,2)


if __name__=='__main__': unittest.main()
