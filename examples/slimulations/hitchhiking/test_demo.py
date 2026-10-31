"""Independent exact diplotype/meiosis and finite multinomial controls, plus real SLiM."""

from fractions import Fraction as F
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.hitchhiking import demo


def enumerate_meiosis(x, selected, c):
    """Enumerate ordered random-pairing diploids, fitness, and four meiosis paths."""
    weights = [F(0)]*4; total = F(0)
    for i, j in itertools.product(range(4), repeat=2):
        w = 1+F(selected, 2)*((i//2)+(j//2))
        mass = x[i]*x[j]*w; total += mass
        for hap, probability in [(i, (1-c)/2), (j, (1-c)/2),
                                 (2*(i//2)+j%2, c/2), (2*(j//2)+i%2, c/2)]:
            weights[hap] += mass*probability
    return [v/total for v in weights]


def enumerate_four_gametes(q):
    """All 4^4 ordered draws, independent of the moment formula implementation."""
    mean = F(0); second = F(0); diversity = F(0); ld = F(0); total = F(0)
    for draw in itertools.product(range(4), repeat=4):
        probability = math.prod(q[h] for h in draw); total += probability
        counts = [draw.count(h) for h in range(4)]
        p = F(counts[1]+counts[3], 4)
        mean += probability*p; second += probability*p*p
        diversity += probability*2*p*(1-p)
        ld += probability*F(counts[3]*counts[0]-counts[2]*counts[1], 16)
    assert total == 1
    return mean, second-mean*mean, diversity, ld


class HitchhikingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: demo.run_counts(selected=s, reversed=r, c=c, seed=demo.SEED+i)
                       for i, (name, s, r, c) in enumerate(demo.TREATMENTS)}
        cls.report = demo.build_report(results=cls.results)

    def test_exact_founder_diplotypes(self):
        pairs = demo.founder_pairs(400)
        self.assertEqual(pairs, {(0,0):100,(0,1):160,(0,2):0,(0,3):40,(1,1):64,
                                (1,2):0,(1,3):32,(2,2):0,(2,3):0,(3,3):4})
        for reverse in (0, 1):
            p = demo.founder_pairs(400, reverse)
            self.assertEqual(sum(p.values()), 400)
            h = [sum(n*((a==k)+(b==k)) for (a,b),n in p.items()) for k in range(4)]
            self.assertEqual(h, [int(800*x) for x in demo.founder_frequencies(reverse)])

    def test_recurrence_matches_independent_exact_meiosis(self):
        for x in (demo.founder_frequencies(0),demo.founder_frequencies(1),[F(1,4)]*4,[F(1,10),F(2,10),F(3,10),F(4,10)]):
            for s,c in itertools.product((0,1),(F(0),F(1,10),F(1,2))):
                self.assertEqual(demo.gamete_step(x,s,c),enumerate_meiosis(x,s,c))

    def test_exact_first_generation_values(self):
        a = demo.first_generation_prediction(400,0,1,0)
        b = demo.first_generation_prediction(400,1,1,0)
        self.assertEqual(a['expected_pB_fraction'],'23/44')
        self.assertEqual(b['expected_pB_fraction'],'21/44')
        self.assertEqual(a['expected_pA'],float(F(31,220)))
        self.assertEqual(demo.first_generation_prediction(400,0,0,0)['expected_pB_fraction'],'1/2')

    def test_finite_moments_match_full_enumeration(self):
        for q in ([F(1,4)]*4, [F(1,10),F(2,10),F(3,10),F(4,10)]):
            mean,var,h,d=enumerate_four_gametes(q)
            observed=demo.finite_moments(q,2)
            self.assertEqual([observed[k] for k in ('expected_pB','pB_variance','expected_gene_diversity_B','expected_sample_LD')], [mean,var,h,d])

    def test_recombination_preserves_first_generation_marginals(self):
        p=[demo.first_generation_prediction(400,0,1,c) for c in (0,.1,.5)]
        self.assertEqual({x['expected_pB'] for x in p},{float(F(23,44))})
        self.assertEqual({x['expected_pA'] for x in p},{float(F(31,220))})
        self.assertGreater(p[0]['expected_sample_LD'],p[-1]['expected_sample_LD'])

    def test_no_selection_neutral_marginals(self):
        for c in (0,.1,.5):
            for x in demo.deterministic(0,0,c,30):
                self.assertAlmostEqual(x[1]+x[3],.5)
                self.assertAlmostEqual(x[2]+x[3],.1)

    def test_zero_initial_association_stays_zero_deterministically(self):
        x=[F(1,4)]*4
        for _ in range(5):
            x=demo.gamete_step(x,1,F(0))
            self.assertEqual(x[1]+x[3],F(1,2))
            self.assertEqual(x[3]*x[0]-x[2]*x[1],0)

    def test_reversing_marker_complements_trajectory(self):
        for c in (0,.1,.5):
            forward=demo.deterministic(0,1,c,30);reverse=demo.deterministic(1,1,c,30)
            for x,y in zip(forward,reverse):
                self.assertAlmostEqual(x[1]+x[3]+y[1]+y[3],1,places=12)
                self.assertAlmostEqual(x[2]+x[3],y[2]+y[3],places=12)

    def test_probability_simplex_and_selected_allele_growth(self):
        for c in (0,.1,.5):
            last=.1
            for x in demo.deterministic(0,1,c,100):
                self.assertTrue(all(v>=-1e-14 for v in x));self.assertAlmostEqual(sum(x),1,places=12)
                self.assertGreaterEqual(x[2]+x[3]+1e-14,last);last=x[2]+x[3]

    def test_actual_all_replicates_and_founders(self):
        for name,_,reverse,_ in demo.TREATMENTS:
            rows=self.results[name]
            self.assertEqual(set(rows),{0,1,10,30})
            for counts in rows.values():
                self.assertEqual(len(counts),256)
                self.assertTrue(all(sum(r)==800 and min(r)>=0 for r in counts))
            self.assertEqual(rows[0],[[int(800*x) for x in demo.founder_frequencies(reverse)]]*256)

    def test_first_generation_means_within_monte_carlo_uncertainty(self):
        for name,s,reverse,c in demo.TREATMENTS:
            expected=demo.first_generation_prediction(400,reverse,s,c)
            actual=demo.statistics_for(self.results[name][1],400)
            # A 5-SE numerical sanity bound, not a confidence claim.
            self.assertLess(abs(actual['mean_pB']-expected['expected_pB']),5*math.sqrt(expected['pB_variance']/256))

    def test_first_generation_sampling_variance(self):
        for name,s,reverse,c in demo.TREATMENTS:
            expected=demo.first_generation_prediction(400,reverse,s,c)['pB_variance']
            observed=demo.statistics_for(self.results[name][1],400)['pB_sample_variance']
            self.assertLess(abs(observed/expected-1),.3)

    def test_first_generation_diversity_and_LD(self):
        for name,s,reverse,c in demo.TREATMENTS:
            rows=self.results[name][1];pred=demo.first_generation_prediction(400,reverse,s,c)
            for key,values in [('expected_gene_diversity_B',[2*((r[1]+r[3])/800)*(1-(r[1]+r[3])/800) for r in rows]),
                               ('expected_sample_LD',[(r[3]*r[0]-r[2]*r[1])/800**2 for r in rows])]:
                se=statistics.stdev(values)/math.sqrt(256)
                self.assertLess(abs(statistics.mean(values)-pred[key]),5*se)

    def test_no_selection_exact_finite_diversity_curve(self):
        for t,rows in self.results['linked_neutral'].items():
            if t==0:continue
            observed=demo.statistics_for(rows,400)
            expected=.5*(F(799,800)**t)
            self.assertLess(abs(observed['mean_gene_diversity_B']-float(expected)),5*observed['gene_diversity_B_monte_carlo_se'])

    def test_hitchhiking_and_controls_have_expected_direction(self):
        end={name:demo.statistics_for(rows[30],400) for name,rows in self.results.items()}
        self.assertGreater(end['linked_selected']['mean_pB'],.98)
        self.assertLess(end['reversed_linked_selected']['mean_pB'],.02)
        self.assertLess(abs(end['linked_neutral']['mean_pB']-.5),.04)
        self.assertLess(end['free_recombination_selected']['mean_pB'],.65)
        self.assertGreater(end['free_recombination_selected']['mean_gene_diversity_B'],.4)

    def test_complete_linkage_preserves_absent_haplotype(self):
        for name,index in [('linked_selected',2),('linked_neutral',2),('reversed_linked_selected',3)]:
            self.assertTrue(all(row[index]==0 for rows in self.results[name].values() for row in rows))
        self.assertTrue(any(row[2]>0 for row in self.results['free_recombination_selected'][1]))

    def test_selected_fixation_implies_marker_direction_with_no_recombination(self):
        for row in self.results['linked_selected'][30]:
            if row[2]+row[3]==800:self.assertEqual(row[3],800)
        for row in self.results['reversed_linked_selected'][30]:
            if row[2]+row[3]==800:self.assertEqual(row[2],800)

    def test_versions_and_actual_native_fitness_audit(self):
        self.assertEqual(platform.python_version(),'3.14.5')
        self.assertEqual(demo.check_slim_version('slim'),'4.0.1')
        # run_counts parses each engine's version(), exact native fitness, and every founder pair.
        for treatment in self.report['treatments']:
            self.assertIn(treatment['fitness_aa_Aa_AA'],[[1.,1.5,2.],[1.,1.,1.]])

    def test_zero_generation_run_and_determinism(self):
        a=demo.run_counts(n=100,replicates=2,generations=0,seed=123)
        self.assertEqual(set(a),{0})
        x=demo.run_counts(n=100,replicates=2,generations=2,c=.5,seed=123)
        self.assertEqual(x,demo.run_counts(n=100,replicates=2,generations=2,c=.5,seed=123))

    def test_invalid_python_parameters(self):
        for kwargs in [{'n':150},{'n':True},{'replicates':0},{'generations':-1},{'selected':2},
                       {'reversed':True},{'c':float('nan')},{'c':.51},{'seed':0}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):demo.run_counts(**kwargs)
        for x in ([.1,.2],[-.1,.1,.5,.5],[.1]*4,[float('inf'),0,0,0]):
            with self.assertRaises(ValueError):demo.gamete_step(x,1,0)

    def test_eidos_rejects_invalid_configuration(self):
        process=subprocess.run(['slim','-d','N=150',str(demo.SCRIPT)],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(process.returncode,0);self.assertIn('invalid N',process.stderr)

    def test_parser_rejects_missing_duplicate_or_wrong_audits(self):
        process=subprocess.run(['slim','-s','123','-d','N=100','-d','REPLICATES=1','-d','GENERATIONS=0',str(demo.SCRIPT)],capture_output=True,text=True,check=True,timeout=10)
        text=process.stdout
        good=demo.parse_output(text,100,1,0,1,0);self.assertEqual(good[0],[[100,80,0,20]])
        for wrong in ['',text.replace('FIT,1,1.5','FIT,1,1.0'),text.replace('INIT,0,0,0,25','INIT,0,0,0,24'),
                      text+'DATA,0,0,100,80,0,20\n',text.replace('Eidos version 3.0.1','Eidos version 9.0')]:
            with self.assertRaises(ValueError):demo.parse_output(wrong,100,1,0,1,0)

    def test_summary_sample_and_nonconditioned_counts(self):
        self.assertEqual(demo.render(self.report),demo.SAMPLE.read_text())
        for t in self.report['treatments']:
            last=t['checkpoints'][-1]
            self.assertLessEqual(last['selected_A_fixed']+last['selected_A_lost'],256)
            self.assertEqual(len(self.results[t['name']][30]),256)

    def test_cli_small_run_and_invalid_flag(self):
        command=[sys.executable,'-m','examples.slimulations.hitchhiking.demo']
        p=subprocess.run(command+['--n','100','--replicates','2','--generations','1'],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(json.loads(p.stdout)['treatments']),4)
        bad=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(bad.returncode,2)


if __name__=='__main__':unittest.main()
