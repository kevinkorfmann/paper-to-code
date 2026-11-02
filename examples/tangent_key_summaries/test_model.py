"""Independent Decimal oracle, convexity identities and diagnostic counterexamples."""
import copy
from decimal import Decimal as D,localcontext
import json
import math
import platform
import subprocess
import sys
import unittest

from examples.tangent_key_summaries import demo,model


def oracle(keys,reference,query):
    """Direct 80-digit formulas, not production softmax or summary helpers."""
    with localcontext() as context:
        context.prec=80
        keys=[[D(str(x)) for x in k] for k in keys]
        r=[D(str(x)) for x in reference];q=[D(str(x)) for x in query]
        scale=D(len(r)).sqrt()
        reference_scores=[sum(a*b for a,b in zip(r,k))/scale for k in keys]
        exp=[x.exp() for x in reference_scores];total=sum(exp)
        p=[x/total for x in exp]
        key=[sum(w*k[j] for w,k in zip(p,keys)) for j in range(len(r))]
        bias=-sum(w*w.ln() for w in p)
        tangent=sum(a*b for a,b in zip(q,key))/scale+bias
        query_scores=[sum(a*b for a,b in zip(q,k))/scale for k in keys]
        exact=sum(x.exp() for x in query_scores).ln()
        query_p=[(x-exact).exp() for x in query_scores]
        kl=sum(w*(w.ln()-v.ln()) for w,v in zip(p,query_p))
        return {'key':list(map(float,key)),'bias':float(bias),'tangent':float(tangent),'exact':float(exact),'kl':float(kl)}


class TangentTests(unittest.TestCase):
    def assert_summary_close(self,a,b,places=12):
        for x,y in zip(a.key,b.key): self.assertAlmostEqual(x,y,places=places)
        self.assertAlmostEqual(a.bias,b.bias,places=places)

    def test_zero_reference_is_mean_key_and_log_count(self):
        s=model.summarize([[1.,2.],[3.,4.],[5.,9.]],[0.,0.])
        self.assertEqual(s.key,(3.,5.));self.assertAlmostEqual(s.bias,math.log(3),places=14)

    def test_singleton_is_exact_for_every_query(self):
        for r in ([0.,0.],[1.,-2.]):
            s=model.summarize([[3.,4.]],r)
            self.assertEqual(s.key,(3.,4.));self.assertEqual(s.bias,0)
            for q in ([0.,0.],[-3.,2.]): self.assertEqual(model.score(s,q),model.log_mass([[3.,4.]],q))

    def test_identical_keys_are_exact_and_need_count_bias(self):
        keys=[[2.,-1.]]*5;s=model.summarize(keys,[1.,2.])
        self.assertAlmostEqual(s.bias,math.log(5),places=14)
        self.assertAlmostEqual(model.score(s,[-2.,3.]),model.log_mass(keys,[-2.,3.]),places=13)

    def test_entropy_bounds(self):
        keys=[[-3.,2.],[1.,-4.],[5.,2.]]
        for r in ([0.,0.],[3.,-2.],[1000.,0.]):
            s=model.summarize(keys,r)
            self.assertGreaterEqual(s.bias,-1e-14);self.assertLessEqual(s.bias,math.log(3)+1e-14)

    def test_tangent_touches_at_reference(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]]
        for r in ([0.,0.],[.2,.7],[-1.,2.]):
            self.assertAlmostEqual(model.score(model.summarize(keys,r),r),model.log_mass(keys,r),places=13)

    def test_key_is_independent_finite_difference_gradient(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]];r=[.2,.7];s=model.summarize(keys,r)
        h=1e-5
        for j in range(2):
            left=r.copy();right=r.copy();left[j]-=h;right[j]+=h
            derivative=(model.log_mass(keys,right)-model.log_mass(keys,left))/(2*h)
            self.assertAlmostEqual(derivative,s.key[j]/math.sqrt(2),places=9)

    def test_entropy_matches_lse_intercept(self):
        keys=[[-3.,2.],[1.,-4.],[5.,2.]];r=[.3,.1];s=model.summarize(keys,r)
        self.assertAlmostEqual(s.bias,model.log_mass(keys,r)-model.dot_scaled(r,s.key),places=14)

    def test_80_digit_oracle_for_components_and_query_scores(self):
        fixtures=[([[-2.],[2.]],[1.],[2.]),
                  ([[-2.,1.],[0.,-1.],[3.,2.]],[.2,.7],[-1.,.3]),
                  ([[0.,1.,2.,3.],[4.,-1.,0.,2.]],[-.2,.7,1.,.4],[1.,0.,-.3,.1])]
        for keys,r,q in fixtures:
            expected=oracle(keys,r,q);actual=model.summarize(keys,r)
            for a,b in zip(actual.key,expected['key']): self.assertAlmostEqual(a,b,places=13)
            self.assertAlmostEqual(actual.bias,expected['bias'],places=13)
            self.assertAlmostEqual(model.score(actual,q),expected['tangent'],places=13)
            self.assertAlmostEqual(model.log_mass(keys,q),expected['exact'],places=13)

    def test_sqrt_dimension_scaling(self):
        self.assertEqual(model.dot_scaled([1.,1.,1.,1.],[1.,1.,1.,1.]),2.)
        self.assertAlmostEqual(model.log_mass([[1.,1.,1.,1.]]*2,[1.,1.,1.,1.]),2+math.log(2),places=14)

    def test_lower_bound_on_two_dimensional_query_grid(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]]
        summaries=[model.summarize(keys,r) for r in ([0.,0.],[1.,-1.],[-1.,2.])]
        for x in range(-5,6):
            for y in range(-5,6):
                self.assertLessEqual(model.envelope(summaries,[x/2,y/2]),model.log_mass(keys,[x/2,y/2])+1e-13)

    def test_gap_equals_forward_kl_from_reference_to_query(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]];r=[.2,.7]
        for q in ([0.,0.],[2.,-1.],[-1.,.3]):
            result=oracle(keys,r,q)
            gap=model.log_mass(keys,q)-model.score(model.summarize(keys,r),q)
            self.assertAlmostEqual(gap,result['kl'],places=13)
            self.assertGreaterEqual(gap,-1e-13)

    def test_adding_reference_cannot_reduce_envelope(self):
        keys=[[-2.],[2.]];a=model.summarize(keys,[0.]);b=model.summarize(keys,[1.])
        for q in ([-2.],[0.],[.5],[1.],[2.]):
            self.assertGreaterEqual(model.envelope([a,b],q),model.envelope([a],q))

    def test_reference_order_and_duplicates_do_not_change_envelope(self):
        keys=[[-2.],[2.]];a=model.summarize(keys,[0.]);b=model.summarize(keys,[1.])
        self.assertEqual(model.envelope([a,b],[2.]),model.envelope([b,a,b],[2.]))

    def test_key_permutation_invariance(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]];r=[.2,.7]
        self.assert_summary_close(model.summarize(keys,r),model.summarize(list(reversed(keys)),r))

    def test_key_translation_shifts_score_but_not_entropy(self):
        keys=[[-2.,1.],[0.,-1.],[3.,2.]];r=[.2,.7];q=[-1.,.3];shift=[4.,-2.]
        a=model.summarize(keys,r);b=model.summarize([[x+y for x,y in zip(k,shift)] for k in keys],r)
        self.assertAlmostEqual(a.bias,b.bias,places=13)
        for x,y,c in zip(a.key,b.key,shift): self.assertAlmostEqual(x+c,y,places=13)
        self.assertAlmostEqual(model.score(b,q)-model.score(a,q),model.dot_scaled(q,shift),places=13)

    def test_duplicating_every_key_adds_log_multiplicity(self):
        keys=[[-2.],[2.]];a=model.summarize(keys,[1.]);b=model.summarize(keys*3,[1.])
        self.assertAlmostEqual(a.key[0],b.key[0],places=13)
        self.assertAlmostEqual(b.bias-a.bias,math.log(3),places=13)

    def test_orthogonal_rotation_preserves_scores_and_bias(self):
        rotate=lambda v:[-v[1],v[0]]
        keys=[[-2.,1.],[0.,-1.],[3.,2.]];r=[.2,.7];q=[-1.,.3]
        a=model.summarize(keys,r);b=model.summarize([rotate(k) for k in keys],rotate(r))
        self.assertAlmostEqual(a.bias,b.bias,places=13)
        self.assertAlmostEqual(model.score(a,q),model.score(b,rotate(q)),places=13)

    def test_shared_reference_merge_matches_flat_unequal_chunks(self):
        chunks=[[[0.],[1.]],[[2.],[3.],[4.]]];r=[.7]
        self.assert_summary_close(model.merge([model.summarize(c,r) for c in chunks],r),model.summarize(sum(chunks,[]),r))

    def test_multilevel_shared_reference_merge_matches_flat(self):
        keys=[[float(i),float((i*i)%5)] for i in range(12)];r=[.3,-.7]
        level=[model.summarize(keys[i:i+2],r) for i in range(0,12,2)]
        mid=[model.merge(level[i:i+3],r) for i in range(0,6,3)]
        self.assert_summary_close(model.merge(mid,r),model.summarize(keys,r))

    def test_child_biases_are_required_when_merging(self):
        children=[model.summarize([[0.]]*n,[0.]) for n in (2,3)]
        correct=model.merge(children,[0.]);wrong=model.merge([model.Tangent(c.key,0.) for c in children],[0.])
        self.assertAlmostEqual(correct.bias,math.log(5),places=14)
        self.assertAlmostEqual(wrong.bias,math.log(2),places=14)

    def test_mismatched_reference_merge_is_not_flat_exactness(self):
        chunks=[[[-2.],[2.]],[[0.],[1.]]]
        merged=model.merge([model.summarize(c,[0.]) for c in chunks],[1.])
        direct=model.summarize(sum(chunks,[]),[1.])
        self.assertGreater(abs(merged.key[0]-direct.key[0]),.1)
        for q in ([-2.],[0.],[1.],[2.]):
            self.assertLessEqual(model.score(merged,q),model.log_mass(sum(chunks,[]),q)+1e-13)

    def test_normalized_probabilities_are_not_lower_bounds(self):
        result=model.chunk_probabilities([[-6.],[6.],[2.],[2.]],[1.],[[0.]],2)
        self.assertLess(result['estimated_log_masses'][0],result['exact_log_masses'][0])
        self.assertGreater(result['estimated_chunk_probabilities'][1],result['exact_chunk_probabilities'][1]+.8)

    def test_constructed_ranking_failure_and_reference_repair(self):
        r=demo.build_report()['chunk_ranking_counterexample']
        self.assertEqual(r['zero_reference']['top_one_chunk_diagnostic'],1)
        self.assertEqual(r['zero_and_one_references']['top_one_chunk_diagnostic'],0)
        self.assertAlmostEqual(r['dense_attention_output_for_values_1100'],.964663365,places=9)
        self.assertEqual(r['zero_reference']['diagnostic_selected_chunk_output'],0)
        self.assertEqual(r['zero_and_one_references']['diagnostic_selected_chunk_output'],1)

    def test_visible_prefix_ignores_extreme_future_keys(self):
        keys=[[-6.],[6.],[2.],[2.]]
        args=([1.],[[0.],[1.]],2)
        self.assertEqual(model.chunk_probabilities(keys,*args),model.chunk_probabilities(keys+[[-10000.],[10000.]],*args,visible=4))

    def test_partial_visible_chunk_is_recomputed(self):
        keys=[[0.],[1.],[2.],[10000.]]
        self.assertEqual(model.chunk_probabilities(keys,[1.],[[0.]],2,visible=3),model.chunk_probabilities(keys[:3],[1.],[[0.]],2))

    def test_extreme_logits_remain_finite(self):
        keys=[[-1000.],[1000.]];s=model.summarize(keys,[1.])
        self.assertEqual(s.key,(1000.,));self.assertEqual(s.bias,0.)
        self.assertEqual(model.score(s,[1.]),1000.)
        self.assertEqual(model.log_mass(keys,[1.]),1000.)

    def test_inputs_are_not_mutated(self):
        keys=[[-2.],[2.]];refs=[[0.],[1.]];q=[.5]
        before=copy.deepcopy((keys,refs,q))
        model.chunk_probabilities(keys,q,refs,1)
        self.assertEqual((keys,refs,q),before)

    def test_invalid_vectors_matrices_and_ranges(self):
        for keys,r in [([], [0.]),([[1.],[]],[0.]),([[1.,2.]],[0.]),([[True]],[0.]),([[float('nan')]],[0.]),([[1.]],[float('inf')])]:
            with self.assertRaises(ValueError): model.summarize(keys,r)
        with self.assertRaises(ValueError): model.dot_scaled([1e308],[1e308])
        for bias in (True,float('nan'),float('inf')):
            with self.assertRaises(ValueError): model.Tangent((1.,),bias)

    def test_invalid_merge_envelope_and_mask(self):
        with self.assertRaises(ValueError): model.merge([],[1.])
        with self.assertRaises(ValueError): model.envelope([],[1.])
        with self.assertRaises(ValueError): model.score(None,[1.])
        with self.assertRaises(ValueError): model.merge([model.Tangent((1.,2.),0.)],[1.])
        for size,visible in ((0,1),(True,1),(1,0),(1,3),(1,True)):
            with self.assertRaises(ValueError): model.chunk_probabilities([[1.],[2.]],[1.],[[0.]],size,visible)
        with self.assertRaises(ValueError): model.chunk_probabilities([[1.]],[1.],[[0.,1.]],1)

    def test_checked_sample_output(self):
        self.assertEqual(demo.render(demo.build_report()),demo.SAMPLE.read_text())

    def test_cli_checked_output_and_unknown_flag(self):
        command=[sys.executable,'-m','examples.tangent_key_summaries.demo']
        result=subprocess.run(command+['--check-sample'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout,demo.SAMPLE.read_text())
        bad=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(bad.returncode,2)

    def test_pinned_interpreter_and_determinism(self):
        self.assertEqual(platform.python_version(),'3.14.5')
        self.assertEqual(demo.build_report(),demo.build_report())


if __name__=='__main__': unittest.main()
