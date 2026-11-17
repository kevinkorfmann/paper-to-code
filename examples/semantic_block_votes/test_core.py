import json
import math
from decimal import Decimal as D, localcontext
from pathlib import Path
import subprocess
import sys
import unittest
from .core import route, attend
from .demo import run


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.q=[[2.,0.],[0.,2.]]
        self.c=[[1.,0.],[0.,1.],[math.sqrt(.5),math.sqrt(.5)]]
        self.labels=[0,1,2,0,1,2]

    def route(self, **kw):
        args=dict(queries=self.q,centroids=self.c,labels=self.labels,query_index=5,dense_positions=[],budget=1)
        args.update(kw)
        return route(**args)

    def test_decimal_votes(self):
        # Independent high-precision norms, exponentials and normalization.
        with localcontext() as ctx:
            ctx.prec=80
            for temp in (.05,.1,1.):
                r=self.route(temperature=temp)
                for q,actual in zip(self.q,r['head_votes']):
                    q=list(map(lambda x:D(str(x)),q))
                    norm=sum(x*x for x in q).sqrt()
                    raw=[(sum(a*D(str(b)) for a,b in zip(q,c))/norm/D(str(temp))).exp() for c in self.c]
                    for a,b in zip(actual,raw): self.assertAlmostEqual(a,float(b/sum(raw)),places=13)

    def test_vote_order_not_logit_order(self):
        r=run()
        self.assertEqual(r['selected_clusters'],[0])
        self.assertEqual(r['alternative_selected_cluster'],2)

    def test_head_vote_normalization(self):
        for row in self.route()['head_votes']: self.assertAlmostEqual(sum(row),1)

    def test_query_scaling_invariance(self):
        a=self.route(); b=self.route(queries=[[200.,0.],[0.,.02]])
        self.assertEqual(a['indices'],b['indices'])
        for x,y in zip(a['scores'],b['scores']): self.assertAlmostEqual(x,y)

    def test_head_permutation(self):
        self.assertEqual(self.route()['indices'],self.route(queries=self.q[::-1])['indices'])

    def test_duplicate_heads(self):
        a=self.route(); b=self.route(queries=self.q*2)
        self.assertEqual(a['indices'],b['indices'])
        for x,y in zip(a['scores'],b['scores']): self.assertAlmostEqual(y,2*x)

    def test_whole_block_overshoot(self):
        r=self.route()
        self.assertEqual(r['historical_count'],2)
        self.assertEqual(r['indices'],[0,3])

    def test_zero_budget_dense_only(self):
        r=self.route(budget=0,dense_positions=[4,5])
        self.assertEqual(r['indices'],[4,5]); self.assertEqual(r['clusters'],[])

    def test_all_keys_recovery(self):
        self.assertEqual(self.route(budget=999)['indices'],list(range(6)))
        self.assertEqual(run()['all_keys_max_error'],0)

    def test_size_penalty_changes_rank(self):
        args=dict(queries=[[1.,0.]],centroids=[[1.,0.],[1.,0.]],labels=[0,0,0,1],query_index=3,budget=1)
        self.assertEqual(self.route(**args,beta=0)['clusters'],[0])
        self.assertEqual(self.route(**args,beta=.25)['clusters'],[1])

    def test_causal_sizes_exclude_dense_and_future(self):
        r=self.route(labels=self.labels+[0,1,2],query_index=4,dense_positions=[4],budget=2)
        self.assertEqual(r['visible_block_sizes'],[2,1,1])
        self.assertEqual(r['indices'],[0,1,3,4])
        self.assertEqual(len(r['indices']),len(set(r['indices'])))
        self.assertLessEqual(max(r['indices']),4)

    def test_future_assignments_not_counts(self):
        a=self.route(query_index=2)
        b=self.route(labels=[0,1,2,2,2,2],query_index=2)
        self.assertEqual(a,b)

    def test_empty_blocks_skipped(self):
        r=self.route(labels=[1]*6)
        self.assertEqual(r['clusters'],[1]); self.assertEqual(r['indices'],list(range(6)))

    def test_scalar_attention_decimal(self):
        with localcontext() as ctx:
            ctx.prec=80
            for q in (-3,0,2):
                raw=[(D(q)*D(k)).exp() for k in (1,-1)]
                expected=(raw[0]*D(2)+raw[1]*D(-3))/sum(raw)
                result=attend([[q]],[[1],[-1]],[[2],[-3]],[0,1])[0][0]
                self.assertAlmostEqual(result,float(expected),places=13)

    def test_original_query_norm_used_in_attention(self):
        a=attend([[1]],[[1],[-1]],[[1],[-1]],[0,1])[0][0]
        b=attend([[2]],[[1],[-1]],[[1],[-1]],[0,1])[0][0]
        self.assertGreater(b,a)

    def test_attention_scale(self):
        a=attend([[1,1]],[[1,1],[0,0]],[[1],[0]],[0,1])[0][0]
        self.assertAlmostEqual(a,1/(1+math.exp(-math.sqrt(2))))

    def test_values_constant(self):
        for q in self.q:
            self.assertAlmostEqual(attend([q],self.c,[[7]]*3,[0,2])[0][0],7)

    def test_index_order_invariant(self):
        a=attend(self.q,self.c,[[1],[2],[3]],[0,2])
        b=attend(self.q,self.c,[[1],[2],[3]],[2,0])
        for x,y in zip(a,b): self.assertAlmostEqual(x[0],y[0])

    def test_bad_budget(self):
        for b in (-1,True,.5):
            with self.assertRaises(ValueError): self.route(budget=b)

    def test_bad_vectors(self):
        for q in ([[0,0]],[[math.nan,0]],[[1]],[[1],[1,2]],[1,2]):
            with self.assertRaises(ValueError): self.route(queries=q)
        with self.assertRaises(ValueError): self.route(centroids=[[2,0]])

    def test_bad_positions(self):
        for dense in ([6],[1,1],[-1],[True]):
            with self.assertRaises(ValueError): self.route(dense_positions=dense)
        for t in (-1,6,True):
            with self.assertRaises(ValueError): self.route(query_index=t)

    def test_bad_labels(self):
        for labels in ([],[3],[True],[-1]):
            with self.assertRaises(ValueError): self.route(labels=labels)

    def test_bad_temperature_beta(self):
        for temp in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError): self.route(temperature=temp)
        for beta in (-1,math.inf,math.nan):
            with self.assertRaises(ValueError): self.route(beta=beta)

    def test_invalid_attention(self):
        for ids in ([],[0,0],[3],[-1],[True]):
            with self.assertRaises(ValueError): attend(self.q,self.c,[[1]]*3,ids)
        with self.assertRaises(ValueError): attend(self.q,self.c,[[1]],[0])

    def test_sample(self):
        self.assertEqual(run(),json.loads(Path(__file__).with_name('sample_output.json').read_text()))

    def test_cli(self):
        out=subprocess.check_output([sys.executable,'-m','examples.semantic_block_votes.demo'],text=True)
        self.assertEqual(out,Path(__file__).with_name('sample_output.json').read_text())
