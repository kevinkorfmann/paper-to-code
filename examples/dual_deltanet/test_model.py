"""Independent dense Decimal oracle, algebraic controls, and causal checks."""

import copy
from decimal import Decimal, localcontext
import itertools
import json
import math
import subprocess
import sys
import unittest

from examples.dual_deltanet import demo, model


def decimal_oracle(state, key, value, alpha, mask, beta, gamma):
    # Independent dense A*(D*S)*B + eta*k*vhat^T, using 80-digit arithmetic.
    with localcontext() as ctx:
        ctx.prec = 80
        D = lambda x: Decimal(str(x))
        def normalized(v):
            norm = sum(x*x for x in v).sqrt()
            return [x/norm for x in v]
        s = [[D(x) for x in row] for row in state]
        k = normalized(list(map(D, key)))
        v = list(map(D, value))
        norm_v = sum(x*x for x in v).sqrt()
        vh = [x/norm_v for x in v]
        e = normalized([D(m)*x for m,x in zip(mask,vh)])
        b, g = D(beta), D(gamma)
        tau = sum(x*y for x,y in zip(v,e))
        a = [[Decimal(i==j)-b*k[i]*k[j] for j in range(len(k))] for i in range(len(k))]
        z = [[D(alpha[i])*x for x in row] for i,row in enumerate(s)]
        right = [[Decimal(i==j)-g*e[i]*vh[j] for j in range(len(v))] for i in range(len(v))]
        def multiply(x,y):
            return [[sum(x[i][j]*y[j][l] for j in range(len(y))) for l in range(len(y[0]))] for i in range(len(x))]
        base = multiply(multiply(a,z),right)
        eta = b*norm_v+g*(1-b)*tau
        return [[float(base[i][j]+eta*k[i]*vh[j]) for j in range(len(v))] for i in range(len(k))]


class DualDeltaTests(unittest.TestCase):
    def setUp(self):
        self.args = ([[.2,-.7,1.], [1.1,.4,-.3]], [3.,4.], [2.,-1.,3.], [.8,.6], [.2,.9,.5], .4, .7)

    def matrix_close(self,a,b,tol=2e-12):
        self.assertEqual((len(a),len(a[0])),(len(b),len(b[0])))
        for x,y in zip(itertools.chain.from_iterable(a),itertools.chain.from_iterable(b)):
            self.assertAlmostEqual(x,y,delta=tol*max(1,abs(y)))

    def test_sequential_matches_independent_decimal(self):
        self.matrix_close(model.step(*self.args),decimal_oracle(*self.args))

    def test_rank_two_matches_independent_decimal(self):
        self.matrix_close(model.factorized_step(*self.args),decimal_oracle(*self.args))

    def test_parameter_grid_matches_oracle(self):
        for beta,gamma in itertools.product((0.,.25,1.),repeat=2):
            args=(*self.args[:-2],beta,gamma)
            self.matrix_close(model.step(*args),decimal_oracle(*args))
            self.matrix_close(model.factorized_step(*args),decimal_oracle(*args))

    def test_gamma_zero_reduces_to_key_update(self):
        s,k,v,a,m,b,g=self.args;k=model.unit(k)
        retained=[[a[i]*x for x in row] for i,row in enumerate(s)]
        old=[sum(k[i]*retained[i][j] for i in range(len(k))) for j in range(len(v))]
        expected=[[retained[i][j]+b*k[i]*(v[j]-old[j]) for j in range(len(v))] for i in range(len(k))]
        self.matrix_close(model.step(s,k,v,a,m,b,0),expected)

    def test_beta_zero_can_still_reorganize_values(self):
        args=(*self.args[:-2],0.,.7)
        actual=model.step(*args)
        self.matrix_close(actual,decimal_oracle(*args))
        self.assertNotEqual(actual,[[self.args[3][i]*x for x in row] for i,row in enumerate(self.args[0])])

    def test_both_gates_zero_only_decay(self):
        s,k,v,a,m,_,_=self.args
        self.matrix_close(model.step(s,k,v,a,m,0,0),[[a[i]*x for x in row] for i,row in enumerate(s)])

    def test_zero_decay_forgets_previous_state(self):
        s,k,v,a,m,b,g=self.args
        first=model.step(s,k,v,[0,0],m,b,g)
        second=model.step([[99.,-88.,77.],[-66.,55.,-44.]],k,v,[0,0],m,b,g)
        self.matrix_close(first,second)

    def test_channel_decay_scales_rows(self):
        s,k,v,a,m,_,_=self.args
        result=model.step(s,k,v,[.1,.9],m,0,0)
        self.matrix_close(result,[[.1*x for x in s[0]],[.9*x for x in s[1]]])

    def test_uniform_selector_exact_address_at_gamma_one(self):
        s,k,v,a,_,b,_=self.args
        result=model.step(s,k,v,a,[.5]*3,b,1)
        vh=model.unit(v); norm=math.hypot(*v);k=model.unit(k)
        for i,row in enumerate(result):
            self.assertAlmostEqual(model.dot(row,vh),norm*k[i],places=12)

    def test_nonuniform_selector_residual_is_oblique(self):
        s,k,v,a,m,b,g=self.args
        kda=model.step(s,k,v,a,m,b,0)
        dual=model.step(s,k,v,a,m,b,g)
        vh=model.unit(v);e=model.unit([x*y for x,y in zip(m,vh)])
        tau=model.dot(v,e);k=model.unit(k)
        coefficient=1-g*model.dot(vh,e)
        for i in range(len(k)):
            before=tau*k[i]-model.dot(kda[i],e)
            after=tau*k[i]-model.dot(dual[i],e)
            self.assertAlmostEqual(after,coefficient*before,places=13)
        self.assertGreater(demo.report()['nonuniform_selector']['address_squared_error_after'],.3)

    def test_value_correction_preserves_orthogonal_projection(self):
        s,k,v,a,m,b,g=self.args
        kda=model.step(s,k,v,a,m,b,0);dual=model.step(*self.args)
        orthogonal=[1.,2.,0.] # dot([2,-1,3], orthogonal)=0
        for r1,r2 in zip(kda,dual):
            self.assertAlmostEqual(model.dot(r1,orthogonal),model.dot(r2,orthogonal),places=13)

    def test_beta_one_preserves_exact_key_association(self):
        s,k,v,a,m,_,g=self.args
        result=model.step(s,k,v,a,m,1,g);k=model.unit(k)
        for j in range(len(v)):
            self.assertAlmostEqual(sum(k[i]*result[i][j] for i in range(len(k))),v[j],places=12)

    def test_rank_two_correction_has_zero_three_by_three_determinant(self):
        s=[[1.,2.,3.],[-2.,.5,4.],[.7,-.9,2.]]
        a=[.8,.4,.7]
        result=model.step(s,[1,2,3],[4,-2,1],a,[.2,.3,.8],.4,.6)
        c=[[result[i][j]-a[i]*s[i][j] for j in range(3)] for i in range(3)]
        determinant=sum((1 if perm in [(0,1,2),(1,2,0),(2,0,1)] else -1)*math.prod(c[i][perm[i]] for i in range(3)) for perm in itertools.permutations(range(3)))
        self.assertAlmostEqual(determinant,0.,places=12)

    def test_preservation_counterexample(self):
        result=demo.report()['retention_counterexample']
        self.assertEqual(result['key_only_squared_error'],0.)
        self.assertAlmostEqual(result['dual_squared_error'],1/12,places=11)

    def test_exact_constructed_address_example(self):
        result=demo.report()['uniform_selector']
        self.assertEqual(result['address_before'],[1.,.5,1.])
        self.assertEqual(result['address_after'],[.5,.75,.5])
        self.assertEqual(result['address_squared_error_after'],.5625)
        self.assertEqual(result['orthogonal_value_channel_unchanged'],[2.,0.,3.])

    def test_inputs_are_not_mutated(self):
        before=copy.deepcopy(self.args)
        model.step(*self.args);model.factorized_step(*self.args)
        self.assertEqual(before,self.args)

    def test_key_scaling_invariance(self):
        s,k,v,a,m,b,g=self.args
        self.matrix_close(model.step(s,[10*x for x in k],v,a,m,b,g),model.step(*self.args))

    def test_joint_state_value_scaling(self):
        s,k,v,a,m,b,g=self.args
        actual=model.step([[7*x for x in row] for row in s],k,[7*x for x in v],a,m,b,g)
        self.matrix_close(actual,[[7*x for x in row] for row in model.step(*self.args)])

    def test_row_and_column_permutation_equivariance(self):
        s,k,v,a,m,b,g=self.args;rows=[1,0];cols=[2,0,1]
        actual=model.step([[s[i][j] for j in cols] for i in rows],[k[i] for i in rows],[v[j] for j in cols],[a[i] for i in rows],[m[j] for j in cols],b,g)
        original=model.step(*self.args)
        self.matrix_close(actual,[[original[i][j] for j in cols] for i in rows])

    def test_readout_scale_and_query_normalization(self):
        s=[[1.,2.],[3.,4.]]
        self.assertEqual(model.read(s,[10,0]),[1/math.sqrt(2),2/math.sqrt(2)])
        self.assertEqual(model.read(s,[-10,0]),[-1/math.sqrt(2),-2/math.sqrt(2)])

    def tokens(self):
        return [dict(key=[1,2],value=[3,-1,2],alpha=[.8,.7],channel_gate=[.2,.7,.9],beta=.3,gamma=.6,query=[2,-1]),
                dict(key=[-1,3],value=[-1,2,4],alpha=[.9,.6],channel_gate=[.8,.4,.1],beta=.7,gamma=.2,query=[1,3])]

    def test_sequence_factorized_equivalence(self):
        for first,second in zip(model.run(self.args[0],self.tokens()),model.run(self.args[0],self.tokens(),factorized=True)):
            self.matrix_close(first,second)

    def test_causal_prefix_invariance(self):
        tokens=self.tokens();prefix=model.run(self.args[0],tokens[:1])[1]
        tokens[1]['value']=[1e100,-1e100,1e100]
        self.assertEqual(model.run(self.args[0],tokens)[1][:1],prefix)

    def test_recurrence_order_matters(self):
        forward=model.run(self.args[0],self.tokens())[0]
        reverse=model.run(self.args[0],list(reversed(self.tokens())))[0]
        self.assertGreater(sum(abs(x-y) for x,y in zip(itertools.chain.from_iterable(forward),itertools.chain.from_iterable(reverse))),.1)

    def test_empty_sequence_returns_copy(self):
        result,outputs=model.run(self.args[0],[])
        self.assertEqual(result,self.args[0]);self.assertEqual(outputs,[])
        result[0][0]=99
        self.assertNotEqual(result,self.args[0])

    def test_normalization_extremes(self):
        for scale in (1e308,1e-300):
            normalized=model.unit([scale,scale])
            self.assertAlmostEqual(normalized[0],1/math.sqrt(2),places=14)
        self.assertEqual(model.unit([1e-320,0]),[1.,0.])

    def test_invalid_shapes(self):
        for idx,value in ((0,[]),(0,[[1,2],[3]]),(1,[1]),(2,[1]),(3,[1]),(4,[1])):
            args=list(copy.deepcopy(self.args));args[idx]=value
            with self.subTest(idx=idx),self.assertRaises(ValueError):model.step(*args)

    def test_invalid_and_nonfinite_numbers(self):
        for value in (True,'1',math.nan,math.inf):
            args=list(copy.deepcopy(self.args));args[0][0][0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):model.step(*args)

    def test_zero_direction_and_selector_rejected(self):
        for idx,value in ((1,[0,0]),(2,[0,0,0]),(4,[0,0,0])):
            args=list(copy.deepcopy(self.args));args[idx]=value
            with self.subTest(idx=idx),self.assertRaises(ValueError):model.step(*args)
        with self.assertRaises(ValueError):model.read(self.args[0],[0,0])

    def test_invalid_gates_rejected(self):
        for index,value in ((3,[-.1,.5]),(4,[1.1,.5,.5]),(5,-1),(6,2)):
            args=list(copy.deepcopy(self.args));args[index]=value
            with self.subTest(index=index),self.assertRaises(ValueError):model.step(*args)

    def test_overflow_is_explicit(self):
        with self.assertRaises(ValueError):model.dot([1e308,1e308],[1e308,1e308])

    def test_checked_output(self):
        self.assertEqual(demo.report(),json.loads(demo.SAMPLE.read_text()))

    def test_cli_sample_and_bad_flag(self):
        command=[sys.executable,'-m','examples.dual_deltanet.demo']
        actual=subprocess.run([*command,'--check-sample'],capture_output=True,text=True,check=True)
        self.assertEqual(actual.stdout,demo.SAMPLE.read_text())
        self.assertNotEqual(subprocess.run([*command,'--unknown'],capture_output=True).returncode,0)


if __name__=='__main__':
    unittest.main()
