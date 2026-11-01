"""Independent Gaussian-mixture Decimal and finite-difference gradient controls."""

import copy
from decimal import Decimal as D, localcontext
import json
import math
import platform
import subprocess
import sys
import unittest

from examples.switching_linear_attention import demo, model


def decimal_step(state, key, value, logits, beta, query, read_logits):
    """Direct prior * Gaussian likelihood normalization; no production helpers."""
    with localcontext() as ctx:
        ctx.prec = 80
        convert = lambda x: D(str(x))
        w = [[[convert(x) for x in row] for row in expert] for expert in state]
        k,v,q = [list(map(convert, vector)) for vector in (key,value,query)]
        j,d = len(w),len(v)
        post = [[D(0)]*d for _ in range(j)]
        new = copy.deepcopy(w)
        for c in range(d):
            priors = [convert(logits[e][c]).exp() for e in range(j)]
            priors = [p/sum(priors) for p in priors]
            residual = [v[c]-sum(x*y for x,y in zip(w[e][c],k)) for e in range(j)]
            mass = [priors[e]*(-residual[e]**2/2).exp() for e in range(j)]
            for e in range(j):
                post[e][c] = mass[e]/sum(mass)
                for i in range(d):
                    new[e][c][i] += convert(beta[e][c])*post[e][c]*residual[e]*k[i]
        out=[]
        for c in range(d):
            priors=[convert(read_logits[e][c]).exp() for e in range(j)]
            out.append(sum(priors[e]*sum(x*y for x,y in zip(new[e][c],q)) for e in range(j))/sum(priors))
        return new,post,out


def objective(state,key,value,logits):
    """Independent marginal log likelihood, omitting the constant normal density term."""
    result=0.
    for c,target in enumerate(value):
        prior=[math.exp(row[c]) for row in logits]
        masses=[p*math.exp(-.5*(target-sum(a*b for a,b in zip(w[c],key)))**2)
                for p,w in zip(prior,state)]
        result+=math.log(sum(masses)/sum(prior))
    return result


class SwitchingTests(unittest.TestCase):
    def setUp(self):
        self.state=[[[.3,.4],[-.2,.1]],[[.7,-.3],[.2,-.4]],[[.1,-.5],[.8,.2]]]
        self.key=[.6,.8];self.value=[.7,-.2];self.query=[-.4,.5]
        self.logits=[[.2,-.3],[-.5,.4],[.7,.1]]
        self.beta=[[.2,.4],[.6,.8],[.3,.7]]
        self.read_logits=[[-.4,.7],[.1,-.2],[.3,.5]]

    def assertNested(self,a,b,places=12):
        if isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assertNested(x,y,places)
        else:self.assertAlmostEqual(float(a),float(b),places=places)

    def update(self,state=None,**kwargs):
        args=dict(key=self.key,value=self.value,write_logits=self.logits,beta=self.beta)
        args.update(kwargs)
        return model.update(self.state if state is None else state,**args)

    def token(self,**kwargs):
        result=dict(key=self.key,value=self.value,query=self.query,write_logits=self.logits,
                    beta=self.beta,read_logits=self.read_logits)
        result.update(kwargs);return result

    def test_analytical_coordinate_responsibilities(self):
        data=demo.report()['coordinate_responsibilities'];p=1/(1+math.exp(-4))
        self.assertNested(data['write_responsibilities'],[[p,1-p],[1-p,p]])
        self.assertNested(data['updated_state'],[[[2-p,0],[-2+3*(1-p),0]],
                                                [[-2+3*(1-p),0],[2-p,0]]])

    def test_coordinate_posterior_is_not_one_joint_expert_choice(self):
        data=demo.report()['coordinate_responsibilities']
        self.assertEqual([sum(x*x for x in row) for row in data['old_prediction_errors']],[10.,10.])
        self.assertGreater(data['write_responsibilities'][0][0],.98)
        self.assertLess(data['write_responsibilities'][0][1],.02)

    def test_decimal_gaussian_oracle(self):
        expected=decimal_step(self.state,self.key,self.value,self.logits,self.beta,self.query,self.read_logits)
        new,post=self.update()
        self.assertNested(new,expected[0]);self.assertNested(post,expected[1])
        self.assertNested(model.read(new,self.query,self.read_logits),expected[2])

    def test_decimal_multistep_oracle(self):
        state=self.state;oracle=self.state
        for t in range(5):
            key=[.6,(-1)**t*.8];value=[.7+t/10,-.2]
            oracle,post,out=decimal_step(oracle,key,value,self.logits,self.beta,self.query,self.read_logits)
            state,actual_post=model.update(state,key,value,self.logits,self.beta)
            self.assertNested(state,oracle);self.assertNested(actual_post,post)
            self.assertNested(model.read(state,self.query,self.read_logits),out)

    def test_update_matches_independent_likelihood_gradient(self):
        new,_=self.update();h=1e-5
        for e in range(3):
            for c in range(2):
                for i in range(2):
                    plus=copy.deepcopy(self.state);minus=copy.deepcopy(self.state)
                    plus[e][c][i]+=h;minus[e][c][i]-=h
                    gradient=(objective(plus,self.key,self.value,self.logits)-objective(minus,self.key,self.value,self.logits))/(2*h)
                    self.assertAlmostEqual((new[e][c][i]-self.state[e][c][i])/self.beta[e][c],gradient,places=9)

    def test_single_expert_reduces_to_delta_rule(self):
        old=self.state[:1];beta=self.beta[:1]
        new,post=model.update(old,self.key,self.value,[[8.,-5.]],beta)
        self.assertEqual(post,[[1.,1.]])
        expected=[[[old[0][c][i]+beta[0][c]*(self.value[c]-sum(x*y for x,y in zip(old[0][c],self.key)))*self.key[i]
                    for i in range(2)] for c in range(2)]]
        self.assertNested(new,expected)

    def test_zero_learning_rate_freezes_state(self):
        new,_=self.update(beta=[[0.,0.]]*3);self.assertEqual(new,self.state)

    def test_zero_key_freezes_state_and_posterior_equals_prior(self):
        new,post=self.update(key=[0.,0.]);self.assertEqual(new,self.state)
        self.assertNested(post,model.column_softmax(self.logits))

    def test_exact_predictions_leave_state_unchanged(self):
        state=[[[1.,0.],[2.,0.]]]*3
        new,post=self.update(state,key=[1.,0.],value=[1.,2.])
        self.assertEqual(new,state);self.assertNested(post,model.column_softmax(self.logits))

    def test_identical_experts_stay_identical(self):
        data=demo.report()['identical_expert_control']
        self.assertEqual(data['final_state_after_three_identical_tokens'][0],data['final_state_after_three_identical_tokens'][1])
        self.assertNested(data['outputs'],[[.5,.5],[.75,.75],[.875,.875]])

    def test_asymmetric_write_priors_can_break_identical_initial_state(self):
        state=[[[0.]],[[0.]]]
        new,post=model.update(state,[1.],[1.],[[math.log(3)],[0.]],[[1.],[1.]])
        self.assertNested(post,[[.75],[.25]]);self.assertNested(new,[[[.75]],[[.25]]])

    def test_updates_preserve_orthogonal_key_directions(self):
        new,_=self.update();orthogonal=[-.8,.6]
        for old_expert,new_expert in zip(self.state,new):
            for old_row,new_row in zip(old_expert,new_expert):
                self.assertAlmostEqual(model.dot(old_row,orthogonal),model.dot(new_row,orthogonal),places=13)

    def test_unit_key_contracts_each_components_current_error(self):
        new,post=self.update()
        for e in range(3):
            for c in range(2):
                before=self.value[c]-model.dot(self.state[e][c],self.key)
                after=self.value[c]-model.dot(new[e][c],self.key)
                self.assertAlmostEqual(after,(1-self.beta[e][c]*post[e][c])*before,places=13)
                self.assertLessEqual(abs(after),abs(before)+1e-14)

    def test_large_key_counterexample(self):
        data=demo.report()['large_key_control']
        self.assertEqual(data['squared_error_before'],1.)
        self.assertEqual(data['squared_error_after'],9.)

    def test_expert_permutation_equivariance(self):
        order=[2,0,1];new,post=self.update()
        permuted,permuted_post=model.update([self.state[i] for i in order],self.key,self.value,
                                           [self.logits[i] for i in order],[self.beta[i] for i in order])
        self.assertNested(permuted,[new[i] for i in order]);self.assertNested(permuted_post,[post[i] for i in order])
        self.assertNested(model.read(permuted,self.query,[self.read_logits[i] for i in order]),model.read(new,self.query,self.read_logits))

    def test_output_coordinate_permutation_equivariance(self):
        new,post=self.update()
        a,b=model.update([list(reversed(e)) for e in self.state],self.key,list(reversed(self.value)),
                          [list(reversed(r)) for r in self.logits],[list(reversed(r)) for r in self.beta])
        self.assertNested(a,[list(reversed(e)) for e in new]);self.assertNested(b,[list(reversed(r)) for r in post])

    def test_key_basis_swap_preserves_predictions(self):
        new,post=self.update()
        a,b=model.update([[list(reversed(r)) for r in e] for e in self.state],list(reversed(self.key)),
                         self.value,self.logits,self.beta)
        self.assertNested(a,[[list(reversed(r)) for r in e] for e in new]);self.assertNested(b,post)
        self.assertNested(model.read(a,list(reversed(self.query)),self.read_logits),model.read(new,self.query,self.read_logits))

    def test_per_coordinate_logit_shift_invariance(self):
        new,post=self.update()
        shifted=[[r[0]+1000,r[1]-1000] for r in self.logits]
        a,b=self.update(write_logits=shifted)
        self.assertNested(a,new,11);self.assertNested(b,post,11)
        self.assertNested(model.read(new,self.query,[[r[0]-1000,r[1]+1000] for r in self.read_logits]),
                          model.read(new,self.query,self.read_logits),11)

    def test_extreme_finite_logits_are_stable(self):
        self.assertEqual(model.column_softmax([[1000.,-1000.],[-1000.,1000.]]),[[1.,0.],[0.,1.]])
        self.assertEqual(model.column_softmax([[1e308],[-1e308]]),[[1.],[0.]])

    def test_posterior_simplex_per_coordinate(self):
        _,post=self.update()
        for c in range(2):
            self.assertAlmostEqual(sum(r[c] for r in post),1.)
            self.assertTrue(all(0<=r[c]<=1 for r in post))

    def test_query_priors_control_read_independently(self):
        data=demo.report()['coordinate_responsibilities']
        a,b,c=[data[k][0] for k in ('read_with_supplied_query_priors','read_with_uniform_query_priors','read_with_reversed_query_priors')]
        self.assertGreater(a,b);self.assertGreater(b,c)
        self.assertAlmostEqual(b,(a+c)/2,places=13)

    def test_inputs_are_not_mutated(self):
        before=copy.deepcopy((self.state,self.key,self.value,self.logits,self.beta,self.query,self.read_logits))
        new,_=self.update();model.read(new,self.query,self.read_logits)
        self.assertEqual(before,(self.state,self.key,self.value,self.logits,self.beta,self.query,self.read_logits))

    def test_causal_prefix_invariance_to_future_token(self):
        tokens=[self.token(value=[.1*t,-.2]) for t in range(4)]
        _,prefix=model.run(self.state,tokens)
        _,extended=model.run(self.state,tokens+[self.token(value=[100.,-100.])])
        self.assertEqual(prefix,extended[:4])

    def test_chunked_scan_matches_whole_scan(self):
        tokens=[self.token(value=[.1*t,-.2]) for t in range(5)]
        state,outputs=model.run(self.state,tokens)
        middle,first=model.run(self.state,tokens[:2]);last,second=model.run(middle,tokens[2:])
        self.assertEqual(state,last);self.assertEqual(outputs,first+second)

    def test_sequence_order_can_change_state(self):
        a=self.token(value=[1.,0.]);b=self.token(value=[-1.,0.])
        first,_=model.run(self.state,[a,b]);second,_=model.run(self.state,[b,a])
        self.assertNotEqual(first,second)

    def test_invalid_shapes(self):
        for state in ([],[[]],[[[1.,2.]]],self.state+[[[0.]]]):
            with self.assertRaises(ValueError):self.update(state)
        for kwargs in ({'key':[1.]},{'value':[]},{'write_logits':[[0.]]},{'beta':[[1.]]}):
            with self.assertRaises(ValueError):self.update(**kwargs)
        with self.assertRaises(ValueError):model.read(self.state,[0.],self.read_logits)

    def test_invalid_nonfinite_and_learning_rates(self):
        for value in (float('nan'),float('inf'),True,'1',10**1000):
            with self.assertRaises(ValueError):self.update(key=[value,0.])
        for beta in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):self.update(beta=[[beta,1.]]*3)

    def test_unrepresentable_arithmetic_is_rejected(self):
        with self.assertRaises(ValueError):model.update([[[1e308]]],[1e308],[1.],[[0.]],[[1.]])
        with self.assertRaises(ValueError):model.update([[[0.]]],[1.],[1e308],[[0.]],[[1.]])

    def test_empty_scan_returns_copy(self):
        state,outputs=model.run(self.state,[])
        self.assertEqual(state,self.state);self.assertEqual(outputs,[])
        state[0][0][0]=10;self.assertNotEqual(state,self.state)

    def test_reference_python_version(self):
        self.assertEqual(platform.python_implementation(),'CPython')
        self.assertEqual(platform.python_version(),'3.14.5')

    def test_checked_output(self):
        self.assertEqual(demo.render(demo.report()),demo.SAMPLE.read_text())

    def test_cli(self):
        command=[sys.executable,'-m','examples.switching_linear_attention.demo']
        p=subprocess.run(command+['--check-sample'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,demo.SAMPLE.read_text())
        self.assertEqual(json.loads(p.stdout)['python_version'],'3.14.5')
        p=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,2)


if __name__=='__main__':
    unittest.main()
