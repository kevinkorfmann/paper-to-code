"""Independent numerical and structural checks for the row relaxation example."""
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from .model import advance, diagnostics, log_probabilities, logits_from_probabilities, probabilities
from .demo import result

ROOT = Path(__file__).resolve().parents[2]


def decimal_step(u, s, t, tau=1., gamma=1., d=None):
    """Uncentered closed form, evaluated at 100 decimal digits then gauged."""
    with localcontext() as ctx:
        ctx.prec = 100
        u, s = [Decimal(str(x)) for x in u], [Decimal(str(x)) for x in s]
        t, tau, gamma = map(lambda x: Decimal(str(x)), (t, tau, gamma))
        d = [Decimal(str(x)) for x in (d or [0]*len(u))]
        k = gamma*tau
        r = (-k*t).exp()
        v = [r*a+(1-r)*b/tau+c/tau*(t-(1-r)/k) for a,b,c in zip(u,s,d)]
        m = max(v)
        return [float(x-m) for x in v]


def rk4(p, s, d, t, tau, gamma, steps=2000):
    """Independent integration directly on the probability simplex."""
    def rhs(q, time):
        gain = [a+b*time-tau*math.log(c) for a,b,c in zip(s,d,q)]
        mean = sum(a*b for a,b in zip(q,gain))
        return [gamma*a*(b-mean) for a,b in zip(q,gain)]
    h = t/steps
    p = p[:]
    for j in range(steps):
        k1 = rhs(p,j*h)
        k2 = rhs([a+h*b/2 for a,b in zip(p,k1)],(j+.5)*h)
        k3 = rhs([a+h*b/2 for a,b in zip(p,k2)],(j+.5)*h)
        k4 = rhs([a+h*b for a,b in zip(p,k3)],(j+1)*h)
        p = [a+h*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(p,k1,k2,k3,k4)]
    return p


class FlowTests(unittest.TestCase):
    def close(self, actual, expected, tol=2e-13):
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            self.assertTrue(math.isclose(a,b,rel_tol=tol,abs_tol=tol), (a,b))

    def test_decimal_constant_nonunit_parameters(self):
        for tau,gamma,t in [(1.,1.,.3),(.37,2.4,1.7),(3.2,.14,2.1)]:
            with self.subTest(tau=tau,gamma=gamma):
                self.close(advance([2.,-.3,1.],[-.7,1.2,.4],t,temperature=tau,mobility=gamma),decimal_step([2.,-.3,1.],[-.7,1.2,.4],t,tau,gamma))

    def test_decimal_affine_nonunit_parameters(self):
        for tau,gamma,t in [(.37,2.4,1.7),(3.2,.14,2.1),(1.,1.,.0001)]:
            with self.subTest(tau=tau,gamma=gamma):
                self.close(advance([2.,-.3,1.],[-.7,1.2,.4],t,temperature=tau,mobility=gamma,slope=[.2,-.5,1.]),decimal_step([2.,-.3,1.],[-.7,1.2,.4],t,tau,gamma,[.2,-.5,1.]))

    def test_tiny_affine_increment_decimal(self):
        # Huge slope makes the tiny O(dt**2) contribution visible.
        for t in [1e-14,1e-9,9.999e-5,1.001e-4]:
            with self.subTest(t=t):
                self.close(advance([0.,0.],[0.,0.],t,slope=[0.,-1/t**2]),decimal_step([0.,0.],[0.,0.],t,d=[0.,-1/t**2]),tol=4e-12)

    def test_rk4_constant_probability_coordinates(self):
        p,s=[.2,.3,.5],[.7,-.3,1.1]
        self.close(probabilities(advance(logits_from_probabilities(p),s,1.2,temperature=.6,mobility=1.7)),rk4(p,s,[0.]*3,1.2,.6,1.7),tol=2e-12)

    def test_rk4_affine_probability_coordinates(self):
        p,s,d=[.2,.3,.5],[.7,-.3,1.1],[-.2,.8,.3]
        self.close(probabilities(advance(logits_from_probabilities(p),s,1.2,temperature=.6,mobility=1.7,slope=d)),rk4(p,s,d,1.2,.6,1.7),tol=2e-12)

    def test_constant_semigroup(self):
        u,s=[.1,-.8,.5],[.3,1.2,-.6]
        options=dict(temperature=.4,mobility=2.3)
        self.close(advance(advance(u,s,.37,**options),s,.82,**options),advance(u,s,1.19,**options))

    def test_affine_semigroup_updates_score_origin(self):
        u,s,d=[.1,-.8,.5],[.3,1.2,-.6],[-.7,.2,.8]
        options=dict(temperature=2.1,mobility=.3,slope=d)
        first=advance(u,s,.37,**options)
        self.close(advance(first,[a+.37*b for a,b in zip(s,d)],.82,**options),advance(u,s,1.19,**options))

    def test_zero_slope_reduces_constant(self):
        self.close(advance([0.,-2.],[1.,3.],.7),advance([0.,-2.],[1.,3.],.7,slope=[0.,0.]))

    def test_zero_time_returns_centered_copy(self):
        u=[3.,1.]
        answer=advance(u,[1.,2.],0.,slope=[7.,-8.])
        self.assertEqual(answer,[0.,-2.]); self.assertIsNot(answer,u)

    def test_inputs_not_mutated(self):
        u,s,d=[3.,1.],[1.,2.],[7.,-8.]
        advance(u,s,.2,slope=d); diagnostics(u,s,slope=d)
        self.assertEqual((u,s,d),([3.,1.],[1.,2.],[7.,-8.]))

    def test_logit_score_and_slope_shift_invariance(self):
        u,s,d=[-.5,.8,1.1],[.7,-.2,.9],[.3,.2,-.8]
        want=advance(u,s,.7,temperature=.3,mobility=1.4,slope=d)
        self.close(advance([v+20 for v in u],[v-17 for v in s],.7,temperature=.3,mobility=1.4,slope=[v+13 for v in d]),want)

    def test_permutation_equivariance(self):
        u,s,d=[-.5,.8,1.1],[.7,-.2,.9],[.3,.2,-.8]
        perm=[2,0,1]
        out=advance(u,s,.7,slope=d)
        self.close(advance([u[i] for i in perm],[s[i] for i in perm],.7,slope=[d[i] for i in perm]),[out[i] for i in perm])

    def test_mobility_time_scaling_constant(self):
        self.close(advance([.1,-.2],[.3,.7],.9,mobility=2.3),advance([.1,-.2],[.3,.7],.9*2.3))

    def test_fixed_point_at_nonunit_temperature(self):
        s,tau=[1.,-2.,.7],.4
        u=[v/tau for v in s]
        self.close(probabilities(advance(u,s,3.,temperature=tau,mobility=.3)),probabilities(u))

    def test_large_time_converges_to_target(self):
        self.close(probabilities(advance([1.,-2.,.4],[.3,.7,-.5],1e5,temperature=.7,mobility=.2)),probabilities([v/.7 for v in [.3,.7,-.5]]))

    def test_large_affine_time_known_lag(self):
        u=advance([0.,0.],[0.,0.],1e5,temperature=2.,mobility=.25,slope=[0.,-1.])
        self.assertAlmostEqual(u[1],-(1e5-2)/2)

    def test_half_life_is_log_odds_not_probability_average(self):
        p=probabilities(advance([0.,0.],[0.,math.log(9)],math.log(2)))
        self.close(p,[.25,.75]); self.assertGreater(abs(p[1]-.7),.04)

    def test_fixed_score_kl_decreases(self):
        s=[1.,-.4,.3]; u=[-.2,.6,.1]
        values=[diagnostics(advance(u,s,t,temperature=.7,mobility=1.3),s,temperature=.7,mobility=1.3)['kl_to_target'] for t in [0.,.1,.3,.8,2.,5.]]
        self.assertTrue(all(a>b>=0 for a,b in zip(values,values[1:])))

    def test_energy_gap_equals_temperature_kl(self):
        u,s,tau=[-.2,.6,.1],[1.,-.4,.3],.7
        info=diagnostics(u,s,temperature=tau)
        minimum=-tau*math.log(sum(math.exp(v/tau) for v in s))
        self.assertAlmostEqual(info['free_energy']-minimum,info['free_energy_gap'],places=13)
        self.assertAlmostEqual(info['free_energy_gap'],tau*info['kl_to_target'],places=14)

    def test_energy_balance_finite_difference_affine(self):
        u,s,d=[-.2,.6,.1],[1.,-.4,.3],[.7,-.8,.1]
        tau,gamma,t,h=.7,1.3,.8,1e-5
        def energy(time):
            q=advance(u,s,time,temperature=tau,mobility=gamma,slope=d)
            return diagnostics(q,[a+time*b for a,b in zip(s,d)],temperature=tau)['free_energy']
        state=advance(u,s,t,temperature=tau,mobility=gamma,slope=d)
        info=diagnostics(state,[a+t*b for a,b in zip(s,d)],temperature=tau,mobility=gamma,slope=d)
        self.assertAlmostEqual((energy(t+h)-energy(t-h))/(2*h),info['energy_derivative'],places=9)
        self.assertAlmostEqual(info['energy_derivative'],info['score_work']-info['dissipation'],places=14)

    def test_probability_velocity_finite_difference(self):
        u,s=[-.2,.6,.1],[1.,-.4,.3]
        h=1e-6; tau=.7; gamma=1.3
        p0=probabilities(u); p1=probabilities(advance(u,s,h,temperature=tau,mobility=gamma))
        v=diagnostics(u,s,temperature=tau,mobility=gamma)['probability_derivative']
        self.close([(b-a)/h for a,b in zip(p0,p1)],v,tol=8e-7)
        self.assertAlmostEqual(sum(v),0.,places=14)

    def test_driven_energy_can_increase(self):
        u=advance([0.,0.],[0.,0.],1.,slope=[0.,-1.])
        info=diagnostics(u,[0.,-1.],slope=[0.,-1.])
        self.assertGreater(info['energy_derivative'],0.)
        self.assertGreater(info['free_energy'],diagnostics([0.,0.],[0.,0.])['free_energy'])

    def test_diagnostic_score_shift(self):
        u,s=[-.2,.6,.1],[1.,-.4,.3]
        a=diagnostics(u,s); b=diagnostics(u,[v+7 for v in s])
        self.assertAlmostEqual(a['free_energy']-7,b['free_energy'],places=13)
        for key in ['kl_to_target','free_energy_gap','dissipation']:
            self.assertAlmostEqual(a[key],b[key],places=13)
        self.close(a['probability_derivative'],b['probability_derivative'])

    def test_probability_round_trip(self):
        p=[.1,.2,.7]; self.close(probabilities(logits_from_probabilities(p)),p)

    def test_extreme_logits_retain_finite_log_mass(self):
        self.assertEqual(probabilities([0.,-1000.]),[1.,0.])
        self.assertEqual(log_probabilities([0.,-1000.]),[0.,-1000.])
        info=diagnostics([0.,-1000.],[0.,0.])
        self.assertTrue(all(math.isfinite(info[k]) for k in ['free_energy','kl_to_target','dissipation']))
        self.assertGreater(probabilities(advance([0.,-1000.],[0.,0.],10.))[1],.4)

    def test_validation_probability_rows(self):
        for p in [[],[1.],[0.,1.],[-.1,1.1],[.2,.3],[True,0.],[float('nan'),.5]]:
            with self.subTest(p=p), self.assertRaises(ValueError): logits_from_probabilities(p)

    def test_validation_vectors_and_shapes(self):
        for bad in [[],[1.],"01",[0.,True],[0.,float('inf')],[0.,float('nan')],[0.,10**1000]]:
            with self.subTest(bad=repr(bad)[:30]), self.assertRaises(ValueError): probabilities(bad)
        for fn in [lambda:advance([0.,0.],[0.,0.,0.],1.),lambda:advance([0.,0.],[0.,0.],1.,slope=[0.,0.,0.]),lambda:diagnostics([0.,0.],[0.,0.,0.]),lambda:diagnostics([0.,0.],[0.,0.],slope=[0.,0.,0.])]:
            with self.assertRaises(ValueError): fn()

    def test_validation_scalar_parameters(self):
        for parameter,values in [('dt',[-1.,True,float('inf'),float('nan'),'1']),('temperature',[0.,-1.,True,float('inf')]),('mobility',[0.,-1.,True,float('nan')])]:
            for value in values:
                with self.subTest(parameter=parameter,value=value),self.assertRaises(ValueError):
                    kwargs={'dt':1.,parameter:value}; advance([0.,0.],[0.,1.],**kwargs)
        for parameter in ['temperature','mobility']:
            with self.assertRaises(ValueError): diagnostics([0.,0.],[0.,1.],**{parameter:0.})

    def test_rejected_arithmetic_ranges(self):
        cases=[lambda:probabilities([-1e308,1e308]),lambda:advance([0.,0.],[0.,1.],1.,temperature=1e308,mobility=1e308),lambda:advance([0.,0.],[0.,1.],1.,temperature=1e-300,mobility=1e-300),lambda:advance([0.,0.],[0.,1.],1e-300,temperature=1e-100),lambda:diagnostics([0.,-1.],[1e308,-1e308])]
        for case in cases:
            with self.assertRaises(ValueError): case()

    def test_checked_sample_and_determinism(self):
        expected=Path(__file__).with_name('sample_output.json').read_text()
        self.assertEqual(json.dumps(result(),indent=2,allow_nan=False)+'\n',expected)
        self.assertEqual(result(),result())

    def test_cli_default(self):
        run=subprocess.run([sys.executable,'-m','examples.fisher_rao_attention.demo'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(run.stdout,Path(__file__).with_name('sample_output.json').read_text()); self.assertEqual(run.stderr,'')

    def test_cli_check_sample(self):
        run=subprocess.run([sys.executable,'-m','examples.fisher_rao_attention.demo','--check-sample'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(run.stdout,'Sample output matches.\n')

    def test_cli_rejects_unknown_argument(self):
        run=subprocess.run([sys.executable,'-m','examples.fisher_rao_attention.demo','--unknown'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,2); self.assertIn('unrecognized arguments',run.stderr)


if __name__=='__main__':
    unittest.main()
