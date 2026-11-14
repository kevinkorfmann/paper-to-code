import itertools
import math
import subprocess
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from .core import estimate, dense


def oracle(q, keys, values, labels, selected):
    # Expand every token, substituting its cluster's all-member means only
    # when it is not selected. Independent high-precision direct exponentials.
    with localcontext() as ctx:
        ctx.prec = 80
        q = Decimal(str(q))
        k = list(map(lambda x: Decimal(str(x)), keys))
        v = list(map(lambda x: Decimal(str(x)), values))
        num, den = Decimal(0), Decimal(0)
        for i in range(len(k)):
            group = [j for j in range(len(k)) if labels[j] == labels[i]]
            ki = k[i] if i in selected else sum(k[j] for j in group) / len(group)
            vi = v[i] if i in selected else sum(v[j] for j in group) / len(group)
            w = (q * ki).exp()
            num += w * vi
            den += w
        return float(num / den)


class Tests(unittest.TestCase):
    def test_decimal_all_subsets(self):
        for q in [-2., 0., .25, 2.]:
            for labels in [[0,0,0,0], [0,1,0,1], [0,0,1,2]]:
                for bits in itertools.product([False,True], repeat=4):
                    s = [i for i,b in enumerate(bits) if b]
                    got = estimate(q, [-2.,1.,3.,-.5], [4.,-1.,2.,8.], labels, s)
                    self.assertAlmostEqual(got.output, oracle(q, [-2.,1.,3.,-.5], [4.,-1.,2.,8.], labels, s), places=12)
    def test_rational_zero_query(self):
        vals = [Fraction(1,2), Fraction(7,2), Fraction(-2)]
        for bits in itertools.product([False,True], repeat=3):
            s = [i for i,b in enumerate(bits) if b]
            expected = sum(vals[i] if i in s else sum(vals)/3 for i in range(3))/3
            self.assertAlmostEqual(estimate(0.,[1.,2.,3.],list(map(float,vals)),[0,0,0],s).output,float(expected))
    def test_cancellation_nonmonotonic(self):
        self.assertEqual([estimate(0,[0,0],[0,2],[0,0],s).output for s in [(),(0,),(0,1)]],[1,.5,1])
    def test_exact_endpoint(self):
        self.assertAlmostEqual(dense(1,[-1,1],[-1,1]).output, math.tanh(1))
    def test_singletons(self):
        self.assertAlmostEqual(estimate(.7,[1,2,3],[4,5,1],[0,1,2]).output,dense(.7,[1,2,3],[4,5,1]).output)
    def test_counts(self):
        r=estimate(0,[0,0,0],[0,0,9],[0,0,1],[0])
        self.assertEqual(r.skipped_counts,(1,1)); self.assertEqual(r.output,3)
    def test_full_means_retained(self):
        self.assertEqual(estimate(0,[0,0],[0,2],[0,0],[0]).output,.5)
    def test_constant_values(self):
        for s in [(),(0,),(1,2)]:
            self.assertAlmostEqual(estimate(3,[1,2,3],[7,7,7],[0,0,1],s).output,7)
    def test_key_translation(self):
        a=estimate(2,[-1,2,3],[3,-2,1],[0,0,1],[0]).output
        b=estimate(2,[999,1002,1003],[3,-2,1],[0,0,1],[0]).output
        self.assertAlmostEqual(a,b)
    def test_value_affinity(self):
        a=estimate(.5,[-1,2,3],[3,-2,1],[0,0,1],[0]).output
        b=estimate(.5,[-1,2,3],[11,1,7],[0,0,1],[0]).output
        self.assertAlmostEqual(b,2*a+5)
    def test_permutation(self):
        self.assertAlmostEqual(estimate(1,[-1,2,3],[3,-2,1],[0,0,1],[0]).output,estimate(1,[3,-1,2],[1,3,-2],[1,0,0],[1]).output)
    def test_label_permutation(self):
        self.assertAlmostEqual(estimate(1,[1,2,3],[4,2,1],[0,0,1],[1]).output,estimate(1,[1,2,3],[4,2,1],[1,1,0],[1]).output)
    def test_stable_extreme(self):
        self.assertEqual(estimate(1,[10000,-10000],[3,7],[0,1]).output,3)
    def test_empty(self):
        with self.assertRaises(ValueError): estimate(0,[],[],[])
    def test_lengths(self):
        with self.assertRaises(ValueError): estimate(0,[1],[2,3],[0])
    def test_labels(self):
        for labels in [[1],[-1],[True],[.1]]:
            with self.assertRaises(ValueError): estimate(0,[1],[2],labels)
    def test_exact_indices(self):
        for s in [[0,0],[-1],[1],[True]]:
            with self.assertRaises(ValueError): estimate(0,[1],[2],[0],s)
    def test_nonfinite(self):
        for x in [float('nan'),float('inf'),float('-inf')]:
            with self.assertRaises(ValueError): estimate(x,[1],[2],[0])
            with self.assertRaises(ValueError): estimate(1,[x],[2],[0])
            with self.assertRaises(ValueError): estimate(1,[1],[x],[0])
    def test_overflow(self):
        with self.assertRaises(ValueError): estimate(1e308,[1e308],[1],[0])
    def test_residual_identity(self):
        q,k,v,lab,s=.8,[-2.,1.,3.],[4.,-1.,2.],[0,0,1],[0]
        r=estimate(q,k,v,lab,s); y=dense(q,k,v).output
        residual=0.
        for i in [1,2]:
            group=[j for j in range(3) if lab[j]==lab[i]]
            km=sum(k[j] for j in group)/len(group); vm=sum(v[j] for j in group)/len(group)
            w=math.exp(q*k[i]-r.shift); a=math.exp(q*km-r.shift)
            residual += w*v[i]-a*vm-y*(w-a)
        self.assertAlmostEqual(r.output-y,-residual/r.denominator)
    def test_sample(self):
        subprocess.run([sys.executable,'-m','examples.decoupled_compensation.demo','--check-sample'],check=True,capture_output=True)


if __name__=='__main__': unittest.main()
