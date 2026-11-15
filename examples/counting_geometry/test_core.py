import json
import math
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import unittest
from .core import classify, count_coordinate, prototypes, quantize, regression
from .demo import run

class TestGeometry(unittest.TestCase):
    def test_coordinate_reference(self):
        for g in (1,2,7,64):
            for n in range(80):
                self.assertEqual(count_coordinate(n,g),float(F(n,n+g)))
    def test_dense_attention(self):
        for n in range(20):
            weights=[1.0]*n+[2.0]
            values=[1.0]*n+[0.0]
            self.assertAlmostEqual(count_coordinate(n),sum(w*v for w,v in zip(weights,values))/sum(weights))
    def test_all_classes(self):
        for limit in (1,16,64,256):
            ts=prototypes(limit)
            self.assertEqual([classify(t,ts)[0] for t in ts],list(range(limit+1)))
    def test_rational_logit_oracle(self):
        ts=[F(n,n+2) for n in range(41)]
        for n,x in enumerate(ts):
            scores=[2*t*x-t*t for t in ts]
            self.assertEqual(max(range(len(ts)),key=scores.__getitem__),n)
            winner,margin=classify(float(x),[float(t) for t in ts])
            self.assertEqual(winner,n)
            self.assertAlmostEqual(margin,float(sorted(scores,reverse=True)[0]-sorted(scores,reverse=True)[1]),places=14)
    def test_spacing(self):
        for n in range(1,100):
            gap=F(n,n+2)-F(n-1,n+1)
            self.assertEqual(gap,F(2,(n+2)*(n+1)))
    def test_margin_decreases(self):
        margins=[classify(prototypes(n)[-1],prototypes(n))[1] for n in (8,16,32,64)]
        self.assertTrue(all(a>b for a,b in zip(margins,margins[1:])))
    def test_coefficient_bounds(self):
        for t in prototypes(1000):
            self.assertLess(2*t,2)
            self.assertLess(t*t,1)
    def test_midpoint_tie(self):
        self.assertEqual(classify(0.5,[0.,1.]),(0,0.))
    def test_midpoint_neighbors(self):
        self.assertEqual(classify(0.49,[0.,1.])[0],0)
        self.assertEqual(classify(0.51,[0.,1.])[0],1)
    def test_quantization_reference(self):
        for n in range(257):
            self.assertEqual(quantize(count_coordinate(n),8),float(F(round(F(n,n+2)*256),256)))
    def test_collision(self):
        self.assertEqual(quantize(count_coordinate(255),8),quantize(count_coordinate(256),8))
    def test_out_of_range(self):
        self.assertEqual(classify(count_coordinate(300),prototypes(256))[0],256)
    def test_distractor_collision(self):
        self.assertEqual(count_coordinate(1,2,0),count_coordinate(2,2,2))
    def test_regression_error(self):
        for n in range(65):
            self.assertAlmostEqual(n-regression(n,8192),float(F(n*n,n+8192)),places=12)
            self.assertLess(abs(n-regression(n,8192)),0.5)
    def test_regression_extrapolation(self):
        self.assertGreater(128-regression(128,8192),0.5)
    def test_invalid_counts(self):
        for n in (-1,True,1.5):
            with self.assertRaises(ValueError): count_coordinate(n)
    def test_invalid_gamma(self):
        for g in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError): count_coordinate(1,g)
    def test_invalid_limit(self):
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError): prototypes(n)
    def test_invalid_classifier(self):
        for x,ts in ((math.nan,[0]),(0,[]),(0,[math.inf])):
            with self.assertRaises(ValueError): classify(x,ts)
    def test_invalid_grid(self):
        for b in (0,53,True,1.5):
            with self.assertRaises(ValueError): quantize(0.5,b)
        for x in (-1,2,math.nan):
            with self.assertRaises(ValueError): quantize(x,8)
    def test_sample(self):
        expected=json.loads(Path(__file__).with_name('sample_output.json').read_text())
        self.assertEqual(run(),expected)
    def test_cli(self):
        output=subprocess.check_output([sys.executable,'-m','examples.counting_geometry.demo'],text=True)
        self.assertEqual(json.loads(output),run())
