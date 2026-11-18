"""Conditional ancestral-state error inference with dadi, not demography fitting."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import dadi
import numpy as np
HERE = Path(__file__).resolve().parent


def spectrum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Provide at least two finite positive polymorphic-bin values')
    return dadi.Spectrum(np.r_[0., values, 0.])


def misorient(shape, rate):
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Misidentification probability must be in [0,1]')
    return dadi.Numerics.apply_anc_state_misid(spectrum(shape), rate)


def scan(counts, shape, *, folded=False):
    if len(counts) != len(shape):
        raise ValueError('Counts and model must have matching sample sizes')
    data = spectrum(counts)
    if folded: data = data.fold()
    rates = np.linspace(0, .5, 51)
    scores = []
    for rate in rates:
        model = misorient(shape, float(rate))
        if folded: model = model.fold()
        scores.append(float(dadi.Inference.ll_multinom(model, data)))
    best = max(scores)
    maximizers = [round(float(r), 2) for r, score in zip(rates, scores) if abs(score-best) < 1e-9]
    return {'grid_maximizers': maximizers, 'log_likelihood_range': max(scores)-min(scores),
            'log_likelihood_at_zero': scores[0], 'maximum_log_likelihood': best,
            'likelihood_scores': scores}


def clean(x):
    if isinstance(x, float): return round(x, 9)
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    return x


def report():
    fixture = json.loads((HERE/'synthetic_sfs.json').read_text())
    shape, counts = fixture['supplied_shape'], fixture['observed_counts']
    return clean({'versions': {'python': platform.python_version(), 'dadi': importlib.metadata.version('dadi'), 'numpy':np.__version__},
      'seed': None, 'fixture': fixture,
      'unfolded': scan(counts,shape), 'folded': scan(counts,shape,folded=True),
      'symmetric_shape_control': scan(counts,[1,1,1,1]),
      'folded_counts': spectrum(counts).fold().compressed().tolist(),
      'folded_supplied_shape': spectrum(shape).fold().compressed().tolist(),
      'zero_error_control': scan(shape,shape)['grid_maximizers'],
      'interpretation':'Grid fit of one error probability conditional on a supplied spectrum shape. Folding destroys orientation information. No demographic inference or confidence interval.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    output=json.dumps(report(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.check_sample:
        if output != (HERE/'sample_output.json').read_text(): raise SystemExit('Sample output differs')
        print('Checked sample matches')
    else: print(output,end='')
if __name__=='__main__': main()
