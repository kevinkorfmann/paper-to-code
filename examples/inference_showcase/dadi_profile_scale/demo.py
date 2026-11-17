"""Original deterministic example of dadi's SFS scale profiling."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import dadi
import numpy as np

dadi.cuda_enabled(False)
HERE = Path(__file__).resolve().parent


def spectra(counts, shape, exclude_bins=()):
    """Build aligned unfolded spectra; inputs contain only bins 1,...,n-1.

    Only nonnegative integer counts and positive finite model entries accepted.
    Excluded bins are one-based derived-allele-count indexes. Monomorphic
    corners are always masked. This bounded example does not fold or project.
    """
    if len(counts) != len(shape) or len(counts) < 2:
        raise ValueError("aligned lists of at least two polymorphic bins required")
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("counts must be nonnegative integers")
    if any(isinstance(x, (bool, np.bool_)) or not math.isfinite(x) or x <= 0 for x in shape):
        raise ValueError("model entries must be finite and positive")
    if any(type(x) is not int or not 1 <= x <= len(counts) for x in exclude_bins):
        raise ValueError("excluded bins must be valid one-based integers")
    if len(set(exclude_bins)) != len(exclude_bins):
        raise ValueError("excluded bins must be unique")
    if sum(x for i,x in enumerate(counts,1) if i not in exclude_bins) == 0:
        raise ValueError("at least one unmasked observed site required")
    data = dadi.Spectrum([0, *counts, 0], mask_corners=True, data_folded=False)
    model = dadi.Spectrum([0, *shape, 0], mask_corners=True, data_folded=False)
    for i in exclude_bins:
        data.mask[i] = True
        model.mask[i] = True
    return model, data


def fit(counts, shape, exclude_bins=()):
    """Fit only a multiplicative model scale, conditional on shape and masks."""
    model, data = spectra(counts, shape, exclude_bins)
    scale = float(dadi.Inference.optimal_sfs_scaling(model, data))
    profile_ll = float(dadi.Inference.ll_multinom(model, data))
    means = model * scale
    return {
        "scale": scale,
        "profile_poisson_log_likelihood": profile_ll,
        "unscaled_poisson_log_likelihood": float(dadi.Inference.ll(model, data)),
        "active_bins": [i for i in range(1,len(counts)+1) if not data.mask[i]],
        "fitted_means": [None if data.mask[i] else float(means[i]) for i in range(1,len(counts)+1)],
    }


def clean(value):
    if isinstance(value,float):
        return round(value,9)
    if isinstance(value,list):
        return [clean(x) for x in value]
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()}
    return value


def report():
    fixture = json.loads((HERE/'synthetic_sfs.json').read_text())
    counts = fixture['counts_bins_1_to_4']
    shape = [1/i for i in range(1,5)]
    altered_counts = [120, *counts[1:]]
    return clean({
        'versions': {'python': platform.python_version(), 'dadi': importlib.metadata.version('dadi'), 'numpy': np.__version__},
        'seed': None,
        'input': fixture,
        'scale_controls': {str(mult): fit(counts, [mult*x for x in shape]) for mult in (0.1,1,10)},
        'different_shape_control': fit(counts, [1,1,1,1]),
        'singleton_sensitivity': {
            'altered_counts_bins_1_to_4': altered_counts,
            'all_bins': fit(altered_counts,shape),
            'exclude_singletons': fit(altered_counts,shape,exclude_bins=(1,)),
            'warning': 'Different masks change the data and target; do not compare their absolute scores as model evidence.'
        },
        'interpretation': 'Only a conditional SFS scale is fitted. Model shape is supplied, not estimated.'
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args = parser.parse_args()
    rendered=json.dumps(report(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.check_sample:
        if rendered != (HERE/'sample_output.json').read_text():
            raise SystemExit('Checked sample differs')
        print('Checked sample matches')
    else:
        print(rendered,end='')


if __name__=='__main__':
    main()
