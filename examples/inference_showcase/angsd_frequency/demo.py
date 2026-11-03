"""Reproduce actual ANGSD output and independent analytical controls."""
import argparse
import json
from pathlib import Path
import platform

from .install import REVISION
from .model import (constructed_sites, hard_call_frequency, log_likelihood,
                    normalized_likelihood_dosage, repeated_heterozygote_mle, run_angsd)


def run_demo():
    sites = constructed_sites()
    actual = run_angsd(sites)
    main = {}
    for name, rows in sites[:2]:
        exact = repeated_heterozygote_mle(rows[0][0], rows[0][2])
        main[name] = {**actual[name], 'exact_population_mle': round(exact, 9),
                      'hard_call_alt_frequency': hard_call_frequency(rows),
                      'normalized_gl_dosage_frequency': round(normalized_likelihood_dosage(rows), 9),
                      'log_likelihood_loss_at_hard_call': round(log_likelihood(rows, exact) -
                                                               log_likelihood(rows, 0.5), 9)}
    return {'versions': {'python': platform.python_version(), 'angsd': '0.941', 'htslib': '1.24',
                         'angsd_revision': REVISION},
            'data': {'diploids': 20, 'sites': len(sites), 'seed': None,
                     'origin': 'Deterministic constructed likelihoods; no biological sample data'},
            'main_comparison': main,
            'controls': {**{name: actual[name] for name in actual if name not in main},
                         'all_uninformative': {'emitted': 'all_uninformative' in actual,
                                               'likelihood': 'constant for every p in [0,1]'}},
            'scope': 'Population ALT frequency conditional on supplied likelihoods and diploid HWE; not sample allele count or a calibrated uncertainty interval'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    output = json.dumps(run_demo(), indent=2, sort_keys=True) + '\n'
    if args.check_sample:
        expected = Path(__file__).with_name('sample_output.json').read_text()
        if output != expected:
            raise SystemExit('Output differs from checked sample')
    print(output, end='')


if __name__ == '__main__':
    main()
