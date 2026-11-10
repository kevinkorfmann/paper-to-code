"""Deterministic three-token constructed example; no training or random sampling."""
import argparse
import json
from pathlib import Path
from .core import diffuse, weights


def result():
    q = [[1,0,0],[0,1,0],[0,0,1]]
    # S sends 0->1->2->2. Columns of S are rows of K.
    k = [[0,0,0],[1,0,0],[0,1,1]]
    v = [[0],[0],[1]]
    return {
        "environment": "CPython 3.14.5; standard library; deterministic",
        "beta": 0.5,
        "retained_hops": [0,1,2],
        "conditioned_length_weights": weights(0.5,3),
        "one_hop_value_at_token_0": 0,
        "conditioned_diffusion": [row[0] for row in diffuse(q,k,v)],
        "forced_stop_at_cap_weights": [0.5,0.25,0.25],
        "forced_stop_value_at_token_0": 0.25,
        "infinite_geometric_value_at_token_0": 0.25,
        "constant_value_exact_normalization": [r[0] for r in diffuse(q,k,[[1],[1],[1]])],
        "constant_value_epsilon_1e_6": [r[0] for r in diffuse(q,k,[[1],[1],[1]],epsilon=1e-6)],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample',action='store_true')
    args=parser.parse_args()
    text=json.dumps(result(), indent=2, allow_nan=False)+'\n'
    if args.check_sample:
        expected=Path(__file__).with_name('sample_output.json').read_text()
        if text != expected:
            raise SystemExit('sample output mismatch')
        print('Sample output matches.')
    else:
        print(text,end='')


if __name__=='__main__':
    main()
