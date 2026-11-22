"""Synthetic rank-reversal diagnostic using scikit-allel's native PBS."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import platform

import allel
import numpy as np

N_CHROMOSOMES = 20
WINDOW_SIZE = 3
WINDOW_NAMES = ("high_background", "low_background", "nonadditive", "equal_counts")
# Each row is a different synthetic biallelic SNP, with ALT counts in A/B/C.
ALT = (
    (19, 1, 14), (19, 2, 14), (18, 1, 13),
    (17, 6, 7), (17, 6, 8), (18, 7, 7),
    (10, 1, 19), (9, 2, 19), (11, 1, 18),
    (10, 10, 10), (9, 9, 9), (11, 11, 11),
)
SAMPLE = Path(__file__).with_name("sample_output.json")


def allele_counts(alt=ALT):
    a = np.asarray(alt, dtype=int)
    return [np.stack((N_CHROMOSOMES - col, col), axis=1) for col in a.T]


def summarize():
    ac = allele_counts()
    raw = allel.pbs(*ac, window_size=WINDOW_SIZE, normed=False)
    default = allel.pbs(*ac, window_size=WINDOW_SIZE)
    fst = np.stack([allel.moving_hudson_fst(ac[i], ac[j], size=WINDOW_SIZE)
                    for i, j in combinations(range(3), 2)], axis=1)
    distances = -np.log1p(-np.clip(fst, 0, .99999))
    normalizer = 1 + distances.sum(axis=1) / 2
    rows = []
    for i, name in enumerate(WINDOW_NAMES):
        rows.append({
            "window": name,
            "alt_counts_A_B_C": ALT[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE],
            "pairwise_hudson_fst_before_clip": [round(float(x), 9) for x in fst[i]],
            "normalizer": round(float(normalizer[i]), 9),
            "raw_pbs_A": round(float(raw[i]), 9),
            "default_pbsn1_A": round(float(default[i]), 9),
        })
    return {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_allel": allel.__version__},
        "seed": None, "called_haploid_chromosomes_per_population": N_CHROMOSOMES,
        "pair_order": ["AB", "AC", "BC"],
        "variants_per_window": WINDOW_SIZE,
        "top_raw_window": WINDOW_NAMES[int(np.argmax(raw))],
        "top_default_window": WINDOW_NAMES[int(np.argmax(default))],
        "windows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = json.dumps(summarize(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check_sample:
        if output.encode() != SAMPLE.read_bytes():
            parser.exit(1, "Sample output differs. Use the reference environment.\n")
        print("PBS sample output matches.")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
