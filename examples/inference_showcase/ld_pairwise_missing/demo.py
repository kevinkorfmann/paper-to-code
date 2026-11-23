"""Pairwise genotype deletion can make a signed LD matrix indefinite."""
from argparse import ArgumentParser
from itertools import combinations
import json
from pathlib import Path

import allel
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ("A", "B", "C")


def fixture():
    """Latent complete diploid dosages and the constructed observation mask."""
    same = [0]*4 + [2]*4
    opposite = [2]*4 + [0]*4
    middle = [1]*8
    # Blocks: AB (+), AC (-), BC (+), and a shared orthogonal four.
    full = np.array([
        same + same + middle + [0,0,2,2],
        same + middle + same + [0,2,0,2],
        middle + opposite + same + [0,2,2,0],
    ], dtype=np.int8)
    masked = full.copy()
    masked[2, :8] = -1
    masked[1, 8:16] = -1
    masked[0, 16:24] = -1
    return full, masked


def statistic(dosage):
    """Actual scikit-allel pairwise LD, overlap audit, and eigenvalues."""
    dosage = np.asarray(dosage)
    if (dosage.ndim != 2 or dosage.shape[0] != 3
            or not np.isin(dosage, (-1,0,1,2)).all()):
        raise ValueError("require three rows of biallelic dosages in {-1,0,1,2}")
    dosage = dosage.astype(np.int8)
    values = allel.rogers_huff_r(dosage)
    pairs = list(combinations(range(3), 2))
    matrix = np.eye(3)
    overlaps = {}
    for (i, j), r in zip(pairs, values, strict=True):
        overlap = np.sum((dosage[i] >= 0) & (dosage[j] >= 0))
        if overlap < 2 or not np.isfinite(r):
            raise ValueError("at least one correlation is undefined")
        matrix[i, j] = matrix[j, i] = r
        overlaps[NAMES[i] + NAMES[j]] = int(overlap)
    return {
        "pair_counts": overlaps,
        "signed_ld_matrix": np.round(matrix, 6).tolist(),
        "eigenvalues": np.round(np.linalg.eigvalsh(matrix), 6).tolist(),
    }


def report():
    full, masked = fixture()
    common = (masked >= 0).all(axis=0)
    return {
        "scikit_allel": allel.__version__,
        "numpy": np.__version__,
        "markers": NAMES,
        "common_complete_people": int(common.sum()),
        "masked_pairwise": statistic(masked),
        "same_complete_people_for_all_pairs": statistic(masked[:, common]),
        "synthetic_full_truth_restored": statistic(full),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = json.dumps(report(), indent=2, allow_nan=False) + "\n"
    if args.check_sample:
        if output.encode() != (HERE / "sample_output.json").read_bytes():
            raise SystemExit("output differs from checked sample")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
