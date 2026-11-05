"""Call scikit-allel, showing that copied sites are not independent evidence."""

import argparse
import json
import math
from pathlib import Path
import platform
import warnings

import allel
import numpy as np


HERE = Path(__file__).resolve().parent


def validate_counts(counts):
    """Require complete, biallelic counts at aligned sites in four populations.

    Shape is (sites, populations A/B/C/D, alleles REF/ALT). The wrapper
    deliberately rejects missing populations rather than silently omitting them.
    Positive sample sizes may differ between populations and sites.
    """
    x = np.asarray(counts)
    if x.ndim != 3 or x.shape[1:] != (4, 2) or len(x) == 0:
        raise ValueError("counts must have nonempty shape (sites, 4, 2)")
    if x.dtype.kind not in "iu" or np.any(x < 0):
        raise ValueError("counts must be nonnegative integers, not booleans")
    # Bound inputs before int64 conversion and summation.
    if np.any(x > 1_000_000_000):
        raise ValueError("each count must be at most one billion")
    x = np.array(x, dtype=np.int64, copy=True)
    if np.any(x.sum(axis=2) == 0):
        raise ValueError("every population must have called chromosomes at every site")
    return x


def site_components(counts):
    x = validate_counts(counts)
    return allel.patterson_d(*(x[:, p, :] for p in range(4)))


def estimate(counts, block_size):
    """Use actual tool estimates, retaining undefined values as None.

    Blocks are consecutive, equally sized groups of rows. Requiring exact
    divisibility prevents the tool's moving windows from dropping a final
    incomplete block. This is a strict pedagogical wrapper, not a VCF pipeline.
    """
    x = validate_counts(counts)
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be a positive integer")
    if block_size < 1 or len(x) % block_size or len(x) // block_size < 2:
        raise ValueError("require at least two complete equal-size blocks")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        num, den = allel.patterson_d(*(x[:, p, :] for p in range(4)))
        d, se, z, per_block, leave_one_out = allel.average_patterson_d(
            *(x[:, p, :] for p in range(4)), blen=int(block_size)
        )
    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else None
    return {
        "sites": len(x),
        "block_size": int(block_size),
        "blocks": len(x) // int(block_size),
        "numerator_sum": float(np.sum(num)),
        "denominator_sum": float(np.sum(den)),
        "d": finite(d),
        "jackknife_se": finite(se),
        "z": finite(z),
        "per_block_d": [finite(v) for v in per_block],
        "leave_one_block_out_d": [finite(v) for v in leave_one_out],
    }


def from_alt_counts(alt_counts, chromosomes=20):
    """Construct REF/ALT counts from small, explicit synthetic ALT counts."""
    alt = np.asarray(alt_counts, dtype=np.int64)
    return np.stack([chromosomes - alt, alt], axis=2)


def fixtures():
    # D(A,B;C,D) has the BABA-minus-ABBA sign for these labels.
    original = from_alt_counts([[20, 0, 20, 0]] * 6 + [[0, 20, 20, 0]] * 2)
    return {
        "original": original,
        "copies": np.repeat(original, 10, axis=0),
        "polymorphic": from_alt_counts([
            [16, 2, 18, 0], [12, 4, 16, 2], [10, 2, 12, 0], [18, 4, 14, 2],
            [2, 12, 18, 0], [4, 16, 12, 2], [14, 6, 10, 2], [8, 4, 16, 0],
        ]),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {key: rounded(v) for key, v in value.items()}
    return value


def run():
    data = fixtures()
    original, copies = data["original"], data["copies"]
    return rounded({
        "versions": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                     "scikit_allel": allel.__version__, "numpy": np.__version__},
        "randomness": "none; no seed is used",
        "population_order": ["A", "B", "C", "D"],
        "chromosomes_per_population": 20,
        "original_alt_counts": original[:, :, 1].tolist(),
        "polymorphic_alt_counts": data["polymorphic"][:, :, 1].tolist(),
        "original_eight_sites": estimate(original, 1),
        "ten_copies_kept_in_eight_blocks": estimate(copies, 10),
        "same_copies_treated_as_eighty_blocks": estimate(copies, 1),
        "swap_A_B": estimate(original[:, [1, 0, 2, 3], :], 1),
        "reverse_REF_ALT": estimate(original[:, :, ::-1], 1),
        "polymorphic_control": estimate(data["polymorphic"], 2),
        "homogeneous_zero_se_control": estimate(from_alt_counts([[20, 0, 20, 0]] * 8), 1),
        "uninformative_monomorphic_control": estimate(from_alt_counts([[0, 0, 0, 0]] * 8), 1),
        "interpretation": "Copying records adds no independent evidence. These constructed counts do not calibrate biological standard errors or establish gene flow.",
    })


def render():
    return json.dumps(run(), indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    actual = render()
    if args.check_sample:
        if actual != (HERE / "sample_output.json").read_text():
            raise SystemExit("checked output differs; inspect versions and values")
        print("sample output matches")
    else:
        print(actual, end="")


if __name__ == "__main__":
    main()
