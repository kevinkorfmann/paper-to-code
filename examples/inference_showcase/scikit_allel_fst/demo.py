"""Estimate Hudson F_ST from synthetic allele counts, not a demographic model."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import allel
import numpy as np

SEED = 20261026
N_LOCI = 1200
N_CHROMOSOMES = 80
BLOCK_SIZE = 60
BASE_FREQUENCIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
FIXTURE = Path(__file__).with_name("synthetic_counts.json")
SAMPLE = Path(__file__).with_name("sample_summary.json")


def generate_input(seed=SEED):
    """Sample independent haploid chromosomes at a preselected panel of loci."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    base = np.tile(BASE_FREQUENCIES, N_LOCI // len(BASE_FREQUENCIES))
    scenarios = {}
    for name, delta in (("null", 0.0), ("differentiated", 0.15)):
        p_a, p_b = base - delta, base + delta
        alt_a = rng.binomial(N_CHROMOSOMES, p_a)
        alt_b = rng.binomial(N_CHROMOSOMES, p_b)
        scenarios[name] = {
            "population_a": np.column_stack((N_CHROMOSOMES-alt_a, alt_a)).tolist(),
            "population_b": np.column_stack((N_CHROMOSOMES-alt_b, alt_b)).tolist(),
            "frequency_offset_for_evaluation_only": delta,
        }
    return {
        "seed": seed,
        "generator": "numpy PCG64; independent binomial counts at preselected loci",
        "loci": N_LOCI,
        "chromosomes_per_population": N_CHROMOSOMES,
        "base_frequencies_for_evaluation_only": list(BASE_FREQUENCIES),
        "scenarios": scenarios,
    }


def validate_counts(ac_a, ac_b):
    """Narrow, explicit input contract: aligned nonnegative biallelic counts."""
    arrays = [np.asarray(ac_a), np.asarray(ac_b)]
    for ac in arrays:
        if ac.ndim != 2 or ac.shape[1] != 2 or ac.shape[0] == 0:
            raise ValueError("counts must have nonempty shape (loci, 2)")
        if ac.dtype.kind not in "iu" or np.any(ac < 0):
            raise ValueError("counts must be nonnegative integers")
        # Bound before summing to avoid integer overflow on pathological input.
        if np.any(ac > 1_000_000_000):
            raise ValueError("counts exceed this example's supported range")
        if np.any(ac.astype(np.int64).sum(axis=1) < 2):
            raise ValueError("each population needs at least two calls at every locus")
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("population counts must have the same shape and locus order")
    return tuple(ac.astype(np.int64, copy=False) for ac in arrays)


def components(ac_a, ac_b):
    ac_a, ac_b = validate_counts(ac_a, ac_b)
    num, den = allel.hudson_fst(ac_a, ac_b)
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("nonfinite Hudson components")
    if np.sum(den) <= 0:
        raise ValueError("F_ST is undefined: no between-population diversity")
    return num, den


def estimate(ac_a, ac_b, block_size=BLOCK_SIZE):
    """Infer using counts alone. No simulation frequencies or target are read."""
    ac_a, ac_b = validate_counts(ac_a, ac_b)
    num, den = components(ac_a, ac_b)
    n = len(ac_a)
    if (type(block_size) is not int or block_size < 1 or
            n % block_size or n // block_size < 2):
        raise ValueError("use at least two complete, equally sized blocks")
    den_blocks = den.reshape(-1, block_size).sum(axis=1)
    if np.any(den_blocks <= 0):
        raise ValueError("every jackknife block must have between-population diversity")
    fst, se, block_values, leave_one_out = allel.average_hudson_fst(
        ac_a, ac_b, blen=block_size)
    if not np.all(np.isfinite([fst, se])):
        raise ValueError("nonfinite estimate or jackknife standard error")
    positive = den > 0
    return {
        "fst_ratio_of_sums": float(fst),
        "block_jackknife_se": float(se),
        "mean_of_per_locus_ratios_not_the_estimator": float(np.mean(num[positive]/den[positive])),
        "negative_per_locus_estimates": int(np.sum(num[positive] < 0)),
        "undefined_per_locus_ratios": int(np.sum(~positive)),
        "blocks": len(block_values),
        "block_values": block_values,
        "leave_one_block_out": leave_one_out,
    }


def population_target(base_frequencies, offset):
    """Evaluation only: ratio of population-frequency moments on this panel."""
    base = np.asarray(base_frequencies, dtype=float)
    p_a, p_b = base-offset, base+offset
    return float(np.sum((p_a-p_b)**2) / np.sum(p_a*(1-p_b)+p_b*(1-p_a)))


def aggregation_counterexample():
    """Constructed counts, not a second generative model or biological result."""
    ac_a = np.array([[80, 20]]*20 + [[99, 1]]*180)
    ac_b = np.array([[20, 80]]*20 + [[100, 0]]*180)
    rows = {}
    for name, stop in (("20_common_loci", 20), ("plus_180_rare_loci", 200)):
        num, den = components(ac_a[:stop], ac_b[:stop])
        rows[name] = {
            "ratio_of_sums": round(float(num.sum()/den.sum()), 9),
            "mean_of_per_locus_ratios": round(float(np.mean(num/den)), 9),
        }
    return rows


def build_report(data=None, block_size=BLOCK_SIZE):
    data = json.loads(FIXTURE.read_text()) if data is None else data
    scenarios = {}
    for name, scenario in data["scenarios"].items():
        result = estimate(scenario["population_a"], scenario["population_b"], block_size)
        scenarios[name] = {
            key: round(value, 9) if type(value) is float else value
            for key, value in result.items()
            if key not in {"block_values", "leave_one_block_out"}
        }
        scenarios[name]["population_target_for_evaluation_only"] = round(population_target(
            data["base_frequencies_for_evaluation_only"],
            scenario["frequency_offset_for_evaluation_only"]), 9)
    return {
        "example": "scikit-allel: Hudson F_ST and aggregation sensitivity",
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("scikit-allel", "numpy", "dask")}},
        "seed": data["seed"],
        "loci_per_scenario": data["loci"],
        "chromosomes_per_population": data["chromosomes_per_population"],
        "block_size_loci": block_size,
        "scenarios": scenarios,
        "constructed_aggregation_counterexample": aggregation_counterexample(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(block_size=args.block_size)
    except ValueError as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.check_sample and output != SAMPLE.read_text():
        raise SystemExit("sample output differs; use the pinned reference environment")
    print(output, end="")


if __name__ == "__main__":
    main()
