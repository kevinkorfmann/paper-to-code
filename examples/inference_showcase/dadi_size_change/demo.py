"""Fit one population-size ratio with dadi, holding the change time fixed."""

import argparse
import importlib.metadata
import json
from math import isfinite
from pathlib import Path

import dadi
import numpy as np

# This example uses only dadi's CPU diffusion solver.
dadi.cuda_enabled(False)
MODEL = dadi.Numerics.make_extrap_log_func(dadi.Demographics1D.two_epoch)
FIXTURE = Path(__file__).with_name("synthetic_sfs.json")
SEED, SAMPLE_SIZE = 20261023, 20
TRUE_NU, TRUE_TIME, TRUE_THETA = 3.0, 0.2, 500.0
GRIDS, REFINED_GRIDS = (40, 50, 60), (60, 70, 80)
STARTS = (0.5, 2.0, 6.0)


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def spectrum(nu, time, sample_size=SAMPLE_SIZE, grids=GRIDS):
    """Expected unfolded SFS for theta=1; monomorphic corners are masked."""
    nu, time = positive(nu, "nu"), positive(time, "time")
    if type(sample_size) is not int or sample_size < 4:
        raise ValueError("sample_size must be an integer of at least four")
    if (len(grids) != 3 or any(type(g) is not int or g <= sample_size for g in grids)
            or not grids[0] < grids[1] < grids[2]):
        raise ValueError("three increasing integer grids larger than sample_size are required")
    result = MODEL([nu, time], [sample_size], list(grids))
    if not np.all(np.isfinite(result[1:-1])) or not np.all(result[1:-1] > 0):
        raise ValueError("diffusion solver returned an invalid spectrum")
    return result


def generate_input(seed=SEED):
    """Independent Poisson SFS-bin counts, not a sequence or coalescent sample."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    mean = TRUE_THETA*spectrum(TRUE_NU, TRUE_TIME)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = [0, *map(int, rng.poisson(mean[1:-1])), 0]
    return {
        "generator": "independent Poisson counts; numpy PCG64; dadi two_epoch expectation",
        "seed": seed, "sample_size_haploid": SAMPLE_SIZE,
        "unfolded_counts_including_masked_corners": counts,
        "truth_for_evaluation_only": {"nu": TRUE_NU, "time_2Nanc_generations": TRUE_TIME,
                                     "theta": TRUE_THETA},
        "generating_grids": list(GRIDS),
    }


def load_sfs(data):
    """Read counts and sample size only: generation parameters are not fit inputs."""
    n = data["sample_size_haploid"]
    counts = data["unfolded_counts_including_masked_corners"]
    if type(n) is not int or n < 4 or len(counts) != n+1:
        raise ValueError("counts must have n+1 entries for integer sample size n >= 4")
    if any(type(c) is not int or c < 0 for c in counts):
        raise ValueError("observed counts must be nonnegative integers")
    if sum(counts[1:-1]) == 0:
        raise ValueError("at least one segregating site is required")
    return dadi.Spectrum(counts, mask_corners=True, data_folded=False)


def fit(data, time=TRUE_TIME, grids=GRIDS, starts=STARTS):
    """Profile theta and optimize nu in [0.2, 10], conditional on time.

    NLopt's deterministic BOBYQA is called through dadi, from three starts.
    log_opt=False optimizes nu directly. The supplied Spectrum may contain
    fractional expected counts for the explicitly labelled noiseless control.
    """
    time = positive(time, "time")
    if not 0.001 <= time <= 2:
        raise ValueError("this demo supports fixed times in [0.001, 2]")
    if (data.ndim != 1 or data.folded or len(data) < 5 or not data.mask[0]
            or not data.mask[-1] or np.any(data.mask[1:-1])
            or not np.all(np.isfinite(data.compressed()))
            or np.any(data.compressed() < 0) or data.sum() <= 0):
        raise ValueError("an unfolded nonempty 1D SFS with only the corners masked is required")
    if not starts or any(not 0.2 < positive(s, "start") < 10 for s in starts):
        raise ValueError("at least one starting nu strictly inside [0.2, 10] is required")
    n = len(data)-1
    spectrum(1, time, n, grids)  # Validate grid and solver output before optimization.
    runs = []
    for start in starts:
        params, reported_ll = dadi.NLopt_mod.opt(
            [start, time], data, MODEL, list(grids), multinom=True,
            lower_bound=[0.2, 0.001], upper_bound=[10.0, 2.0],
            fixed_params=[None, time], log_opt=False,
            ftol_abs=1e-9, xtol_abs=1e-8, maxeval=150,
        )
        if not np.all(np.isfinite(params)) or not isfinite(reported_ll):
            raise RuntimeError("dadi optimizer failed to return a finite fit")
        model = spectrum(float(params[0]), time, n, grids)
        ll = float(dadi.Inference.ll_multinom(model, data))
        if not np.isclose(ll, reported_ll, atol=1e-6, rtol=0):
            raise RuntimeError("optimizer likelihood disagrees with the returned parameters")
        runs.append({"start_nu": start, "nu": float(params[0]),
                     "theta": float(dadi.Inference.optimal_sfs_scaling(model, data)),
                     "profile_poisson_log_likelihood": ll})
    best = max(runs, key=lambda r: r["profile_poisson_log_likelihood"])
    if max(r["nu"] for r in runs)-min(r["nu"] for r in runs) > 1e-4:
        raise RuntimeError("starting points did not agree; inspect the likelihood surface")
    if not 0.2001 < best["nu"] < 9.9999:
        raise RuntimeError("optimum is at a bound; this small example needs a wider model review")
    return {"fixed_time_2Nanc_generations": time, "grids": list(grids),
            "nu": best["nu"], "theta": best["theta"],
            "profile_poisson_log_likelihood": best["profile_poisson_log_likelihood"],
            "constant_size_profile_log_likelihood": float(dadi.Inference.ll_multinom(
                spectrum(1, time, n, grids), data)),
            "nu_from_each_start": [r["nu"] for r in runs]}


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return round(value, 6) if type(value) is float else value


def build_report(data=None, time=TRUE_TIME):
    data = json.loads(FIXTURE.read_text()) if data is None else data
    observed = load_sfs(data)
    fitted = fit(observed, time)
    wrong_time = fit(observed, 0.8)
    refined = fit(observed, time, REFINED_GRIDS)
    # Separate same-solver diagnostic: fractional expected counts have no Poisson noise.
    noiseless = fit(TRUE_THETA*spectrum(TRUE_NU, TRUE_TIME), TRUE_TIME)
    return rounded({
        "example": "dadi: estimate a size ratio conditional on a fixed change time",
        "versions": {name: importlib.metadata.version(name) for name in ["dadi", "numpy", "scipy", "nlopt"]},
        "generator_seed": data["seed"], "haploid_sample_size": len(observed)-1,
        "segregating_sites": int(observed.sum()),
        "input_counts_bins_1_to_n_minus_1": list(map(int, observed[1:-1])),
        "optimizer": "dadi NLopt BOBYQA; direct nu; starts 0.5, 2, 6; bounds 0.2 to 10",
        "fit": fitted, "assumed_time_0_8_control": wrong_time,
        "refined_grid_fit": refined, "same_solver_noiseless_control": noiseless,
        "grid_refinement_relative_nu_change": abs(refined["nu"]/fitted["nu"]-1),
        "interpretation": "nu is Ncurrent/Nancestral, conditional on an assumed time. "
                          "Theta is a profiled nuisance scale. In the default checked run, "
                          "the wrong-time control changes the answer and fits better. This "
                          "does not identify a unique history or absolute population size.",
    })


def check_sample(actual, expected):
    """Allow 2e-5 absolute/relative float tolerance across CPU libraries."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("report keys differ")
        for key in expected:
            check_sample(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("report lengths differ")
        for a, b in zip(actual, expected):
            check_sample(a, b)
    elif type(expected) is float:
        if not np.isclose(actual, expected, rtol=2e-5, atol=2e-5):
            raise AssertionError(f"numerical mismatch: {actual} vs {expected}")
    elif actual != expected:
        raise AssertionError(f"sample mismatch: {actual!r} vs {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=TRUE_TIME,
                        help="assumed change time, in 2*Nancestral generations")
    parser.add_argument("--generate-input", action="store_true")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    if args.generate_input and args.check_sample:
        parser.error("choose input generation or sample checking")
    try:
        report = generate_input() if args.generate_input else build_report(time=args.time)
        if args.check_sample:
            check_sample(report, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))
    except (ValueError, RuntimeError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
