"""Infer symmetric migration conditional on a split time with moments-popgen."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import moments
import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "synthetic_sfs.json"
SAMPLE = HERE / "sample_summary.json"
SEED = 20261027
SAMPLE_SIZES = (12, 12)
TIME = 0.5
PADDING = 8
DT = 0.02
STARTS = (0.2, 1.0, 4.0)
BOUNDS = (0.01, 10.0)


def expected_sfs(migration, sample_sizes=SAMPLE_SIZES, time=TIME,
                 padding=PADDING, dt=DT):
    """Expected unfolded SFS at theta=1; M=2*Nref*m, T=t/(2*Nref).

    Both descendant sizes equal the ancestral reference size. Integrating a
    larger spectrum before projection is a moment-closure sensitivity control.
    This calls installed moments APIs; no upstream solver is vendored.
    """
    ns = tuple(sample_sizes)
    if len(ns) != 2 or any(type(n) not in (int, np.int64) or not 4 <= n <= 40 for n in ns):
        raise ValueError("sample sizes must be two integers between 4 and 40")
    if type(padding) is not int or not 0 <= padding <= 32:
        raise ValueError("padding must be an integer between 0 and 32")
    if (not math.isfinite(migration) or not 0 <= migration <= 10 or
            not math.isfinite(time) or not 0 <= time <= 2 or
            not math.isfinite(dt) or not 0.001 <= dt <= 0.05):
        raise ValueError("require 0<=M<=10, 0<=T<=2, and 0.001<=dt<=0.05")
    enlarged = [n + padding for n in ns]
    fs = moments.Demographics2D.snm(enlarged)
    if time:
        fs.integrate([1.0, 1.0], time, dt_fac=dt,
                     m=np.array([[0.0, migration], [migration, 0.0]]))
    if padding:
        fs = fs.project(list(ns))
    if np.any(~np.isfinite(fs.compressed())) or np.any(fs.compressed() <= 0):
        raise RuntimeError("model produced nonpositive or nonfinite unmasked bins")
    return fs


def observed_sfs(counts):
    """Counts only; fractional expected counts are allowed for noiseless checks."""
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 2 or any(not 5 <= size <= 41 for size in arr.shape):
        raise ValueError("counts need two dimensions, with 4..40 chromosomes per population")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1e9):
        raise ValueError("counts must be finite, nonnegative, and at most 1e9 per bin")
    if arr[0, 0] != 0 or arr[-1, -1] != 0:
        raise ValueError("globally monomorphic corner bins must be zero")
    if arr.sum() == 0:
        raise ValueError("at least one segregating site is required")
    return moments.Spectrum(arr, mask_corners=True)


def fit(counts, time=TIME, padding=PADDING, dt=DT, starts=STARTS):
    """Estimate M, profiling theta; takes no truth or generation metadata."""
    if not math.isfinite(time) or not 0 < time <= 2:
        raise ValueError("fitting needs 0<T<=2; M is unidentifiable at T=0")
    if not starts or any(not math.isfinite(x) or not BOUNDS[0] < x < BOUNDS[1] for x in starts):
        raise ValueError("provide interior positive starting values")
    data = observed_sfs(counts)

    def model(params, ns):
        return expected_sfs(float(params[0]), tuple(int(n) for n in ns), time, padding, dt)

    rows = []
    for start in starts:
        params, neg_ll, _, _, flag = moments.Inference.optimize_log_fmin(
            [start], data, model, lower_bound=[BOUNDS[0]], upper_bound=[BOUNDS[1]],
            full_output=True, maxiter=150, maxfun=300)
        m = float(params[0])
        if flag or not math.isfinite(neg_ll) or not BOUNDS[0]*1.01 < m < BOUNDS[1]*0.99:
            raise RuntimeError("optimization did not converge to an interior solution")
        fs = model(params, data.sample_sizes)
        rows.append({"start": start, "M": m,
                     "theta": float(moments.Inference.optimal_sfs_scaling(fs, data)),
                     "profile_log_likelihood": float(moments.Inference.ll_multinom(fs, data)),
                     "optimizer_warnflag": int(flag)})
    if max(row["M"] for row in rows) - min(row["M"] for row in rows) > 0.001:
        raise RuntimeError("starting values converged to different estimates")
    best = max(rows, key=lambda row: row["profile_log_likelihood"])
    return {"M": best["M"], "theta": best["theta"],
            "profile_log_likelihood": best["profile_log_likelihood"],
            "assumed_T": time, "padding_per_population": padding, "dt_fac": dt,
            "starts": rows}


def independent_profile_ll(model, counts):
    """Scalar math oracle, including Poisson constants (no moments likelihood)."""
    data = np.asarray(counts, dtype=float)
    valid = ~np.ma.getmaskarray(model)
    expected = np.asarray(model)[valid]
    observed = data[valid]
    scale = math.fsum(observed) / math.fsum(expected)
    ll = math.fsum(float(k)*math.log(scale*float(v)) - scale*float(v)
                   - math.lgamma(float(k)+1) for k, v in zip(observed, expected))
    return scale, ll


def generate_fixture(seed=SEED):
    fs = expected_sfs(1.0)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros(fs.shape, dtype=np.int64)
    counts[~fs.mask] = rng.poisson(500.0*fs.compressed())
    return {"sample_sizes": list(SAMPLE_SIZES), "counts": counts.tolist(),
            "generation_for_evaluation_only": {
                "seed": seed, "rng": "NumPy PCG64", "M": 1.0, "T": TIME,
                "theta": 500.0, "padding_per_population": PADDING, "dt_fac": DT,
                "sampling": "independent Poisson bins; not a simulated genome"}}


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return round(value, 9) if type(value) is float else value


def report(time=TIME):
    fixture = json.loads(FIXTURE.read_text())
    counts = fixture["counts"]
    primary = fit(counts, time=time)
    wrong_time = fit(counts, time=1.0)
    finer_time = fit(counts, time=time, dt=0.01)
    larger_spectrum = fit(counts, time=time, padding=16)
    noiseless_counts = (500.0*expected_sfs(1.0)).filled(0)
    noiseless = fit(noiseless_counts)
    zero_m = expected_sfs(0.0, time=time)
    fs = expected_sfs(primary["M"], time=time)
    oracle_scale, oracle_ll = independent_profile_ll(fs, counts)
    data = observed_sfs(counts)
    return rounded({
        "example": "moments-popgen: symmetric migration conditional on split time",
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("moments-popgen", "numpy", "scipy")}},
        "sample_sizes": fixture["sample_sizes"], "segregating_sites": int(data.S()),
        "generation_for_evaluation_only": fixture["generation_for_evaluation_only"],
        "fits": {"observed": primary, "assumed_T_1": wrong_time,
                 "halved_time_step": finer_time, "larger_integrated_spectrum": larger_spectrum,
                 "noiseless_same_solver": noiseless},
        "checks": {
            "zero_migration_profile_log_likelihood": float(moments.Inference.ll_multinom(zero_m, data)),
            "independent_theta": oracle_scale, "independent_profile_log_likelihood": oracle_ll,
            "halved_time_step_relative_M_change": abs(finer_time["M"]/primary["M"]-1),
            "larger_spectrum_relative_M_change": abs(larger_spectrum["M"]/primary["M"]-1)},
    })


def assert_close(actual, expected, path="root"):
    """Regression tolerance for CPU/optimizer variation; never a confidence bound."""
    if type(expected) is float:
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=2e-4, abs_tol=2e-4):
            raise AssertionError(f"sample mismatch at {path}: {actual} != {expected}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"sample keys differ at {path}")
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"sample list differs at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"sample mismatch at {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=TIME, help="assumed split time, in 2*Nref generations")
    parser.add_argument("--generate-input", action="store_true")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    if args.check_sample and (args.generate_input or args.time != TIME):
        parser.error("--check-sample requires the default experiment")
    try:
        result = generate_fixture() if args.generate_input else report(time=args.time)
        if args.check_sample:
            assert_close(result, json.loads(SAMPLE.read_text()))
    except (ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
