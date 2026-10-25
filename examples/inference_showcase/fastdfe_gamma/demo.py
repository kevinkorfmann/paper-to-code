"""Infer a deleterious gamma DFE and inspect sensitivity to target-site counts."""

import argparse
from functools import lru_cache
import importlib.metadata
import json
import logging
from math import isfinite
from pathlib import Path

import fastdfe as fd
import numpy as np

# No worker pools, bootstrap jobs, plotting, or sequence-data downloads.
fd.Settings.disable_pbar = True
fd.Settings.parallelize = False
logging.getLogger("fastdfe").setLevel(logging.WARNING)

FIXTURE = Path(__file__).with_name("synthetic_spectra.json")
SEED, N = 20261025, 20
NEUTRAL_TARGET, SELECTED_TARGET = 1_000_000, 2_000_000
THETA = 0.001
TRUTH = {"S_d": -20.0, "b": 0.5}
FIXED = {"p_b": 0.0, "S_b": 1.0, "eps": 0.0, "h": 0.5}
STARTS = ((-5.0, 0.8), (-50.0, 0.3), (-150.0, 1.2))
RESOLUTION, REFINED_RESOLUTION = 400, 800


@lru_cache(maxsize=4)
def discretization(n=N, resolution=RESOLUTION):
    """Cache only the tool's numerical kernel; never cache fitted answers."""
    if type(n) is not int or not 4 <= n <= 100:
        raise ValueError("sample size must be an integer from 4 to 100")
    if type(resolution) is not int or not 100 <= resolution <= 1600:
        raise ValueError("resolution must be an integer from 100 to 1600")
    return fd.Discretization(
        n=n, h=0.5, intervals_del=(-1e8, -1e-6, resolution),
        intervals_ben=(1e-6, 1e3, resolution),
        integration_mode="midpoint", linearized=True, parallelize=False,
    )


def expected_spectra():
    """Same-tool expectation, not a forward or coalescent simulation."""
    neutral = fd.Simulation.get_neutral_sfs(n=N, n_sites=NEUTRAL_TARGET, theta=THETA)
    selected_at_neutral_target = fd.Simulation(
        sfs_neut=neutral, params=TRUTH | FIXED, model=fd.GammaExpParametrization(),
        discretization=discretization(), parallelize=False,
    ).run()
    counts = selected_at_neutral_target.polymorphic * SELECTED_TARGET/NEUTRAL_TARGET
    selected = fd.Spectrum([SELECTED_TARGET-float(sum(counts)), *counts, 0])
    return neutral, selected


def generate_input(seed=SEED):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    neutral, selected = expected_spectra()
    rng = np.random.Generator(np.random.PCG64(seed))
    return {
        "generator": "independent Poisson polymorphic-bin counts around fastDFE expectations",
        "seed": seed, "haploid_sample_size": N,
        "neutral_target_sites": NEUTRAL_TARGET, "selected_target_sites": SELECTED_TARGET,
        "neutral_polymorphic_counts": list(map(int, rng.poisson(neutral.polymorphic))),
        "selected_polymorphic_counts": list(map(int, rng.poisson(selected.polymorphic))),
        "truth_for_evaluation_only": TRUTH | {"theta_per_site": THETA},
        "generating_resolution": RESOLUTION,
    }


def observed_spectra(data, selected_target_factor=1.0):
    """Read observed counts/targets only. Preserve counts when changing the target."""
    factor = selected_target_factor
    if isinstance(factor, bool) or not isfinite(factor) or factor <= 0:
        raise ValueError("selected_target_factor must be positive and finite")
    n = data["haploid_sample_size"]
    if type(n) is not int or not 4 <= n <= 100:
        raise ValueError("haploid_sample_size must be an integer from 4 to 100")
    output = []
    for category, scale in (("neutral", 1.0), ("selected", factor)):
        total = data[f"{category}_target_sites"]
        counts = data[f"{category}_polymorphic_counts"]
        if type(total) is not int or total <= 0:
            raise ValueError("target sites must be positive integers")
        if len(counts) != n-1 or any(type(c) is not int or c < 0 for c in counts):
            raise ValueError("provide n-1 nonnegative integer polymorphic counts")
        if min(counts) == 0 and category == "neutral":
            raise ValueError("this small example requires a nonzero neutral count in every bin")
        if sum(counts) == 0 or total*scale <= sum(counts):
            raise ValueError("target sites must exceed a positive number of polymorphic sites")
        # Bin zero stores unmutated target sites. Fixed-derived divergence is unused.
        output.append(fd.Spectrum([total*scale-sum(counts), *counts, 0]))
    return tuple(output)


def dfe_masses(mean, shape):
    """Probability in magnitude bins [0,1), [1,10), [10,100), [100,infinity)."""
    if (isinstance(mean, bool) or isinstance(shape, bool) or not isfinite(mean)
            or not isfinite(shape) or mean >= 0 or shape <= 0):
        raise ValueError("mean must be negative and shape positive, both finite")
    cdf = fd.GammaExpParametrization().get_cdf(S_d=mean, b=shape, p_b=0, S_b=1)
    # The DFE is reflected: F(-x) = P(|S| >= x).
    tail = np.asarray(cdf(np.array([-1.0, -10.0, -100.0])))
    return [float(1-tail[0]), float(tail[0]-tail[1]), float(tail[1]-tail[2]), float(tail[2])]


def fit(neutral, selected, resolution=RESOLUTION, starts=STARTS):
    """Fit mean S_d and shape b through fastDFE's actual likelihood optimizer.

    Three explicitly supplied starts; no random multistart or bootstrap. The
    kernel is reused, while each optimization receives fresh Spectrum objects.
    """
    n = neutral.n
    for sfs in (neutral, selected):
        values = np.asarray(sfs.data)
        if (sfs.n != n or np.any(~np.isfinite(values)) or np.any(values < 0)
                or values[0] <= 0 or values[-1] != 0 or sfs.n_polymorphic <= 0):
            raise ValueError("spectra must be finite, nonnegative, matched, and have zero divergence")
    if np.any(neutral.polymorphic <= 0):
        raise ValueError("neutral bins must be nonzero for this demo")
    if not starts:
        raise ValueError("provide at least one starting point")
    for mean, shape in starts:
        if (isinstance(mean, bool) or isinstance(shape, bool)
                or not -1e4 < mean < -0.1 or not 0.1 < shape < 5):
            raise ValueError("starting mean and shape must be strictly inside the fitting bounds")
    runs = []
    for mean, shape in starts:
        inference = fd.BaseInference(
            sfs_neut=fd.Spectrum(neutral.data.copy()), sfs_sel=fd.Spectrum(selected.data.copy()),
            model=fd.GammaExpParametrization(), discretization=discretization(int(n), resolution),
            fixed_params={"all": dict(FIXED)}, x0={"all": {"S_d": mean, "b": shape}},
            bounds={"S_d": (-1e4, -0.1), "b": (0.1, 5.0)},
            opts_mle={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 600},
            method_mle="Nelder-Mead", n_runs=1, seed=SEED,
            do_bootstrap=False, parallelize=False, folded=False, include_divergence=False,
        )
        modeled = inference.run(do_bootstrap=False, pbar=False)
        params = inference.params_mle
        if (not inference.result.success or not isfinite(inference.likelihood)
                or any(not isfinite(float(params[k])) for k in ("S_d", "b"))):
            raise RuntimeError("fastDFE optimization did not converge to a finite fit")
        if any(float(params[key]) != value for key, value in FIXED.items()):
            raise RuntimeError("a parameter intended to remain fixed changed")
        if not -9999 < params["S_d"] < -0.10001 or not 0.10001 < params["b"] < 4.9999:
            raise RuntimeError("boundary optimum: inspect the model and fitting bounds")
        runs.append({"start": [mean, shape], "S_d": float(params["S_d"]),
                     "b": float(params["b"]), "log_likelihood": float(inference.likelihood),
                     "expected_selected_counts": list(map(float, modeled.polymorphic))})
    best = max(runs, key=lambda row: row["log_likelihood"])
    for key in ("S_d", "b"):
        if not np.allclose([r[key] for r in runs], best[key], rtol=2e-4, atol=1e-5):
            raise RuntimeError("starting points disagree; inspect the likelihood surface")
    return {"S_d": best["S_d"], "b": best["b"], "log_likelihood": best["log_likelihood"],
            "theta_from_neutral": float(neutral.theta),
            "neutral_target_sites": float(neutral.n_sites),
            "selected_target_sites": float(selected.n_sites),
            "resolution_per_sign": resolution,
            "magnitude_bin_probabilities": dfe_masses(best["S_d"], best["b"]),
            "expected_selected_counts": best["expected_selected_counts"],
            "runs": [{k: v for k, v in row.items() if k != "expected_selected_counts"} for row in runs]}


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return float(f"{value:.9g}") if type(value) is float else value


def build_report(data=None, target_factor=1.0):
    data = json.loads(FIXTURE.read_text()) if data is None else data
    neutral, selected = observed_spectra(data, target_factor)
    baseline = fit(neutral, selected)
    alternate_neutral, alternate_selected = observed_spectra(data, 2.0)
    target_control = fit(alternate_neutral, alternate_selected)
    refined = fit(neutral, selected, REFINED_RESOLUTION)
    noiseless = fit(*expected_spectra())
    return rounded({
        "example": "fastDFE: deleterious gamma DFE and selected target-count sensitivity",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("fastdfe", "sfsutils-popgen", "numpy", "scipy", "pandas")},
        "generator_seed": data["seed"], "haploid_sample_size": neutral.n,
        "neutral_polymorphic_sites": int(neutral.n_polymorphic),
        "selected_polymorphic_sites": int(selected.n_polymorphic),
        "fixed_parameters": FIXED,
        "optimizer": "fastDFE Nelder-Mead; 3 explicit starts; xatol=1e-7; fatol=1e-8; maxiter=600",
        "integration_ranges": {"negative": [-1e8, -1e-6], "positive": [1e-6, 1e3]},
        "magnitude_bins": ["0 <= |S| < 1", "1 <= |S| < 10", "10 <= |S| < 100", "100 <= |S|"],
        "fit": baseline, "double_selected_target_control": target_control,
        "refined_resolution_fit": refined, "same_solver_noiseless_control": noiseless,
        "relative_resolution_change_in_mean_magnitude": abs(refined["S_d"]/baseline["S_d"]-1),
        "relative_resolution_change_in_shape": abs(refined["b"]/baseline["b"]-1),
        "interpretation": "S_d is mean population-scaled effect S=4*Ne*s, conditional on "
                          "the gamma model, mutation-rate sharing, dominance and target sizes. "
                          "Doubling only the selected target preserves all variant counts but "
                          "changes the inferred DFE. No absolute s, confidence interval or "
                          "benchmark claim is provided.",
    })


def check_sample(actual, expected):
    """Regression tolerance, not a statistical uncertainty statement."""
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
        if not np.isclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise AssertionError(f"numerical sample mismatch: {actual} versus {expected}")
    elif actual != expected:
        raise AssertionError(f"sample mismatch: {actual!r} versus {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-factor", type=float, default=1.0)
    parser.add_argument("--generate-input", action="store_true")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    if args.generate_input and args.check_sample:
        parser.error("choose generation or sample checking")
    try:
        result = generate_input() if args.generate_input else build_report(target_factor=args.target_factor)
        if args.check_sample:
            check_sample(result, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))
    except (ValueError, RuntimeError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
