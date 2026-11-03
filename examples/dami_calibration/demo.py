"""Constructed calibration, parameter coupling, shift and interference controls."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform

from .model import Gate, calibrate, diagnose, geometric_timescales, retention_path

SAMPLE = Path(__file__).with_name("sample_output.json")


def build_report():
    activations = (-2.0, -0.5, 0.5, 2.0)
    timescales = geometric_timescales(3, 4.0, 64.0)
    sensitivities = (0.25, 0.75, 0.25)
    rows = []
    for layer, sensitivity in enumerate(sensitivities):
        for head, tau in enumerate(timescales):
            gate = calibrate(activations, tau, sensitivity)
            rows.append({"layer": layer, "head": head, "target_timescale": tau,
                         "target_sensitivity": sensitivity, **asdict(gate),
                         **diagnose(gate, activations)})
    low = calibrate(activations, 16.0, 0.25)
    high = calibrate(activations, 16.0, 0.75)
    bias_only = Gate(high.bias, low.a_log)
    shift = tuple(x + 2 for x in activations)
    calibrated = retention_path(low, activations * 4)
    shifted = retention_path(low, shift * 4)
    # A separate scalar GDN check: with k=1, beta=1 and new value=0,
    # S_new = r*S_old + (0-r*S_old) = 0 despite positive explicit decay.
    decay = low.retention(activations[0])
    state_before = 1.0
    decayed = decay * state_before
    overwritten = decayed + 1.0 * 1.0 * (0.0 - 1.0 * decayed)
    exact_z = (-math.log(3.0), math.log(3.0))
    exact_gate = calibrate(exact_z, 2.0 / math.log(16.0 / 3.0), 0.5)
    return {
        "example": "DAMI scalar gate calibration",
        "python_version": platform.python_version(),
        "randomness": "None; all calibration activations and targets are constructed.",
        "calibration_activations": activations,
        "head_timescales": timescales,
        "supplied_layer_sensitivities": sensitivities,
        "calibrated_grid": rows,
        "parameter_coupling": {
            "low_sensitivity_joint": diagnose(low, activations),
            "change_bias_only": diagnose(bias_only, activations),
            "high_sensitivity_joint": diagnose(high, activations)},
        "decay_envelope": {
            "steps": 16, "calibration_sample_repeated": calibrated["retention"][-1],
            "matched_exp_minus_one": math.exp(-1.0),
            "activations_shifted_by_two": shifted["retention"][-1],
            "shifted_diagnostics_without_recalibration": diagnose(low, shift)},
        "exact_mean_retention_counterexample": {
            "activations": exact_z, **asdict(exact_gate),
            "individual_retentions": [exact_gate.retention(z) for z in exact_z],
            **diagnose(exact_gate, exact_z)},
        "write_interference_counterexample": {
            "state_before": state_before, "explicit_retention": decay,
            "state_after_decay": decayed, "key": 1.0, "beta": 1.0,
            "new_value": 0.0, "state_after_delta_write": overwritten},
        "limitations": "Supplied samples only, not layer-sequential model calibration, training, architecture-specific extensions or retrieval benchmarking. Mean log decay is matched on the calibration sample; distribution shifts and writes can invalidate a memory-retention interpretation. No speed or model-quality claim.",
    }


def render(value):
    def rounded(x):
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(value), indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = render(build_report())
    if args.check_sample and output != SAMPLE.read_text():
        parser.error("Output differs from checked sample")
    print(output, end="")


if __name__ == "__main__":
    main()
