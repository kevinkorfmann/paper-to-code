"""Compare abstention, per-read filtering, and renormalized routing on two values."""

import argparse
from dataclasses import asdict
import json
from math import log

from .model import attend, projection_gates


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return round(value, 9) if type(value) is float else value


def build_report(sink_logit=log(2), real_shift=0.0):
    values = [[2.0, 0.0], [0.0, 2.0]]
    weight = [log(9)/2, -log(9)/2]
    gates = projection_gates(values, weight)
    scores = [real_shift, real_shift]
    baseline = attend(scores, values)
    phantom = attend(scores, values, sink_logit=sink_logit)
    filtered = attend(scores, values, gates=gates)
    combined = attend(scores, values, sink_logit=sink_logit, gates=gates)
    uniform = attend(scores, values, gates=[0.1, 0.1])
    low = attend([-8.0, -8.0], values, sink_logit=0.0)
    cancellation = attend([0, 0], [[1.0, 0.0], [-1.0, 0.0]])
    return rounded({
        "example": "attention abstention, per-read value gates, and retained mass",
        "values": values, "real_logits": scores, "sink_logit": sink_logit,
        "projection_weight": weight, "projection_bias": 0.0,
        "baseline": asdict(baseline), "phantom_only": asdict(phantom),
        "value_gates_only": asdict(filtered), "phantom_plus_value_gates": asdict(combined),
        "post_aggregation_scalar_at_same_retained_mass": [filtered.retained_mass*x for x in baseline.output],
        "uniform_gate_control": asdict(uniform), "low_real_logit_control": asdict(low),
        "cancellation_without_abstention": asdict(cancellation),
        "interpretation": "The phantom contributes zero while absorbing routing mass. "
                          "Per-read gates can change the surviving content; renormalizing "
                          "their coefficients removes retained-mass attenuation. The gates "
                          "are deliberately constructed, not learned relevance judgments. "
                          "Zero output can also arise by cancellation at full routing mass.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sink-logit", type=float, default=log(2))
    parser.add_argument("--real-shift", type=float, default=0.0)
    args = parser.parse_args()
    try:
        report = build_report(args.sink_logit, args.real_shift)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
