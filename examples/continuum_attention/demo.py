"""Compare continuous retrieval, discrete softmax and the endpoint convention."""

import argparse
from dataclasses import asdict
import json
import math
import platform

from .model import attend, segment_integrals


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [rounded(x) for x in value]
    return round(value, 12) if type(value) is float else value


def build_report(gap=math.log(4)):
    if not math.isfinite(gap) or not -100 <= gap <= 100:
        raise ValueError("gap must be finite and in [-100,100]")
    values = [[0.0], [1.0]]
    read = attend([0.0, gap], values)
    peak = max(0.0, gap)
    discrete = math.exp(gap-peak)/(math.exp(-peak)+math.exp(gap-peak))
    flat = attend([0.0]*3, [[0.0], [1.0], [0.0]])
    no_caps = attend([0.0]*3, [[0.0], [1.0], [0.0]], boundary_caps=False)
    future = attend([0.0, gap, 1e6], values+[[1e100]], prefix_length=2)
    order = attend([0, 2, -1], [[0], [1], [0]])
    reordered = attend([2, 0, -1], [[1], [0], [0]])
    return rounded({
        "example": "Continuum Attention: exact forward segment integration",
        "python": platform.python_version(), "randomness": "none; deterministic constructed inputs",
        "two_token_read": {"scaled_logits": [0.0, gap], "values": values,
                           "continuum": asdict(read), "discrete_softmax_output": [discrete]},
        "equal_logit_boundary_control": {"with_half_token_caps": asdict(flat),
                                          "without_caps_ablation": asdict(no_caps)},
        "segment_retrieval_coordinates": [
            {"gap": d, "mean_u": segment_integrals(0, d).mean_coordinate} for d in [-4.0, 0.0, 4.0]],
        "causal_prefix_check": {"before": read.output, "with_extreme_future": future.output,
                                "future_weight": future.token_weights[-1],
                                "maximum_output_difference": max(abs(a-b) for a, b in zip(read.output, future.output, strict=True))},
        "adjacency_control": {"original_order": [0, 1, 2], "permuted_order": [1, 0, 2],
                              "original_output": order.output, "permuted_output": reordered.output},
        "scope": "Forward operator on supplied logits and values; no training, learned projections, "
                 "backward kernel, speed benchmark, or language-model quality claim.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=float, default=math.log(4), help="second endpoint logit; first is zero")
    args = parser.parse_args()
    try:
        result = build_report(args.gap)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
