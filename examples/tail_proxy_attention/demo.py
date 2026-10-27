"""Two exact toy calculations and a causal masking check for tail proxies."""

import argparse
from dataclasses import asdict
import json
import math
import platform

from .model import dense_attention, tail_proxy_attention


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [rounded(x) for x in value]
    return round(value, 12) if type(value) is float else value


def compare(logits, values):
    groups = [list(range(len(logits)))]
    dense = dense_attention(logits, values)
    proxy = tail_proxy_attention(logits, values, groups, exact=[0])
    discarded = dense_attention(logits, values, mask=[i == 0 for i in range(len(logits))])
    no_count = tail_proxy_attention(logits, values, groups, exact=[0], count_weighting=False)
    singletons = tail_proxy_attention(logits, values, [[i] for i in range(len(logits))], exact=[0])
    return {"scaled_logits": logits, "values": values, "exact_indices": [0],
            "groups_before_exact_removal": groups,
            "dense_output": dense.output, "proxy": asdict(proxy),
            "discarded_tail_output": discarded.output,
            "no_count_ablation_output": no_count.output,
            "singleton_tail_output": singletons.output,
            "proxy_absolute_error": abs(proxy.output[0]-dense.output[0]),
            "discard_absolute_error": abs(discarded.output[0]-dense.output[0]),
            "dense_softmax_entries": len(logits),
            "proxy_softmax_entries": len(proxy.component_weights)}


def build_report(spread=2.0):
    if not math.isfinite(spread) or not 0 <= spread <= 100:
        raise ValueError("spread must be finite and between 0 and 100")
    logits = [math.log(4), 0.0, 0.0, 0.0]
    values = [[1.0], [2.0], [4.0], [6.0]]
    uniform = compare(logits, values)
    heterogeneous = compare([0.0, -spread, spread], [[0.0], [1.0], [0.0]])
    base = tail_proxy_attention(logits, values, [[0, 1, 2, 3]], exact=[0])
    future = tail_proxy_attention(logits+[1e6], values+[[1e100]], [[0, 1, 2, 3, 4]],
                                  exact=[0], mask=[True, True, True, True, False])
    return rounded({
        "example": "OpenAttention residual tail-proxy component; constructed groups",
        "python": platform.python_version(), "randomness": "none; deterministic constructed inputs",
        "equal_logit_tail": uniform, "variable_logit_counterexample": heterogeneous,
        "causal_prefix_check": {"prefix_output": base.output, "with_masked_extreme_future_output": future.output,
                                "maximum_absolute_difference": max(abs(x-y) for x, y in zip(base.output, future.output, strict=True)),
                                "residual_members_after_mask": future.proxies[0].members},
        "scope": "Supplied groups and exact indices only. No QGC, QGS, ErrorGate, GPU kernel, "
                 "cache-read measurement, 1% experiment, training, or LLM performance claim.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spread", type=float, default=2.0,
                        help="counterexample tail logits are -spread and +spread")
    args = parser.parse_args()
    try:
        result = build_report(args.spread)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
