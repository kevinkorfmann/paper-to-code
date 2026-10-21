"""A head can decline to contribute, without renormalizing its attention map."""

import argparse
import json
from math import log

from .model import gated_attention


def build_report(gate_logit=None):
    # Every query has the same deliberately chosen attention distribution.
    q = [[1.0]] * 3
    k = [[log(p)] for p in (0.8, 0.1, 0.1)]
    v = [[10.0, 1.0], [10.0, 0.0], [10.0, 0.0]]
    # A one-dimensional hidden feature, projected with w=1, sets each gate.
    logits = [-log(9), 0.0, log(9)] if gate_logit is None else [gate_logit] * 3
    result = gated_attention(q, k, v, [[z] for z in logits], [1.0])
    r = lambda xs: [round(a, 9) for a in xs]
    return {
        "example": "query-dependent headwise sigmoid gating after SDPA",
        "randomness": "none; inputs and gate weights are constructed, not trained",
        "queries": [{
            "gate_logit": round(z, 9), "gate": round(g, 9),
            "attention_map": r(a), "ordinary_output": r(o), "gated_output": r(y),
            "effective_coefficient_sum": round(sum(g * p for p in a), 9),
        } for z, g, a, o, y in zip(logits, result.gates, result.attention,
                                  result.ordinary_output, result.gated_output)],
        "interpretation": (
            "The scalar gate shrinks both channels equally. For these fixed Q/K, "
            "attention still assigns 80% to token 0: this is not evidence that "
            "a gate alone removes an attention sink, improves an LLM, or saves compute."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-logit", type=float)
    args = parser.parse_args()
    try:
        report = build_report(args.gate_logit)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
