"""Deterministic constructed examples; no randomness, data or model weights."""

from dataclasses import asdict
import json
import math

from .model import SegmentMoments, route


def dense_segment_mass(segments, query):
    """Diagnostic only: direct token softmax, grouped by segment (Eq. 10).

    This deliberately scans all tokens and is NOT the moment router.
    """
    scores = [[math.fsum(q*k for q, k in zip(query, key, strict=True))
               for key in segment] for segment in segments]
    peak = max(s for segment in scores for s in segment)
    masses = [math.fsum(math.exp(s-peak) for s in segment) for segment in scores]
    total = math.fsum(masses)
    return tuple(m/total for m in masses)


def _round(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_round(v) for v in value]
    return value


def build_report():
    segments = [[[-2.0], [2.0]], [[0.0], [0.0]]]
    summaries = [SegmentMoments(s) for s in segments]
    result = route(summaries, [1.0], [[1.0, 0.0], [0.0, 1.0]])
    # Both four-token segments have mean 0 and population variance 12.5.
    # They differ in fourth moment: 312.5 versus 168.5, hence exponential evidence.
    collision = [[[-5.0], [0.0], [0.0], [5.0]],
                 [[-4.0], [-3.0], [3.0], [4.0]]]
    collision_result = route([SegmentMoments(s) for s in collision], [1.0], [[1.0], [0.0]])
    return _round({
        "scope": "MERLIN moment routing only; supplied readouts; no training or speed claim",
        "random_seed": None,
        "equal_mean_different_variance": {
            "keys": segments, "query": [1.0], "gamma": 0.5,
            "tau_mean": 1.0, "tau_variance": 1.0,
            **asdict(result),
            "dense_token_softmax_segment_mass": dense_segment_mass(segments, [1.0]),
        },
        "same_first_two_moments_different_higher_moments": {
            "mean_scores": collision_result.mean_scores,
            "variance_scores": collision_result.variance_scores,
            "fourth_moments": [312.5, 168.5],
            "moment_routing_weights": collision_result.weights,
            "dense_token_softmax_segment_mass": dense_segment_mass(collision, [1.0]),
        },
    })


def main():
    print(json.dumps(build_report(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
