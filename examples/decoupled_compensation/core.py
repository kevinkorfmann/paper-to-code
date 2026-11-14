"""Original scalar CENTROID estimator from DecoupAttn (2026), Eq. 5.

Supplied clusters and exact-token sets stand in for blocks. No clustering or
calibration is performed. Means use all members, including corrected tokens.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    output: float
    numerator: float
    denominator: float
    shift: float
    skipped_counts: tuple


def estimate(query, keys, values, labels, exact=()):
    """One scalar query, scalar keys/values; dimension d=1.

    Returns common-shift-scaled numerator/mass, not raw exponentials.
    Labels are contiguous integer cluster IDs. Exact indices are unique.
    """
    keys, values, labels, exact = tuple(keys), tuple(values), tuple(labels), tuple(exact)
    n = len(keys)
    if not n or len(values) != n or len(labels) != n:
        raise ValueError('nonempty equal-length inputs required')
    if not all(math.isfinite(x) for x in (query, *keys, *values)):
        raise ValueError('finite numeric inputs required')
    if any(type(x) is not int or x < 0 for x in labels):
        raise ValueError('labels must be nonnegative integers')
    count = max(labels) + 1
    if set(labels) != set(range(count)):
        raise ValueError('labels must be contiguous')
    if len(set(exact)) != len(exact) or any(type(i) is not int or not 0 <= i < n for i in exact):
        raise ValueError('exact indices must be unique and in range')
    groups = [[i for i in range(n) if labels[i] == c] for c in range(count)]
    means_k = [math.fsum(keys[i] / len(g) for i in g) for g in groups]
    means_v = [math.fsum(values[i] / len(g) for i in g) for g in groups]
    selected = set(exact)
    skipped = tuple(sum(i not in selected for i in g) for g in groups)
    # Direct skipped-count form avoids subtracting large approximate totals.
    scores = [query * keys[i] for i in exact]
    scores += [query * means_k[c] for c in range(count) if skipped[c]]
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('score overflow')
    shift = max(scores)
    terms = [(math.exp(query * keys[i] - shift), values[i]) for i in exact]
    terms += [(skipped[c] * math.exp(query * means_k[c] - shift), means_v[c])
              for c in range(count) if skipped[c]]
    denominator = math.fsum(w for w, _ in terms)
    numerator = math.fsum(w * v for w, v in terms)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError('aggregation overflow')
    return Estimate(numerator / denominator, numerator, denominator, shift, skipped)


def dense(query, keys, values):
    """Dense endpoint of the same scalar estimator."""
    return estimate(query, keys, values, [0] * len(keys), range(len(keys)))
