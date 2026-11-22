"""Minimal exact HiSA score rule with scalar values; see README for limits."""

import json
import math


def frequencies(d, base):
    if d < 2 or d % 2 or not math.isfinite(base) or base <= 1:
        raise ValueError("head dimension must be positive/even and base finite > 1")
    return tuple(base ** (-2 * pair / d) for pair in range(d // 2))


def active_pairs(distance, omega, budget):
    if distance < 0 or math.isnan(budget) or budget <= 0:
        raise ValueError("distance must be nonnegative; budget must be positive")
    # Test the unwrapped angle, not its cosine, sine, or angle modulo 2*pi.
    return tuple(pair for pair, rate in enumerate(omega) if distance * rate < budget)


def score(q, k, distance, omega, budget):
    """Return None for a key absent from softmax (not an included logit 0)."""
    if len(q) != len(k) or len(q) != 2 * len(omega):
        raise ValueError("incompatible head dimensions")
    pairs = active_pairs(distance, omega, budget)
    if not pairs:
        return None
    terms = []
    for pair in pairs:
        x, y = q[2 * pair:2 * pair + 2]
        u, v = k[2 * pair:2 * pair + 2]
        angle = distance * omega[pair]
        # <R_i q, R_j k> = <R_(i-j) q, k>.
        x1 = x * math.cos(angle) - y * math.sin(angle)
        y1 = x * math.sin(angle) + y * math.cos(angle)
        terms.append(x1 * u + y1 * v)
    return math.fsum(terms) / math.sqrt(len(q))


def attention(q, query_pos, keys, base, budget):
    """keys contains (position, unrotated key, scalar value) rows."""
    omega = frequencies(len(q), base)
    eligible = []
    for key_pos, key, value in keys:
        if key_pos <= query_pos:
            logit = score(q, key, query_pos - key_pos, omega, budget)
            if logit is not None:
                eligible.append((logit, value))
    if not eligible:
        raise ValueError("no key is eligible for softmax")
    peak = max(logit for logit, _ in eligible)
    weights = [math.exp(logit - peak) for logit, _ in eligible]
    output = math.fsum(w * value for w, (_, value) in zip(weights, eligible))
    return output / math.fsum(weights)


def pair_work(length, omega, budget):
    """Causal active-pair count and dense count, not machine time (Eq. 5)."""
    if length <= 0:
        raise ValueError("length must be positive")
    kept = sum((length - distance) * len(active_pairs(distance, omega, budget))
               for distance in range(length))
    return kept, len(omega) * length * (length + 1) // 2


def fixture():
    query = (2.0, 0.0, 0.0, 0.0)
    key = (1.0, 0.0, 0.0, 0.0)
    keys = ((7, key, 0.0), (5, key, 10.0), (1, key, -4.0), (0, key, 100.0))
    return query, 7, keys, 16.0, math.pi / 2


def report():
    q, i, keys, base, budget = fixture()
    omega = frequencies(len(q), base)
    rows = []
    for j, key, value in keys:
        distance = i - j
        raw_score = score(q, key, distance, omega, budget)
        rows.append({
            "distance": distance,
            "active_pair_indices": list(active_pairs(distance, omega, budget)),
            "value": value,
            "partial_logit": None if raw_score is None else round(raw_score, 9),
        })
    kept, dense = pair_work(8, omega, budget)
    return {
        "rope_frequencies": omega,
        "angle_budget_radians": round(budget, 9),
        "keys": rows,
        "fast_pair_angle_at_distance_6_modulo_full_turn_radians":
            round(math.remainder(6.0, math.tau), 9),
        "hisa_output": round(attention(q, i, keys, base, budget), 9),
        "dense_output": round(attention(q, i, keys, base, math.inf), 9),
        "causal_pair_products": {"hisa": kept, "dense": dense},
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
