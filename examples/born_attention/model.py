"""Quartic complex-overlap attention and region-intensity readout.

Original implementation of mechanisms in arXiv:2611.05355v1, Eqs. 2 and 4.
No vector normalization, softmax, epsilon, or implicit zero-row fallback.
"""
import math
from numbers import Number


def vector(values, name="vector"):
    """Copy a nonempty finite numeric vector into built-in complex values."""
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty numeric sequence") from exc
    if not values:
        raise ValueError(f"{name} must be nonempty")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ValueError(f"{name} must contain numbers, not booleans or strings")
        try:
            value = complex(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} contains an unrepresentable number") from exc
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError(f"{name} must be finite")
        result.append(value)
    return result


def matrix(values, name):
    try:
        rows = [vector(row, name) for row in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty matrix") from exc
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be nonempty and rectangular")
    return rows


def magnitude(value):
    try:
        return abs(value)
    except OverflowError as exc:
        raise ValueError("Intermediate magnitude overflow") from exc


def normalized_powers(magnitudes, power):
    """Scale before exponentiation; reject undefined all-zero normalization."""
    if not all(math.isfinite(x) for x in magnitudes):
        raise ValueError("Intermediate magnitude overflow")
    scale = max(magnitudes)
    if scale == 0:
        raise ValueError("Zero total intensity: normalization is undefined")
    scores = [(x / scale) ** power for x in magnitudes]
    total = math.fsum(scores)
    return [x / total for x in scores]


def attention(queries, keys, values):
    """Return (weights, mixed_values) for Q[n,d], K[m,d], V[m,dv].

    Uses conjugate(query) dot key, NOT normalized quantum-state fidelity.
    Dense supplied inputs only: callers must select any visible prefix themselves.
    """
    queries = matrix(queries, "queries")
    keys = matrix(keys, "keys")
    values = matrix(values, "values")
    if len(queries[0]) != len(keys[0]) or len(keys) != len(values):
        raise ValueError("Query/key dimensions and key/value row counts must match")
    weights = []
    output = []
    for query in queries:
        overlaps = [sum((q.conjugate() * k for q, k in zip(query, key)), 0j)
                    for key in keys]
        row = normalized_powers([magnitude(x) for x in overlaps], 4)
        mixed = [sum((w * value[j] for w, value in zip(row, values)), 0j)
                 for j in range(len(values[0]))]
        output.append(vector(mixed, "attention output"))
        weights.append(row)
    return weights, output


def born_readout(amplitudes, regions):
    """Return class probabilities and pixel-count baseline for a partition.

    Input is already phi = R sum_t h_t. Every pixel must occur exactly once.
    """
    amplitudes = vector(amplitudes, "amplitudes")
    try:
        regions = [list(region) for region in regions]
    except TypeError as exc:
        raise ValueError("regions must be a partition of pixel indices") from exc
    if not regions or any(not region for region in regions):
        raise ValueError("regions must be nonempty")
    indices = [i for region in regions for i in region]
    if any(type(i) is not int for i in indices):
        raise ValueError("pixel indices must be integers")
    if sorted(indices) != list(range(len(amplitudes))):
        raise ValueError("regions must partition all pixels exactly once")
    intensity = normalized_powers([magnitude(x) for x in amplitudes], 2)
    return ([math.fsum(intensity[i] for i in region) for region in regions],
            [len(region) / len(amplitudes) for region in regions])
