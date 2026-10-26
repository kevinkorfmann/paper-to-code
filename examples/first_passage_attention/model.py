"""Liu (2026), First Passage Attention v1, equation (1).

One causal head with supplied Q/K/V, scalar fields and key levels. The dense
and incremental paths expose prefix rows, not padded matrices. No training,
hard masking of negative biases, or key eviction is performed.
"""

from dataclasses import dataclass
from math import sqrt

from examples.differential_attention.model import _dot, _finite, _matrix, softmax


def _vector(values, name):
    result = tuple(_finite(x, name) for x in values)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _strength(beta):
    result = _finite(beta, "beta")
    if result <= 0:
        raise ValueError("beta must be strictly positive")
    return result


def walk_from_increments(increments):
    """Construct W_t = sum_{u<=t} delta_u for the paper's stack specialization."""
    increments = _vector(increments, "increments")
    total, walk = 0.0, []
    for delta in increments:
        total = _finite(total + delta, "walk")
        walk.append(total)
    return tuple(walk)


def _penalty(minimum, level, beta):
    # Comparing first avoids overflow in a nonnegative difference whose
    # minimum with zero would be zero anyway. Negative overflow is rejected.
    if minimum >= level:
        return 0.0
    return _finite(beta * (minimum - level), "first-passage bias")


def first_passage_bias(field, levels, beta=4.0):
    """Return causal prefix rows of Eq. (1); b[t,t]=0 by definition.

    Scan a query's earlier keys backwards, maintaining min(field[s+1:t+1]).
    This computes all rows in O(T^2) time and O(T^2) returned space.
    """
    phi, psi = _vector(field, "field"), _vector(levels, "levels")
    strength = _strength(beta)
    if len(phi) != len(psi):
        raise ValueError("field and levels must have the same length")
    rows = []
    for t in range(len(phi)):
        row = [0.0] * (t + 1)
        minimum = phi[t]
        for s in range(t - 1, -1, -1):
            row[s] = _penalty(minimum, psi[s], strength)
            minimum = min(minimum, phi[s])
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True)
class AttentionRow:
    biases: tuple[float, ...]
    weights: tuple[float, ...]
    output: tuple[float, ...]


def _read(query, keys, values, biases):
    scale = sqrt(len(query))
    logits = tuple(_finite(_dot(query, key) / scale + bias, "attention logit")
                   for key, bias in zip(keys, biases, strict=True))
    weights = softmax(logits)
    output = tuple(_dot(weights, column) for column in zip(*values))
    return AttentionRow(biases, weights, output)


def causal_attention(queries, keys, values, field, levels, beta=4.0):
    """Full forward pass with Nxd Q/K, Nxd_v V, and N scalar fields/levels.

    Only already available keys enter each row. The output is a tuple of
    AttentionRow objects whose bias/weight lengths are 1, 2, ..., N.
    """
    q, k, v = (_matrix(matrix, name) for matrix, name in
               ((queries, "queries"), (keys, "keys"), (values, "values")))
    if len(k) != len(q) or len(v) != len(q) or len(k[0]) != len(q[0]):
        raise ValueError("Q/K must share Nxd shape; values must have N rows")
    biases = first_passage_bias(field, levels, beta)
    if len(biases) != len(q):
        raise ValueError("one field and one level are required per position")
    return tuple(_read(query, k[:t+1], v[:t+1], biases[t])
                 for t, query in enumerate(q))


class IncrementalAttention:
    """Exact forward recurrence for one head, retaining every cached key.

    A new field updates every old key's subsequent range minimum. The new
    key has no subsequent field yet (None), so its current bias is zero.
    Each step costs O(t*(d+d_v)); the cache uses O(t*(d+d_v)) space, plus two
    scalars per key. This is transparent Python, not an optimized kernel.
    Invalid steps leave the cache unchanged.
    """

    def __init__(self, beta=4.0):
        self._beta = _strength(beta)
        self._keys = ()
        self._values = ()
        self._levels = ()
        self._minima = ()

    def __len__(self):
        return len(self._keys)

    def step(self, query, key, value, field, level):
        q, k, v = (_vector(vector, name) for vector, name in
                   ((query, "query"), (key, "key"), (value, "value")))
        phi, psi = _finite(field, "field"), _finite(level, "level")
        if len(q) != len(k):
            raise ValueError("query and key widths must match")
        if self._keys and (len(k) != len(self._keys[0]) or len(v) != len(self._values[0])):
            raise ValueError("key and value widths cannot change within a cache")
        minima = tuple(phi if previous is None else min(previous, phi)
                       for previous in self._minima)
        biases = tuple(_penalty(m, p, self._beta)
                       for m, p in zip(minima, self._levels, strict=True)) + (0.0,)
        keys, values = self._keys + (k,), self._values + (v,)
        row = _read(q, keys, values, biases)
        # Commit only after validation and all floating-point calculations.
        self._keys, self._values = keys, values
        self._levels = self._levels + (psi,)
        self._minima = minima + (None,)
        return row
