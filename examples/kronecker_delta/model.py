"""Scalar pair-association revision from Hartl et al. (2026), Eq. 3.

This is the ungated core rule, NOT the KronLSTM architecture (Eq. 10).
Memory has value rows and key columns. No inputs are implicitly normalized.
"""

import math


def _finite(value, name):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _vector(values, name):
    try:
        result = tuple(_finite(x, name) for x in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty vector") from exc
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _matrix(memory):
    try:
        result = tuple(_vector(row, "memory") for row in memory)
    except TypeError as exc:
        raise ValueError("memory must be a nonempty rectangular matrix") from exc
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError("memory must be a nonempty rectangular matrix")
    return result


def _dot(left, right):
    try:
        return _finite(math.fsum(x*y for x, y in zip(left, right, strict=True)), "dot product")
    except (OverflowError, ValueError) as exc:
        raise ValueError("dot product exceeds supported floating-point range") from exc


def _inputs(memory, key, value):
    state = _matrix(memory)
    k, v = _vector(key, "key"), _vector(value, "value")
    if len(k) != len(state[0]) or len(v) != len(state):
        raise ValueError("key/value dimensions must match memory columns/rows")
    return state, k, v


def _step_size(beta):
    beta = _finite(beta, "beta")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    return beta


def read(memory, query):
    """Return S q. It can be a superposition, not a unique stored value."""
    state, q = _matrix(memory), _vector(query, "query")
    if len(q) != len(state[0]):
        raise ValueError("query dimension must match memory columns")
    return tuple(_dot(row, q) for row in state)


def association(memory, key, value):
    """Return the scalar v^T S k, distinct from the vector readout S k."""
    state, k, v = _inputs(memory, key, value)
    return _dot(v, tuple(_dot(row, k) for row in state))


def kron_update(memory, key, value, *, target=1.0, beta=1.0):
    """Return S + beta*(target - v^T S k)*v k^T (paper Eq. 3).

    For nonzero k and v, one-step projection uses
    beta = 1/(||k||^2 ||v||^2). Repeated-pair error contracts only when
    0 < beta*||k||^2*||v||^2 < 2. Larger steps are allowed for controls.
    A target of zero erases ONE scalar association at the projection step;
    it does not necessarily erase the whole vector readout.
    """
    state, k, v = _inputs(memory, key, value)
    c, beta = _finite(target, "target"), _step_size(beta)
    if beta == 0 or not any(k) or not any(v):
        return state
    current = _dot(v, tuple(_dot(row, k) for row in state))
    error = _finite(c-current, "association residual")
    amount = _finite(beta*error, "update coefficient")
    return tuple(tuple(_finite(s + amount*vi*kj, "updated memory")
                       for s, kj in zip(row, k, strict=True))
                 for row, vi in zip(state, v, strict=True))


def delta_update(memory, key, value, *, beta=1.0):
    """Vector-residual control S + beta*(v - S k)*k^T (paper Eq. 2).

    This is the ungated Delta Rule, not an implementation of DeltaNet.
    Its objective differs from the Kronecker rule's scalar objective.
    """
    state, k, v = _inputs(memory, key, value)
    beta = _step_size(beta)
    if beta == 0 or not any(k):
        return state
    retrieval = tuple(_dot(row, k) for row in state)
    amounts = tuple(_finite(beta*_finite(vi-ri, "vector residual"), "update coefficient")
                    for vi, ri in zip(v, retrieval, strict=True))
    return tuple(tuple(_finite(s + amount*kj, "updated memory")
                       for s, kj in zip(row, k, strict=True))
                 for row, amount in zip(state, amounts, strict=True))
