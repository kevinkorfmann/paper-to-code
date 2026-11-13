"""Original scalar-loop SPARE compact-state mechanism; no upstream code."""
import math


def vector(x, name):
    x = list(x)
    if not x or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in x):
        raise ValueError(f'{name} must be a nonempty finite real vector')
    return [float(v) for v in x]


def matrix(x, name):
    rows = [vector(row, name) for row in x]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f'{name} must be a nonempty rectangular matrix')
    return rows


def scalar(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f'invalid {name}')
    return float(x)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def basis(p):
    p = matrix(p, 'basis')
    r = len(p[0])
    if r > len(p):
        raise ValueError('rank exceeds key dimension')
    columns = list(zip(*p))
    if any(abs(dot(columns[i], columns[j]) - (i == j)) > 1e-12 for i in range(r) for j in range(r)):
        raise ValueError('basis columns must be orthonormal')
    return p


def project(p, x):
    p, x = basis(p), vector(x, 'vector')
    if len(p) != len(x):
        raise ValueError('projection dimension mismatch')
    return [dot(col, x) for col in zip(*p)]


def read(state, query):
    state, query = matrix(state, 'state'), vector(query, 'query')
    if len(state[0]) != len(query):
        raise ValueError('read dimension mismatch')
    return [dot(row, query) for row in state]


def update(state, key, value, alpha, beta, gain=1.0):
    """Shared delta-rule arithmetic, accepting closed gate boundaries as controls."""
    state, key, value = matrix(state, 'state'), vector(key, 'key'), vector(value, 'value')
    alpha, beta = scalar(alpha, 'alpha', 0, 1), scalar(beta, 'beta', 0, 1)
    gain = scalar(gain, 'gain', 0, float('inf'))
    if len(state) != len(value) or len(state[0]) != len(key):
        raise ValueError('update dimension mismatch')
    previous = read(state, key)
    out = [[alpha * s + beta * gain * (v - alpha * old) * k for s, k in zip(row, key)]
           for row, v, old in zip(state, value, previous)]
    if any(not math.isfinite(z) for row in out for z in row):
        raise ValueError('update overflow')
    return out


def compact_step(c, p, key, value, alpha, beta, *, epsilon=1e-8, cap=16.0, compensate=True):
    """Eqs. 4, 8–9 of Wan et al. (2026), with a supplied orthonormal basis.

    Keys are required to be unit normalized, as in the source architecture.
    Projection is NOT renormalized. epsilon and cap are explicit toy choices.
    """
    p, key = basis(p), vector(key, 'key')
    if abs(dot(key, key) - 1.0) > 1e-12:
        raise ValueError('keys must have unit norm')
    epsilon = scalar(epsilon, 'epsilon', 0, 1)
    cap = scalar(cap, 'cap', 1, float('inf'))
    if epsilon == 0 or not isinstance(compensate, bool):
        raise ValueError('epsilon must be positive; compensate must be boolean')
    khat = project(p, key)
    rho = dot(khat, khat)
    gamma = min(1.0 / max(rho, epsilon), cap) if compensate else 1.0
    c = update(c, khat, value, alpha, beta, gamma)
    return c, {'rho': rho, 'gamma': gamma, 'effective_write': beta * gamma * rho}


def reconstruct(c, p):
    """Diagnostic only; compact recurrence and read do not need a dense state."""
    c, p = matrix(c, 'state'), basis(p)
    if len(c[0]) != len(p[0]):
        raise ValueError('reconstruction dimension mismatch')
    return [[dot(row, prow) for prow in p] for row in c]
