"""Supplied nonnegative feature factors; no learned projections or local branch."""
import math


def matrix(x, name, nonnegative=False):
    if not isinstance(x, (list, tuple)) or not x:
        raise ValueError(f"{name} must be a nonempty matrix")
    out = []
    for row in x:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{name} has an invalid row")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
            raise ValueError(f"{name} needs numeric entries")
        try:
            r = [float(v) for v in row]
        except (OverflowError, ValueError):
            raise ValueError(f"{name} entries must be finite") from None
        if any(not math.isfinite(v) or (nonnegative and v < 0) for v in r):
            raise ValueError(f"{name} has invalid entries")
        if out and len(r) != len(out[0]):
            raise ValueError(f"{name} must be rectangular")
        out.append(r)
    return out


def finite(x):
    if not math.isfinite(x):
        raise ValueError("arithmetic overflow: rescale inputs")
    return x


def mm(a, b):
    """Internal matrix product, with explicit overflow rejection."""
    try:
        return [[finite(math.fsum(x*y for x, y in zip(row, col)))
                 for col in zip(*b)] for row in a]
    except (OverflowError, ValueError):
        raise ValueError("arithmetic overflow: rescale inputs") from None


def transpose(a):
    return [list(row) for row in zip(*a)]


def weights(beta, terms):
    """Geometric masses conditioned on length in 0..terms-1."""
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 1:
        raise ValueError("terms must be a positive integer")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be numeric")
    if not 0 <= beta < 1 or not math.isfinite(beta):
        raise ValueError("beta must be in [0, 1)")
    # Avoid cancellation in (1-beta)/(1-beta**terms) near beta=1.
    powers = [1.0]
    for _ in range(terms-1):
        powers.append(powers[-1] * beta)
    total = math.fsum(powers)
    return [p/total for p in powers]


def factors(q, k, v, epsilon=0.0):
    q, k, v = matrix(q, 'q', True), matrix(k, 'k', True), matrix(v, 'v')
    if len(q) != len(k) or len(q) != len(v) or len(q[0]) != len(k[0]):
        raise ValueError("q, k, v must share token count; q and k share width")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (float, int)):
        raise ValueError("epsilon must be numeric")
    try:
        epsilon = float(epsilon)
    except OverflowError:
        raise ValueError("epsilon must be finite") from None
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    try:
        sums = [finite(math.fsum(col)) for col in zip(*k)]
        den = [finite(math.fsum(x*y for x,y in zip(row,sums)) + epsilon) for row in q]
    except (OverflowError, ValueError):
        raise ValueError("arithmetic overflow: rescale inputs") from None
    if any(d <= 0 for d in den):
        raise ValueError("each row must have positive normalization mass")
    qt = [[finite(x/d) for x in row] for row,d in zip(q,den)]
    return qt, k, v


def diffuse(q, k, v, beta=0.5, terms=3, epsilon=0.0):
    """Return sum_k w_k S**k V without forming token-by-token S.

    S=(D+epsilon I)^-1 Q K^T, D=diag(Q K^T 1).
    epsilon=0 is exactly row stochastic when normalization masses are positive.
    Positive epsilon instead loses mass; it is not an exact random walk.
    """
    w = weights(beta, terms)
    qt, k, v = factors(q,k,v,epsilon)
    out = [[w[0]*x for x in row] for row in v]
    if terms == 1 or beta == 0:
        return out
    kt = transpose(k)
    # For h>=1: S**h V = qt (K^T qt)**(h-1) K^T V.
    state = mm(kt,v)
    transition = mm(kt,qt)
    for hop in range(1,terms):
        contribution = mm(qt,state)
        out = [[finite(a+w[hop]*b) for a,b in zip(ra,rb)]
               for ra,rb in zip(out,contribution)]
        if hop+1 < terms:
            state = mm(transition,state)
    return out


def dense_transition(q,k,epsilon=0.0):
    """Small diagnostic only: explicitly materializes N x N transition."""
    qt, k, _ = factors(q,k,[[0.] for _ in q],epsilon)
    return mm(qt,transpose(k))
