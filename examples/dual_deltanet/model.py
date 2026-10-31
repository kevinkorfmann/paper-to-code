"""Sequential and rank-two forms of Dual DeltaNet, equations 5–10/Algorithm 1."""

import math


def _number(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError("Expected finite real numbers")
    return float(x)


def _vector(xs, length=None):
    if not isinstance(xs, (list, tuple)) or not xs or (length is not None and len(xs) != length):
        raise ValueError("Expected a nonempty vector of the required length")
    return [_number(x) for x in xs]


def _matrix(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Expected a nonempty rectangular state")
    first = _vector(rows[0])
    return [first] + [_vector(row, len(first)) for row in rows[1:]]


def unit(vector):
    """L2 normalization without overflowing on large finite vector components."""
    values = _vector(vector)
    scale = max(abs(x) for x in values)
    if scale == 0:
        raise ValueError("Cannot normalize a zero vector")
    scaled = [x/scale for x in values]
    norm = math.sqrt(math.fsum(x*x for x in scaled))
    return [x/norm for x in scaled]


def dot(a, b):
    try:
        value = math.fsum(x*y for x, y in zip(a, b, strict=True))
    except (OverflowError, ValueError) as error:
        raise ValueError("Dot product overflow or dimension mismatch") from error
    if not math.isfinite(value):
        raise ValueError("Dot product exceeds floating-point range")
    return value


def _gate(x):
    value = _number(x)
    if not 0 <= value <= 1:
        raise ValueError("Gates must lie in [0,1]")
    return value


def prepare(state, key, value, alpha, channel_gate, beta, gamma):
    state = _matrix(state)
    nk, nv = len(state), len(state[0])
    key = unit(_vector(key, nk))
    value = _vector(value, nv)
    direction = unit(value)  # The paper's normalized value, not the gated selector.
    alpha = [_gate(x) for x in _vector(alpha, nk)]
    mask = [_gate(x) for x in _vector(channel_gate, nv)]
    selected = unit([m*v for m, v in zip(mask, direction)])
    retained = [[alpha[i]*x for x in row] for i, row in enumerate(state)]
    return retained, key, value, direction, selected, dot(value, selected), _gate(beta), _gate(gamma)


def step(state, key, value, alpha, channel_gate, beta, gamma):
    """Apply key-side correction, then value-side correction to the updated state."""
    retained, k, v, vh, e, tau, beta, gamma = prepare(state, key, value, alpha, channel_gate, beta, gamma)
    nk, nv = len(k), len(v)
    retrieved = [dot(k, [row[j] for row in retained]) for j in range(nv)]
    kda = [[retained[i][j]+beta*k[i]*(v[j]-retrieved[j]) for j in range(nv)] for i in range(nk)]
    address = [dot(row, e) for row in kda]
    result = [[kda[i][j]+gamma*(tau*k[i]-address[i])*vh[j] for j in range(nv)] for i in range(nk)]
    return _matrix(result)


def factorized_step(state, key, value, alpha, channel_gate, beta, gamma):
    """Algorithm 1: both state reads use the retained pre-correction state."""
    retained, k, v, vh, e, tau, beta, gamma = prepare(state, key, value, alpha, channel_gate, beta, gamma)
    nk, nv = len(k), len(v)
    residual_v = [v[j]-dot(k, [row[j] for row in retained]) for j in range(nv)]
    addresses = [dot(row, e) for row in retained]
    target = tau-beta*dot(residual_v, e)
    residual_k = [target*k[i]-addresses[i] for i in range(nk)]
    result = [[retained[i][j]+beta*k[i]*residual_v[j]+gamma*residual_k[i]*vh[j]
               for j in range(nv)] for i in range(nk)]
    return _matrix(result)


def read(state, query):
    """Algorithm 1's readout: S^T normalized(q) / sqrt(key dimension)."""
    state = _matrix(state)
    q = unit(_vector(query, len(state)))
    return [dot(q, [row[j] for row in state])/math.sqrt(len(q)) for j in range(len(state[0]))]


def run(state, tokens, *, factorized=False):
    """Consume supplied token projections/gates in order; read after each update."""
    state = _matrix(state)
    update = factorized_step if factorized else step
    outputs = []
    for token in tokens:
        state = update(state, token['key'], token['value'], token['alpha'], token['channel_gate'], token['beta'], token['gamma'])
        outputs.append(read(state, token['query']))
    return state, outputs
