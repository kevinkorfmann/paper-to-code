"""Small dense SwiLA recurrence; supplied projections, priors and learning rates.

State shape: [expert][output coordinate][key coordinate], with square D x D
experts. Write/read logits and learning rates have shape [expert][coordinate].
No training, temporal persistence, decay, load balancing or optimized kernel.
"""

import math


def finite(x):
    if type(x) not in (int, float):
        raise ValueError('Expected finite real numbers, excluding booleans')
    try:
        result = float(x)
    except OverflowError as error:
        raise ValueError('Number exceeds the supported finite range') from error
    if not math.isfinite(result):
        raise ValueError('Expected finite real numbers, excluding booleans')
    return result


def vector(x, n):
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise ValueError('Wrong vector shape')
    return [finite(v) for v in x]


def matrix(x, rows, columns):
    if not isinstance(x, (list, tuple)) or len(x) != rows:
        raise ValueError('Wrong matrix shape')
    return [vector(row, columns) for row in x]


def state_copy(state):
    if not isinstance(state, (list, tuple)) or not state or not isinstance(state[0], (list, tuple)) or not state[0]:
        raise ValueError('Expected at least one nonempty expert matrix')
    d = len(state[0])
    return [matrix(expert, d, d) for expert in state]


def dot(a, b):
    # Shapes and finite inputs are validated at the public entry points.
    try:
        return finite(math.fsum(x*y for x, y in zip(a, b)))
    except (OverflowError, ValueError) as error:
        raise ValueError('Dot product exceeds the supported finite range') from error


def column_softmax(logits):
    """Normalize across experts independently for every output coordinate."""
    if not isinstance(logits, (list, tuple)) or not logits or not isinstance(logits[0], (list, tuple)) or not logits[0]:
        raise ValueError('Expected a nonempty logit matrix')
    j, d = len(logits), len(logits[0])
    logits = matrix(logits, j, d)
    result = [[0.]*d for _ in range(j)]
    for coordinate in range(d):
        maximum = max(row[coordinate] for row in logits)
        weights = [math.exp(row[coordinate]-maximum) for row in logits]
        total = math.fsum(weights)
        for expert in range(j):
            result[expert][coordinate] = weights[expert]/total
    return result


def update(state, key, value, write_logits, beta):
    """Equations 3.3–3.4, all errors evaluated at the SAME old state.

    Softmax(write_logits - error²/2) is algebraically equivalent to using
    a softmax prior times a unit-variance Gaussian likelihood.
    """
    old = state_copy(state)
    j, d = len(old), len(old[0])
    key, value = vector(key, d), vector(value, d)
    logits, rates = matrix(write_logits, j, d), matrix(beta, j, d)
    if any(not 0 <= x <= 1 for row in rates for x in row):
        raise ValueError('Learning rates must lie in [0,1]')
    errors = [[finite(value[c]-dot(old[e][c], key)) for c in range(d)] for e in range(j)]
    scores = [[finite(logits[e][c]-.5*errors[e][c]*errors[e][c]) for c in range(d)] for e in range(j)]
    responsibilities = column_softmax(scores)
    new = [[[finite(old[e][c][k]+rates[e][c]*responsibilities[e][c]*errors[e][c]*key[k])
             for k in range(d)] for c in range(d)] for e in range(j)]
    return new, responsibilities


def read(state, query, read_logits):
    """Equation 3.5: use query priors, never the target-conditioned write posterior."""
    state = state_copy(state)
    j, d = len(state), len(state[0])
    query = vector(query, d)
    weights = column_softmax(matrix(read_logits, j, d))
    return [dot([weights[e][c] for e in range(j)], [dot(state[e][c], query) for e in range(j)])
            for c in range(d)]


def run(initial, tokens):
    """Inclusive causal scan: write token t, then read its query. Return final state and outputs."""
    state = state_copy(initial)
    outputs = []
    for token in tokens:
        state, _ = update(state, token['key'], token['value'], token['write_logits'], token['beta'])
        outputs.append(read(state, token['query'], token['read_logits']))
    return state, outputs
