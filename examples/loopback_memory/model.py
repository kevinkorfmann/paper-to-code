"""Original scalar/vector depth reader for Loopback eqs. 14–16, not a model."""
import math


def vector(xs, name):
    try:
        out = [float(x) for x in xs]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite vector') from exc
    if not out or not all(math.isfinite(x) for x in out):
        raise ValueError(name + ' must be a nonempty finite vector')
    return out


def prior(count, stack_size, retention):
    """Weights after count block applications; decay includes the final boundary."""
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    if type(stack_size) is not int or stack_size < 1:
        raise ValueError('stack_size must be a positive integer')
    retention = float(retention)
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError('retention must be in (0, 1]')
    logs = [(count // stack_size - i // stack_size) * math.log(retention)
            for i in range(count)]
    mass = math.fsum(math.exp(x) for x in logs)
    if mass == 0 or not math.isfinite(mass):
        raise ValueError('total coefficient mass is not representable')
    return logs, mass


def allocate(scores, stack_size, retention):
    """Stable log-space allocation, including priors too tiny to exponentiate."""
    scores = vector(scores, 'scores')
    logs, mass = prior(len(scores), stack_size, retention)
    logits = [a + b for a, b in zip(logs, scores)]
    if not all(math.isfinite(x) for x in logits):
        raise ValueError('combined logits overflow')
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = math.fsum(weights)
    return [mass * x / total for x in weights]


def read(initial, updates, query, stack_size, retention, epsilon=1e-6):
    """Return replacement mixture and coefficients, using supplied query/update bank."""
    initial, query = vector(initial, 'initial'), vector(query, 'query')
    updates = [vector(u, 'update') for u in updates]
    d = len(initial)
    if not updates or len(query) != d or any(len(u) != d for u in updates):
        raise ValueError('nonempty updates and matching dimensions required')
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    scores = []
    for u in updates:
        # hypot avoids overflow when squaring a finite update. This equals
        # sqrt(mean(u**2) + epsilon) without constructing the squared norm.
        norm = math.hypot(*(x / math.sqrt(d) for x in u), math.sqrt(epsilon))
        score = math.fsum(q * (x / norm) / math.sqrt(d) for q, x in zip(query, u))
        scores.append(score)
    coefficients = allocate(scores, stack_size, retention)
    result = [math.fsum([initial[j]] + [c*u[j] for c,u in zip(coefficients, updates)])
              for j in range(d)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError('readout overflow')
    return result, coefficients


def rollout(initial, blocks, queries, retention, epsilon=1e-6):
    """Toy supplied callable blocks. Store updates from the current trajectory."""
    initial = vector(initial, 'initial')
    if not blocks:
        raise ValueError('at least one block required')
    state = initial[:]
    updates, states, weights = [], [], []
    for s, q in enumerate(queries):
        output = vector(blocks[s % len(blocks)](state[:]), 'block output')
        if len(output) != len(state):
            raise ValueError('block changed dimension')
        updates.append([a-b for a,b in zip(output, state)])
        state, c = read(initial, updates, q, len(blocks), retention, epsilon)
        states.append(state[:])
        weights.append(c)
    return states, updates, weights
