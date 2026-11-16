"""Original finite single-head bias-pruning reference, standard library only."""
import math


def matrix(x, name):
    if not isinstance(x, (list, tuple)) or not x or not x[0]:
        raise ValueError(f'{name} must be a nonempty matrix')
    width = len(x[0])
    if any(len(row) != width for row in x):
        raise ValueError(f'{name} must be rectangular')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in x for v in row):
        raise ValueError(f'{name} entries must be finite numbers')
    return width


def softmax(logits):
    if not logits or any(not math.isfinite(v) for v in logits):
        raise ValueError('finite nonempty logits required')
    m = max(logits)
    weights = [math.exp(v - m) for v in logits]
    total = math.fsum(weights)
    return [v / total for v in weights]


def select(bias, budget):
    if not bias or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in bias):
        raise ValueError('finite nonempty bias required')
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(bias):
        raise ValueError('budget must be an integer between one and number of keys')
    # Tie-break by original index; return in original order for auditable alignment.
    return sorted(sorted(range(len(bias)), key=lambda j: (-bias[j], j))[:budget])


def attend(queries, keys, values, bias, budget=None):
    """All queries survive. Weights contain only the selected key columns.

    Bias is added AFTER 1/sqrt(d) scaling. Inputs are supplied projections,
    not learned from tokens. No masks, dropout, batches, or training.
    """
    d = matrix(queries, 'queries')
    if matrix(keys, 'keys') != d:
        raise ValueError('query/key dimensions must match')
    dv = matrix(values, 'values')
    if len(keys) != len(values) or len(keys) != len(bias):
        raise ValueError('keys, values and biases must align')
    kept = select(bias, len(keys) if budget is None else budget)
    weights, outputs = [], []
    for q in queries:
        scores = [math.fsum(a*b for a, b in zip(q, keys[j])) / math.sqrt(d) + bias[j]
                  for j in kept]
        row = softmax(scores)
        output = [math.fsum(w*values[j][v] for w, j in zip(row, kept)) for v in range(dv)]
        if any(not math.isfinite(v) for v in output):
            raise ValueError('output exceeds finite numerical range')
        weights.append(row)
        outputs.append(output)
    return {'indices': kept, 'weights': weights, 'outputs': outputs}
