"""One-row count-aware tail proxies: OpenAttention v2, equations 1 and 8.

Inputs are already scaled logits. Groups and exact indices are supplied, not
learned or selected here. Masks and exact-token removal precede every mean.
"""

from dataclasses import dataclass
import math


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numbers, not bools")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite numbers") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite numbers")
    return value


def _inputs(logits, values, mask):
    try:
        scores = tuple(_finite(x, "logits") for x in logits)
        rows = tuple(tuple(_finite(x, "values") for x in row) for row in values)
    except TypeError as exc:
        raise ValueError("logits and value rows must be iterable") from exc
    if not scores or len(scores) != len(rows) or not rows[0]:
        raise ValueError("provide one nonempty value row per logit")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("value rows must have the same width")
    try:
        visible = (True,)*len(scores) if mask is None else tuple(mask)
    except TypeError as exc:
        raise ValueError("mask must be an iterable of bools") from exc
    if len(visible) != len(scores) or any(type(x) is not bool for x in visible):
        raise ValueError("mask must have exactly one bool per token")
    if not any(visible):
        raise ValueError("attention needs at least one visible token")
    return scores, rows, visible


def _indices(indices, n, label):
    try:
        indices = tuple(indices)
    except TypeError as exc:
        raise ValueError(f"{label} must be iterable") from exc
    if any(type(i) is not int or not 0 <= i < n for i in indices):
        raise ValueError(f"{label} must contain integer token indices in range")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} must not contain duplicates")
    return indices


def _mean(values):
    # Scaling avoids overflow when an arithmetic sum would exceed float range.
    scale = max(abs(x) for x in values)
    return 0.0 if scale == 0 else (math.fsum(x/scale for x in values)/len(values))*scale


def _softmax(scores):
    peak = max(scores)
    weights = tuple(math.exp(s-peak) for s in scores)
    total = math.fsum(weights)
    return tuple(w/total for w in weights)


def _mix(weights, rows):
    output = []
    for column in zip(*rows, strict=True):
        scale = max(abs(x) for x in column)
        value = 0.0 if scale == 0 else math.fsum(w*(x/scale) for w, x in zip(weights, column, strict=True))*scale
        if not math.isfinite(value):
            raise ValueError("weighted value is outside finite floating-point range")
        output.append(value)
    return tuple(output)


@dataclass(frozen=True)
class Proxy:
    members: tuple[int, ...]
    count: int
    mean_logit: float
    mean_value: tuple[float, ...]
    centered_logit: float


@dataclass(frozen=True)
class Read:
    output: tuple[float, ...]
    token_weights: tuple[float, ...]
    component_weights: tuple[float, ...]
    centered_component_logits: tuple[float, ...]
    exact_indices: tuple[int, ...]
    proxies: tuple[Proxy, ...]


def dense_attention(logits, values, *, mask=None):
    """Stable dense baseline over visible tokens; output and full token weights."""
    scores, rows, visible = _inputs(logits, values, mask)
    indices = tuple(i for i, ok in enumerate(visible) if ok)
    anchor = max(scores[i] for i in indices)
    centered = tuple(scores[i]-anchor for i in indices)
    weights = _softmax(centered)
    expanded = [0.0]*len(scores)
    for i, w in zip(indices, weights, strict=True):
        expanded[i] = w
    return Read(_mix(weights, tuple(rows[i] for i in indices)), tuple(expanded),
                weights, centered, indices, ())


def tail_proxy_attention(logits, values, groups, *, exact=(), mask=None,
                         count_weighting=True):
    """Retain exact tokens and replace each nonempty residual group by a proxy.

    groups must partition ALL token positions once, including exact and masked
    positions. We remove those positions before forming each residual proxy.
    Component order is visible exact indices followed by nonempty groups.
    count_weighting=False is an explicitly incorrect no-count ablation.
    """
    scores, rows, visible = _inputs(logits, values, mask)
    if type(count_weighting) is not bool:
        raise ValueError("count_weighting must be a bool")
    n = len(scores)
    exact = _indices(exact, n, "exact")
    try:
        groups = tuple(_indices(group, n, "group") for group in groups)
    except TypeError as exc:
        raise ValueError("groups must be an iterable of index groups") from exc
    flattened = tuple(i for group in groups for i in group)
    if any(not group for group in groups) or sorted(flattened) != list(range(n)):
        raise ValueError("nonempty groups must partition every token exactly once")
    exact_set = set(exact)
    active_exact = tuple(i for i in exact if visible[i])
    component_bases = [scores[i] for i in active_exact]
    component_values = [rows[i] for i in active_exact]
    pending = []
    for group in groups:
        members = tuple(i for i in group if visible[i] and i not in exact_set)
        if not members:
            continue
        mean_score = _mean(tuple(scores[i] for i in members))
        mean_value = tuple(_mean(tuple(rows[i][d] for i in members)) for d in range(len(rows[0])))
        pending.append((members, mean_score, mean_value))
        component_bases.append(mean_score)
        component_values.append(mean_value)
    # Center the component means BEFORE adding log(count). In particular, a
    # common offset such as 1e308 must not swallow the small count correction.
    # Choosing the anchor after pooling also handles cancellation within groups.
    anchor = max(component_bases)
    component_scores = [base-anchor for base in component_bases]
    proxies = []
    for j, (members, mean_score, mean_value) in enumerate(pending, len(active_exact)):
        component_scores[j] += math.log(len(members)) if count_weighting else 0.0
        proxies.append(Proxy(members, len(members), mean_score, mean_value, component_scores[j]))
    component_weights = _softmax(component_scores)
    token_weights = [0.0]*n
    for i, weight in zip(active_exact, component_weights[:len(active_exact)], strict=True):
        token_weights[i] = weight
    for proxy, weight in zip(proxies, component_weights[len(active_exact):], strict=True):
        for i in proxy.members:
            token_weights[i] = weight/proxy.count
    return Read(_mix(component_weights, component_values), tuple(token_weights),
                component_weights, tuple(component_scores), active_exact, tuple(proxies))
