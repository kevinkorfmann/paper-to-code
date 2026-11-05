"""Supplied-cluster routing, not clustering, learned representatives, or attention.

AttentionIndex (Park and Hwu, arXiv:2611.02508v1), Eq. 1 and Sections 4.1-4.2.
Original MIT-licensed implementation. Every token ID is its zero-based position.
"""

from dataclasses import dataclass
import math


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list or tuple")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{name} must contain real numbers, not booleans")
    try:
        result = tuple(float(x) for x in value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} contains an unrepresentable number") from exc
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    rows = tuple(_vector(row, name) for row in value)
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _sum(terms):
    try:
        result = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise ValueError("nonfinite intermediate result") from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite intermediate result")
    return result


def _score(heads, weights, key):
    return _sum(w * max(0.0, _sum(q * k for q, k in zip(head, key)))
                for head, w in zip(heads, weights))


def _query(heads, weights, width):
    heads = _matrix(heads, "heads")
    weights = _vector(weights, "weights")
    if len(heads) != len(weights) or len(heads[0]) != width:
        raise ValueError("query head, weight, or key dimensions disagree")
    return heads, weights


def native_scores(heads, weights, keys):
    """Eq. 1: sum of weighted, individually rectified head dot products.

    Weights may be signed. ReLU applies before summing heads. No softmax or
    sqrt(d) scaling is added to the supplied indexer projections.
    """
    keys = _matrix(keys, "keys")
    heads, weights = _query(heads, weights, len(keys[0]))
    return tuple(_score(heads, weights, key) for key in keys)


def _ids(value, n, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if any(type(i) is not int or not 0 <= i < n for i in value):
        raise ValueError(f"{name} contains an invalid token ID")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} repeats a token ID")
    return tuple(sorted(value))


@dataclass(frozen=True, init=False)
class PrefixIndex:
    """Immutable, copied prefix and supplied memberships; means are derived.

    All groups must be fixed using only this prefix. The constructor cannot
    detect whether an external caller inferred memberships from future data.
    """

    keys: tuple
    groups: tuple
    sinks: tuple
    representatives: tuple

    def __init__(self, keys, groups, sinks=()):
        keys = _matrix(keys, "keys")
        if not isinstance(groups, (list, tuple)):
            raise ValueError("groups must be a list or tuple")
        groups = tuple(_ids(group, len(keys), "group") for group in groups)
        sinks = _ids(sinks, len(keys), "sinks")
        if any(not group for group in groups):
            raise ValueError("clusters must not be empty")
        members = sinks + tuple(i for group in groups for i in group)
        if sorted(members) != list(range(len(keys))):
            raise ValueError("clusters and sinks must partition the indexed prefix")
        representatives = tuple(
            tuple(_sum(keys[i][d] / len(group) for i in group)
                  for d in range(len(keys[0]))) for group in groups)
        for name, value in (("keys", keys), ("groups", groups), ("sinks", sinks),
                            ("representatives", representatives)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Selection:
    cluster_scores: tuple
    cluster_order: tuple
    reserved_ids: tuple
    candidate_ids: tuple
    selected_ids: tuple
    selected_scores: tuple


def select(index, visible_keys, heads, weights, *, budget, top_k):
    """Reserve sinks/open window, route mean representatives, refine exactly.

    visible_keys is exactly the query's causal prefix, including the current
    token only if the caller's convention allows self-attention. The snapshot
    may be older, never newer. Ties use cluster order then ascending token ID.
    Last-list truncation uses ascending token IDs, a documented teaching choice.
    """
    if not isinstance(index, PrefixIndex):
        raise ValueError("index must be a PrefixIndex")
    keys = _matrix(visible_keys, "visible_keys")
    if len(keys) < len(index.keys) or keys[:len(index.keys)] != index.keys:
        raise ValueError("snapshot is newer than, or inconsistent with, the visible prefix")
    heads, weights = _query(heads, weights, len(keys[0]))
    if (type(budget) is not int or type(top_k) is not int
            or not 1 <= top_k <= budget <= len(keys)):
        raise ValueError("require integer 1 <= top_k <= budget <= visible key count")
    reserved = index.sinks + tuple(range(len(index.keys), len(keys)))
    if len(reserved) > budget:
        raise ValueError("budget cannot hold all sink and open-window keys")
    scores = tuple(_score(heads, weights, rep) for rep in index.representatives)
    order = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))
    candidates = list(reserved)
    for group_id in order:
        remaining = budget - len(candidates)
        if not remaining:
            break
        candidates.extend(index.groups[group_id][:remaining])
    refined = [(i, _score(heads, weights, keys[i])) for i in candidates]
    refined.sort(key=lambda item: (-item[1], item[0]))
    winners = refined[:top_k]
    return Selection(scores, order, reserved, tuple(candidates),
                     tuple(i for i, _ in winners), tuple(s for _, s in winners))


def exhaustive_top_k(heads, weights, keys, top_k):
    """Reference selection with the same deterministic token-ID tie convention."""
    scores = native_scores(heads, weights, keys)
    if type(top_k) is not int or not 1 <= top_k <= len(scores):
        raise ValueError("invalid top_k")
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k])
