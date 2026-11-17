"""Original GQA routing reference with supplied centroids, not a clustering engine."""
import math


def matrix(rows, name):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(f'{name} must be a nonempty matrix')
    if any(not isinstance(r, (list, tuple)) or not r for r in rows):
        raise ValueError(f'{name} requires nonempty rows')
    d = len(rows[0])
    if any(len(r) != d for r in rows):
        raise ValueError(f'{name} must be rectangular')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for r in rows for v in r):
        raise ValueError(f'{name} requires finite numbers')
    return d


def unit(v):
    n = math.hypot(*v)
    if n == 0 or not math.isfinite(n):
        raise ValueError('nonzero finite vector norm required')
    return [x/n for x in v]


def softmax(scores):
    if not scores or any(not math.isfinite(x) for x in scores):
        raise ValueError('finite nonempty scores required')
    e = [math.exp(x-max(scores)) for x in scores]
    return [x/math.fsum(e) for x in e]


def route(queries, centroids, labels, query_index, dense_positions, budget,
          temperature=0.1, beta=0.25):
    """One position, multiple GQA query heads, shared selected token indices.

    Centroids/labels are supplied, not fitted. Zero-based token positions.
    Budget counts historical tokens outside dense_positions; last block may
    overshoot. Softmax includes every centroid, even one with no visible member.
    """
    d = matrix(queries, 'queries')
    if matrix(centroids, 'centroids') != d:
        raise ValueError('query/centroid dimensions must match')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('nonnegative finite size penalty required')
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    if not labels or any(isinstance(c,bool) or not isinstance(c,int) or not 0 <= c < len(centroids) for c in labels):
        raise ValueError('every token needs a valid cluster label')
    if isinstance(query_index,bool) or not isinstance(query_index,int) or not 0 <= query_index < len(labels):
        raise ValueError('query index out of range')
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0 <= i <= query_index for i in dense_positions):
        raise ValueError('dense positions must be causally visible')
    if len(set(dense_positions)) != len(dense_positions):
        raise ValueError('dense positions must be unique')
    # Only accept the unit centroids specified by the method; do not silently
    # change arbitrary centroids. Normalize queries for routing, never attention.
    if any(not math.isclose(math.hypot(*c),1,rel_tol=1e-12,abs_tol=1e-12) for c in centroids):
        raise ValueError('unit centroids required')
    logits = [[math.fsum(a*b for a,b in zip(unit(q),c))/temperature for c in centroids] for q in queries]
    head_votes = [softmax(row) for row in logits]
    dense = set(dense_positions)
    blocks = [[i for i in range(query_index+1) if labels[i] == c and i not in dense]
              for c in range(len(centroids))]
    scores = [math.fsum(row[c] for row in head_votes)/(1+len(blocks[c]))**beta
              for c in range(len(centroids))]
    order = sorted(range(len(centroids)),key=lambda c:(-scores[c],c))
    chosen, historical = [], []
    for c in order:
        if len(historical) >= budget:
            break
        if blocks[c]:
            chosen.append(c)
            historical.extend(blocks[c])
    return {'head_votes':head_votes,'scores':scores,'clusters':chosen,
            'historical_count':len(historical),'indices':sorted(dense|set(historical)),
            'visible_block_sizes':[len(b) for b in blocks]}


def attend(queries, keys, values, indices):
    """Exact attention on the selected union using ORIGINAL, unnormalized Q/K."""
    d=matrix(queries,'queries')
    if matrix(keys,'keys') != d:
        raise ValueError('query/key dimensions differ')
    dv=matrix(values,'values')
    if len(keys)!=len(values) or not indices or len(set(indices))!=len(indices):
        raise ValueError('aligned KV and a nonempty unique selection required')
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(keys) for i in indices):
        raise ValueError('invalid selected index')
    result=[]
    for q in queries:
        weights=softmax([math.fsum(a*b for a,b in zip(q,keys[i]))/math.sqrt(d) for i in indices])
        result.append([math.fsum(w*values[i][j] for w,i in zip(weights,indices)) for j in range(dv)])
    return result
