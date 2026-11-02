"""Reference-query tangent summaries of log attention mass (ZHA, section III-A)."""
from dataclasses import dataclass
import math


def vector(values):
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError('Expected a nonempty vector')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
        raise ValueError('Vector entries must be finite real numbers')
    return tuple(float(x) for x in values)


def matrix(rows):
    if not isinstance(rows,(list,tuple)) or not rows:
        raise ValueError('Expected a nonempty matrix')
    rows=tuple(vector(row) for row in rows)
    if any(len(row)!=len(rows[0]) for row in rows): raise ValueError('Ragged matrix')
    return rows


def finite(value):
    if not math.isfinite(value): raise ValueError('Floating-point range exceeded')
    return value


def dot_scaled(q,k):
    q,k=vector(q),vector(k)
    if len(q)!=len(k): raise ValueError('Vector dimension mismatch')
    try:
        return finite(math.fsum(a*b/math.sqrt(len(q)) for a,b in zip(q,k)))
    except (OverflowError,ValueError) as error:
        raise ValueError('Dot product exceeds floating-point range') from error


def normalized(logits):
    z=vector(logits);maximum=max(z)
    weights=[math.exp(x-maximum) for x in z]
    total=math.fsum(weights)
    probabilities=tuple(x/total for x in weights)
    logs=tuple(x-maximum-math.log(total) for x in z)
    return probabilities,logs,finite(maximum+math.log(total))


def log_mass(keys,query):
    keys=matrix(keys);q=vector(query)
    return normalized([dot_scaled(q,k) for k in keys])[2]


def entropy(probabilities,log_probabilities):
    return math.fsum(-p*lp for p,lp in zip(probabilities,log_probabilities) if p)


@dataclass(frozen=True)
class Tangent:
    key: tuple
    bias: float

    def __post_init__(self):
        object.__setattr__(self,'key',vector(self.key))
        if isinstance(self.bias,bool) or not isinstance(self.bias,(int,float)) or not math.isfinite(self.bias):
            raise ValueError('Bias must be finite')


def summarize(keys,reference):
    """Attention-weighted key and entropy bias; a tangent at the supplied reference."""
    keys=matrix(keys);r=vector(reference)
    probabilities,logs,_=normalized([dot_scaled(r,k) for k in keys])
    compressed=tuple(finite(math.fsum(p*k[j] for p,k in zip(probabilities,keys))) for j in range(len(r)))
    return Tangent(compressed,entropy(probabilities,logs))


def score(summary,query):
    if not isinstance(summary,Tangent): raise ValueError('Expected a Tangent')
    return finite(dot_scaled(query,summary.key)+summary.bias)


def envelope(summaries,query):
    if not summaries: raise ValueError('At least one tangent is required')
    return max(score(s,query) for s in summaries)


def merge(children,reference):
    """Merge biased child summaries. Exact flat equivalence needs a shared reference."""
    if not children: raise ValueError('At least one child is required')
    r=vector(reference)
    probabilities,logs,_=normalized([score(child,r) for child in children])
    compressed=tuple(finite(math.fsum(p*child.key[j] for p,child in zip(probabilities,children))) for j in range(len(r)))
    bias=finite(entropy(probabilities,logs)+math.fsum(p*child.bias for p,child in zip(probabilities,children)))
    return Tangent(compressed,bias)


def chunk_probabilities(keys,query,references,chunk_size,visible=None):
    """Flat diagnostic only: mask a prefix, compress its chunks and normalize scores.

    This is not ZHA's multilevel top-p traversal or its incomplete-chunk policy.
    """
    keys=matrix(keys);q=vector(query);refs=matrix(references)
    if len(q)!=len(keys[0]) or any(len(r)!=len(q) for r in refs): raise ValueError('Dimension mismatch')
    if type(chunk_size) is not int or chunk_size<1: raise ValueError('Invalid chunk size')
    visible=len(keys) if visible is None else visible
    if type(visible) is not int or not 1<=visible<=len(keys): raise ValueError('Invalid visible prefix')
    chunks=[keys[i:min(i+chunk_size,visible)] for i in range(0,visible,chunk_size)]
    exact=[log_mass(chunk,q) for chunk in chunks]
    approx=[envelope([summarize(chunk,r) for r in refs],q) for chunk in chunks]
    return {'exact_log_masses':exact,'estimated_log_masses':approx,
            'exact_chunk_probabilities':list(normalized(exact)[0]),
            'estimated_chunk_probabilities':list(normalized(approx)[0])}
