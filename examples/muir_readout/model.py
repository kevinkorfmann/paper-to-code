"""Original supplied-moment readout calculation; no trained Muir model."""
import math


def _vector(x, name, positive=False):
    try:
        result = tuple(float(v) for v in x)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector') from exc
    if not result or any(not math.isfinite(v) or (positive and v <= 0) for v in result):
        raise ValueError(f'{name} must be a nonempty finite vector' + (' with positive entries' if positive else ''))
    return result


def moments(key_energy, row_noise, survival, signal_variance=1.0):
    c = _vector(key_energy, 'key_energy', True)
    m = _vector(row_noise, 'row_noise', True)
    a = _vector(survival, 'survival')
    s = float(signal_variance)
    if len(c) != len(m) or len(c) != len(a):
        raise ValueError('moment vectors must have equal length')
    if not math.isclose(math.fsum(c), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError('key energies must sum to one')
    if any(v < 0 or v > 1 for v in a) or not math.isfinite(s) or s <= 0:
        raise ValueError('survival must be in [0,1] and signal variance positive')
    if any(v < 1e-100 or v > 1e100 for v in (*c, *m, s)):
        raise ValueError('positive moments must be in [1e-100,1e100]')
    return c, m, a, s


def loss(gain, key_energy, row_noise, survival, signal_variance=1.0):
    """Muir Eq. 6 under the explicitly supplied diagonal-noise model."""
    c, m, a, s = moments(key_energy, row_noise, survival, signal_variance)
    g = _vector(gain, 'gain')
    if len(g) != len(c):
        raise ValueError('gain dimension mismatch')
    signal = math.fsum(x*y*z for x,y,z in zip(g,c,a))
    noise = math.fsum(x*x*y*z for x,y,z in zip(g,c,m))
    if not all(math.isfinite(v) for v in (signal, noise, s*(1-signal)**2)):
        raise ValueError('gain exceeds supported numerical range')
    return {'signal_gain': signal, 'bias_squared': s*(1-signal)**2,
            'noise_variance': noise, 'mse': s*(1-signal)**2 + noise}


def matched_gain(key_energy, row_noise, survival, signal_variance=1.0, *, unit_signal=False):
    """Eq. 7 (MSE optimum), or separately unit-signal variance optimum."""
    c, m, a, s = moments(key_energy, row_noise, survival, signal_variance)
    z = math.fsum(x*y*y/w for x,y,w in zip(c,a,m))
    if unit_signal and z == 0:
        raise ValueError('unit signal is impossible when no signal survives')
    scale = 1/z if unit_signal else s/(1+s*z)
    result = tuple(scale*y/w for y,w in zip(a,m))
    if not math.isfinite(scale) or any(not math.isfinite(x) for x in result):
        raise ValueError('signal exceeds supported numerical conditioning')
    return result


def fit_scale(direction, key_energy, row_noise, survival, signal_variance=1.0):
    """Best MSE scalar along a fixed nonzero direction, for fair controls."""
    c,m,a,s = moments(key_energy,row_noise,survival,signal_variance)
    d = _vector(direction,'direction')
    if len(d) != len(c) or not any(d):
        raise ValueError('direction must match dimensions and be nonzero')
    magnitude = max(abs(x) for x in d)
    d = tuple(x/magnitude for x in d)
    u = math.fsum(x*y*z for x,y,z in zip(d,c,a))
    n = math.fsum(x*x*y*z for x,y,z in zip(d,c,m))
    scale = s*u/(s*u*u+n)
    return tuple(scale*x for x in d)


def lag_survival(retention, lag):
    """Deterministic row survival r**lag, not a general stochastic-gate law."""
    r = _vector(retention,'retention', True)
    if isinstance(lag,bool) or not isinstance(lag,int) or lag < 1 or any(x >= 1 for x in r):
        raise ValueError('retention must be in (0,1), lag a positive integer')
    return tuple(x**lag for x in r)


def readout(gain, query, state):
    g,q = _vector(gain,'gain'), _vector(query,'query')
    rows = tuple(_vector(r,'state row') for r in state)
    if len(g)!=len(q) or len(rows)!=len(g) or len({len(r) for r in rows})!=1:
        raise ValueError('incompatible readout dimensions')
    return tuple(math.fsum(g[i]*q[i]*rows[i][j] for i in range(len(g))) for j in range(len(rows[0])))


def mixture_gain(key_energy, row_noise, survivals, probabilities, signal_variance=1.0):
    """Exact MSE optimum of a finite supplied lag mixture, via normal equations.

    Unlike substituting mean survival in Eq. 7, includes between-lag variance.
    """
    p = _vector(probabilities,'probabilities',True)
    aa = tuple(survivals)
    if len(aa)!=len(p) or not math.isclose(math.fsum(p),1,rel_tol=0,abs_tol=1e-12):
        raise ValueError('probabilities must match survivals and sum to one')
    mm = [moments(key_energy,row_noise,a,signal_variance) for a in aa]
    c,m,_,s = mm[0]; n=len(c)
    b = [tuple(x*y for x,y in zip(c,t[2])) for t in mm]
    h = [[s*math.fsum(p[k]*b[k][i]*b[k][j] for k in range(len(p))) + (c[i]*m[i] if i==j else 0) for j in range(n)] for i in range(n)]
    rhs=[s*math.fsum(p[k]*b[k][i] for k in range(len(p))) for i in range(n)]
    # Cholesky solve. Positive diagonal noise makes the Hessian positive definite.
    l=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            v=h[i][j]-math.fsum(l[i][k]*l[j][k] for k in range(j))
            if i==j:
                if v<=0: raise ValueError('moments exceed supported numerical conditioning')
                l[i][j]=math.sqrt(v)
            else: l[i][j]=v/l[j][j]
    y=[]
    for i in range(n): y.append((rhs[i]-math.fsum(l[i][j]*y[j] for j in range(i)))/l[i][i])
    x=[0.0]*n
    for i in reversed(range(n)): x[i]=(y[i]-math.fsum(l[j][i]*x[j] for j in range(i+1,n)))/l[i][i]
    return tuple(x)
