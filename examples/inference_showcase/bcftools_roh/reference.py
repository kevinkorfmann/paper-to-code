"""Independent two-state probability model for a consecutive synthetic marker grid."""

import math


def emissions(pl, frequency):
    if len(pl)!=3 or any(type(x) is not int or not 0<=x<=200 for x in pl):
        raise ValueError('Expected three integer PLs in [0,200]')
    if isinstance(frequency,bool) or not math.isfinite(frequency) or not 0<frequency<1:
        raise ValueError('ALT frequency must lie strictly between zero and one')
    likelihood=[10**(-x/10) for x in pl]
    f=frequency
    return ((1-f)**2*likelihood[0]+2*f*(1-f)*likelihood[1]+f*f*likelihood[2],
            (1-f)*likelihood[0]+f*likelihood[2])  # HW, AZ


def infer(pls, frequencies, switch=.02):
    """Scaled sum-product posterior and log-space max-product path, prior (1/2,1/2).

    Symmetric per-marker switch probability, consecutive positions, no map.
    Returned marginals are this independent model's values, not BCFtools Q fields.
    """
    if not pls or len(pls)!=len(frequencies):raise ValueError('Expected matching nonempty site arrays')
    if isinstance(switch,bool) or not math.isfinite(switch) or not 0<switch<.5:
        raise ValueError('Switch probability must lie in (0,0.5)')
    e=[emissions(pl,f) for pl,f in zip(pls,frequencies)]
    transition=[[1-switch,switch],[switch,1-switch]]
    forward=[];current=[.5,.5]
    for emission in e:
        current=[emission[s]*sum(current[p]*transition[p][s] for p in range(2)) for s in range(2)]
        scale=sum(current);current=[x/scale for x in current];forward.append(current)
    posterior=[None]*len(e);backward=[1.,1.]
    for i in range(len(e)-1,-1,-1):
        weights=[forward[i][s]*backward[s] for s in range(2)]
        posterior[i]=weights[1]/sum(weights)
        backward=[sum(transition[s][q]*e[i][q]*backward[q] for q in range(2)) for s in range(2)]
        scale=sum(backward);backward=[x/scale for x in backward]
    scores=[math.log(.5)+math.log(x) for x in e[0]];ancestors=[]
    for emission in e[1:]:
        choices=[];next_scores=[]
        for s in range(2):
            candidates=[scores[p]+math.log(transition[p][s]) for p in range(2)]
            parent=max(range(2),key=lambda p:candidates[p])
            choices.append(parent);next_scores.append(candidates[parent]+math.log(emission[s]))
        ancestors.append(choices);scores=next_scores
    state=max(range(2),key=lambda s:scores[s]);path=[state]
    for choices in reversed(ancestors):state=choices[state];path.append(state)
    return {'viterbi':list(reversed(path)),'posterior_AZ':posterior}


def phred_error(probability):
    if not 0<=probability<=1:raise ValueError('Probability outside [0,1]')
    return -10*math.log10(max(probability,1e-99))
