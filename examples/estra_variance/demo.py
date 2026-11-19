"""ESTRA VCM equations 6–8, fixed clusters and selected indices, scalar values."""
import json
import math


def dot(a,b): return sum(x*y for x,y in zip(a,b))


def softmax_output(logits,values):
    top=max(logits);weights=[math.exp(x-top) for x in logits]
    return sum(w*v for w,v in zip(weights,values))/sum(weights)


def validate(queries,keys,values,clusters,selected,cap):
    if not queries or not keys or not queries[0]:raise ValueError('empty input')
    d=len(queries[0]);n=len(keys)
    if any(len(x)!=d or any(not math.isfinite(v) for v in x) for x in queries+keys):raise ValueError('bad vectors')
    if len(values)!=n or any(not math.isfinite(v) for v in values):raise ValueError('bad values')
    flat=[i for c in clusters for i in c]
    if any(not c for c in clusters) or any(type(i) is not int for i in flat) or sorted(flat)!=list(range(n)):raise ValueError('clusters must partition keys')
    if len(set(selected))!=len(selected) or any(type(i) is not int or not 0<=i<n for i in selected):raise ValueError('invalid selection')
    if not math.isfinite(cap) or cap<0:raise ValueError('invalid cap')
    return d


def approximate(queries,keys,values,clusters,selected,cap=1.0):
    d=validate(queries,keys,values,clusters,selected,cap)
    alpha=sum(dot(q,q) for q in queries)/(2*d*d*len(queries))
    representatives=[]
    for c in clusters:
        remaining=[j for j in c if j not in selected]
        if not remaining:continue
        center=[sum(keys[j][a] for j in c)/len(c) for a in range(d)]
        value=sum(values[j] for j in c)/len(c)
        m2=sum(sum((keys[j][a]-center[a])**2 for a in range(d)) for j in remaining)/len(remaining)
        representatives.append((center,value,len(remaining),min(alpha*m2,cap)))
    out=[]
    for q in queries:
        logits=[dot(q,keys[j])/math.sqrt(d) for j in selected]
        vals=[values[j] for j in selected]
        for center,value,count,delta in representatives:
            logits.append(dot(q,center)/math.sqrt(d)+math.log(count)+delta);vals.append(value)
        out.append(softmax_output(logits,vals))
    return out


def dense(queries,keys,values):
    return [softmax_output([dot(q,k)/math.sqrt(len(q)) for k in keys],values) for q in queries]


def demo():
    cases={
      'symmetric_spread':([[1.0]],[[-.5],[.5],[0.]],[1.,1.,0.],[[0,1],[2]],[2]),
      'orthogonal_spread':([[1.,0.]],[[0.,-1.],[0.,1.],[0.,0.]],[1.,1.,0.],[[0,1],[2]],[2]),
      'selection_mean_shift':([[1.]],[[-1.],[1.],[0.]],[1.,1.,0.],[[0,1],[2]],[1,2])}
    result={}
    for name,args in cases.items():
        exact=dense(*args[:3])[0];mean=approximate(*args,cap=0)[0];vcm=approximate(*args,cap=1)[0]
        result[name]={k:round(v,12) for k,v in dict(dense=exact,mean=mean,vcm=vcm,mean_absolute_error=abs(mean-exact),vcm_absolute_error=abs(vcm-exact)).items()}
    return {'cap':1.0,'cases':result,'note':'Handwritten fixed clusters and selections; no trained model or speed measurement.'}

if __name__=='__main__':print(json.dumps(demo(),indent=2,sort_keys=True))
