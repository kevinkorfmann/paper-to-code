"""Deterministic, standard-library demonstration; no fitted model or randomness."""
import argparse
import json
from pathlib import Path
from .model import fit_scale, lag_survival, loss, matched_gain, mixture_gain


def results():
    c,m,r = (0.5,0.5),(0.1,1.0),(0.5,0.95)
    aa=[lag_survival(r,t) for t in (1,10)]
    out={'reference_python':'CPython 3.14.5','randomness':'none','key_energy':c,'row_noise':m,'retention':r,'lags':[]}
    for lag,a in zip((1,10),aa):
        gains={'matched':matched_gain(c,m,a),'unit_direction':fit_scale((1,1),c,m,a),
               'inverse_noise_direction':fit_scale((1/m[0],1/m[1]),c,m,a),
               'unit_signal_matched':matched_gain(c,m,a,unit_signal=True)}
        out['lags'].append({'lag':lag,'survival':a,'controls':{name:{'gain':g,**loss(g,c,m,a)} for name,g in gains.items()}})
    p=(0.5,0.5); avg=tuple((x+y)/2 for x,y in zip(*aa))
    candidates={'mixture_optimum':mixture_gain(c,m,aa,p),'mean_survival_substitution':matched_gain(c,m,avg)}
    out['mixture']={name:{'gain':g,'mse':sum(w*loss(g,c,m,a)['mse'] for w,a in zip(p,aa))} for name,g in candidates.items()}
    out['adaptive_mixture_mse']=sum(w*loss(matched_gain(c,m,a),c,m,a)['mse'] for w,a in zip(p,aa))
    def rounded(x):
        if isinstance(x,float): return round(x,12)
        if isinstance(x,dict): return {k:rounded(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [rounded(v) for v in x]
        return x
    return rounded(out)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check-sample',action='store_true'); args=parser.parse_args()
    text=json.dumps(results(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.check_sample:
        if text!=Path(__file__).with_name('sample_output.json').read_text(): raise SystemExit('sample mismatch')
        print('sample output matches')
    else: print(text,end='')

if __name__=='__main__': main()
