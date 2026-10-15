from pathlib import Path
import sys,json,time,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'security'))
import experiment as core
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'boundary'

def loggers(p):
    delta=p[1:]-p[0]
    base=.8*p[0]+.02
    w=np.max(np.abs(delta),axis=0)
    d=w/np.maximum(w.sum(axis=1,keepdims=True),1e-300)
    d=np.where(w.sum(axis=1,keepdims=True)>0,d,.1)
    d=.99*d+.001
    out={f'mix_{v:g}':(1-v)*base+v*d for v in [0,.25,.5,.75,1]}
    out['uniform']=np.full_like(base,.1)
    w=np.sqrt(np.mean(delta**2,axis=0));d=w/np.maximum(w.sum(axis=1,keepdims=True),1e-300)
    out['rms']=.99*np.where(w.sum(axis=1,keepdims=True)>0,d,.1)+.001
    return out

def spec(p,mu,n,alpha=.05,clip=None):
    delta=p[1:]-p[0];c=delta/mu
    if clip is not None:c=np.clip(c,-clip,clip)
    bias=.5*np.mean(np.sum(np.abs(delta-mu*c),axis=2),axis=1)
    B=np.max(np.abs(c),axis=(1,2))
    V=.25*np.mean(np.sum(mu*c*c,axis=2),axis=1)
    t=np.log(len(delta)/alpha);a=B*t/(3*n)
    rad=a+np.sqrt(a*a+2*V*t/n)
    return dict(c=c,rad=rad,bias=bias,B=B,V=V)

def run():
    cfg=json.loads((ROOT/'development_protocol.json').read_text())
    data=dict(np.load(PARENT/'frozen_models/models.npz'));p=data['p']; y=data['y'];N=cfg['development_n']
    libs=json.loads((PARENT/'subset_protocol.json').read_text())['libraries']
    truth=(p[1:,np.arange(len(y)),y]-p[0,np.arange(len(y)),y]).mean(axis=1)
    rows=[];start=time.perf_counter()
    for li,(name,mu) in enumerate(loggers(p).items()):
      cost=float(np.mean(p[0,np.arange(len(y)),y]-mu[np.arange(len(y)),y]))
      for lib in libs:
        inds=lib['candidate_indices'];ps=p[[0]+[i+1 for i in inds]];tr=truth[inds]
        for h in [None,.25,.5,1,2,4,8]:
          s=spec(ps,mu,N,cfg['alpha'],h)
          for seed in range(31000,31020):
            x,a,r=core.sample_log(data,mu,N,li/1000,seed)
            c=s['c'][:,x,a];gate=dict(c=c,off=-.5*c.mean(axis=1)-s['rad']-s['bias'])
            k=int(N*.005);lower,upper=core.envelopes(gate,r,k,k)
            possible=core.possible_outcomes(lower,upper);floor=min(core.utility(j,tr) for j in possible)
            ss=core.score(gate,r,k);chosen=core.choose(ss)
            rows.append(dict(logger=name,library=lib['name'],clip='none' if h is None else h,seed=seed,
                clean_utility=core.utility(chosen,tr),utility_floor=floor,harm_possible=int(any(core.utility(j,tr)<0 for j in possible)),
                fallback_possible=int(-1 in possible),logging_cost=cost,mean_radius=float(s['rad'].mean()),mean_bias=float(s['bias'].mean()),max_c=float(s['B'].max())))
      print(name,round(time.perf_counter()-start,2),flush=True)
    core.write_csv(ROOT/'development.csv',rows)
    import pandas as pd
    df=pd.DataFrame(rows);s=df.groupby(['logger','clip']).agg(floor=('utility_floor','mean'),clean=('clean_utility','mean'),harm_possible=('harm_possible','mean'),fallback=('fallback_possible','mean'),cost=('logging_cost','mean'),radius=('mean_radius','mean'),bias=('mean_bias','mean'),maxc=('max_c','max')).reset_index()
    s.to_csv(ROOT/'development_summary.csv',index=False)
    print(s.to_string(index=False))
    core.write_json(ROOT/'development_metadata.json',dict(seconds=time.perf_counter()-start,source_sha256=core.sha(__file__),protocol_sha256=core.sha(ROOT/'development_protocol.json'),outcomes=len(rows)))
if __name__=='__main__':run()
