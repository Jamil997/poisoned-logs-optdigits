from pathlib import Path
import sys,time,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));from develop import core,ROOT,PARENT

def designs(p):
    delta=p[1:]-p[0];N=p.shape[1];base=(.8*p[0]+.02)/N
    w=np.max(abs(delta),axis=0);joint=.99*w/w.sum()+.01*base
    return {f'joint_{v:g}':(1-v)*base+v*joint for v in [0,.25,.5,.75,1]}

def spec(p,Q,n,alpha=.05):
    delta=(p[1:]-p[0])/p.shape[1];c=delta/Q
    B=np.max(abs(c),axis=(1,2));V=.25*np.sum(Q*c*c,axis=(1,2))
    t=np.log(len(delta)/alpha);a=B*t/(3*n)
    return dict(c=c,B=B,V=V,rad=a+np.sqrt(a*a+2*V*t/n))

def sample(data,Q,n,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,71]));u=rng.random(n)
    pairs=np.searchsorted(np.cumsum(Q.ravel()),u,side='right');x,a=np.divmod(pairs,10)
    return x,a,(a==data['y'][x]).astype(float)

def run():
    data=dict(np.load(PARENT/'frozen_models/models.npz'));p=data['p'];y=data['y'];libs=json.loads((PARENT/'subset_protocol.json').read_text())['libraries']
    truth=np.mean(p[1:,np.arange(len(y)),y]-p[0,np.arange(len(y)),y],axis=1);rows=[]
    for name,Q in designs(p).items():
      for n in [5000,20000]:
       for lib in libs:
        inds=lib['candidate_indices'];ps=p[[0]+[i+1 for i in inds]];tr=truth[inds];s=spec(ps,Q,n)
        for seed in range(32000,32020):
          x,a,r=sample(data,Q,n,seed);c=s['c'][:,x,a];gate=dict(c=c,off=-.5*c.mean(axis=1)-s['rad']);k=int(n*.005)
          lower,upper=core.envelopes(gate,r,k,k);poss=core.possible_outcomes(lower,upper)
          rows.append(dict(logger=name,n=n,library=lib['name'],seed=seed,utility_floor=min(core.utility(j,tr) for j in poss),harm_possible=int(any(core.utility(j,tr)<0 for j in poss)),fallback_possible=int(-1 in poss),clean_utility=core.utility(core.choose(core.score(gate,r,k)),tr),radius=s['rad'].mean(),maxc=s['B'].max()))
    core.write_csv(ROOT/'joint_development.csv',rows)
    import pandas as pd
    s=pd.DataFrame(rows).groupby(['logger','n']).mean(numeric_only=True);s.to_csv(ROOT/'joint_development_summary.csv');print(s.to_string())
    core.write_json(ROOT/'joint_development_metadata.json',dict(source_sha256=core.sha(__file__),protocol_sha256=core.sha(ROOT/'joint_development_protocol.json'),outcomes=len(rows)))
if __name__=='__main__':run()
