from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment import ROOT,lower_tables,attack_counts

def run():
 n=20000;k=200;B=np.array([.1,.9]);exact=lower_tables(n,B,.01)[0]['binomial'];threshold=int(np.flatnonzero(exact[1]>1e-9)[0]);radius=(threshold-.5)/n-.5
 conf=n*radius**2/(.5+2*radius/3);effective_alpha=2*np.exp(-conf)
 matched=B[:,None]*(np.arange(n+1)/n-.5-radius)
 assert np.flatnonzero(matched[1]>1e-9)[0]==threshold
 logs=np.load(ROOT/'risk_results/logs.npz');rows=[]
 for seed in range(70000,71000):
  counts=logs[str(seed)]
  for name,table in [('binomial',exact),('matched_bernstein',matched)]:
   sel,*_=attack_counts(counts,k,table,k);u=0. if sel<0 else [.003,-.0009][sel]
   rows.append(dict(seed=seed,method=name,selected=sel,improvement=u,harm=int(u<0),fallback=int(sel<0)))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'matched_risk_runs.csv',index=False);p=df.pivot(index='seed',columns='method',values='improvement');diff=100*(p.binomial-p.matched_bernstein).to_numpy();rng=np.random.default_rng(8175);ix=rng.integers(1000,size=(10000,1000));boot=diff[ix].mean(axis=1)
 out=dict(threshold=threshold,bernstein_effective_alpha=float(effective_alpha),difference_pp=float(diff.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),identical_decisions=bool((df.pivot(index='seed',columns='method',values='selected').diff(axis=1).iloc[:,1]==0).all()),summary=df.groupby('method')[['improvement','harm','fallback']].mean().to_dict('index'))
 (ROOT/'matched_risk_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
