"""Check the complete grid and independently reconstruct matched-risk decisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit import F,critical,capacities_and_edits,independently_compute,pick
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'protocol.json').read_text());counts={};nrows=0
keys=['beta','good_amplitude','bad_amplitude','trusted_size','epsilon','method','condition']
for d in pd.read_csv(ROOT/'results/runs.csv.gz',chunksize=108000):
 for key,sub in d.groupby(keys,sort=False):
  assert key not in counts
  assert len(sub)==2000 and set(sub.seed)==set(range(90000,92000))
  counts[key]=len(sub)
 nrows+=len(d)
assert len(counts)==3*2*5*3*9*3 and nrows==4860000
logs=np.load(ROOT/'results/logs.npz');U=logs['main_0.03_19000'];V=logs['trusted_0.03_1000'];ut=(U@F.T).T;vt=(V@F.T).T
raw=pd.read_csv(ROOT/'matched_risk_runs.csv.gz');err=0.
for (g,b,method),d in raw.groupby(['good_amplitude','bad_amplitude','method']):
 d=d.sort_values('seed');B=2*np.array([g,b]);a,c,den={'pooled_equal':(1,1,20000),'main_only':(1,0,19000),'trusted_only':(0,1,1000),'evidence_weighted':(4,19,95000)}[method]
 h=critical(19000,1000,a,c,.001,.499);corr=a*200;cs=[];sels=[]
 for op in range(3):
  ed=capacities_and_edits(U,200,op==0) if op<2 else np.zeros_like(U)
  score=B[:,None]*(a*(ut+(ed@(1-2*F).T).T)+c*vt-corr-h+.5)/den
  cs.append(score);sels.append(pick(score))
 sels=np.array(sels);rank=np.where(sels==1,0,np.where(sels<0,1,2));op=rank.argmin(axis=0);se=sels[op,np.arange(2000)];sc=np.stack(cs)[op,:,np.arange(2000)].T
 assert np.array_equal(se,d.selected.to_numpy()) and np.array_equal(op,d.attack_option.to_numpy())
 ee=np.max(abs(sc-d[['score0','score1']].to_numpy().T));err=max(err,float(ee));assert ee<1e-12
 truth=np.array([2*g*.03,-2*b*.001]);utility=np.where(se<0,0,truth[np.maximum(se,0)]);assert np.max(abs(utility-d.improvement))<1e-12
report=dict(passed=True,main_grid_cells=len(counts),main_decisions=nrows,matched_risk_decisions=len(raw),max_matched_score_error=err,seed_count_per_cell=2000)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
# Post-hoc monetary-cost/risk diagnostic checked with direct binomial survival.
from scipy.stats import binom
cost=pd.read_csv(ROOT/'cost_risk_runs.csv.gz')
for (g,b,rho,nt,h),df in cost.groupby(['good_amplitude','bad_amplitude','rho','trusted_count','threshold']):
 assert binom.sf(h-1,nt,.499)+1e-12<=.001 and binom.sf(h-2,nt,.499)+1e-12>.001
 df=df.sort_values('seed');tt=(logs[f'trusted_0.03_{nt}']@F.T).T
 se=pick(2*np.array([g,b])[:,None]*(tt-h+.5)/nt)
 assert np.array_equal(se,df.selected)
 u=np.where(se<0,0,np.array([2*g*.03,-2*b*.001])[np.maximum(se,0)])
 assert np.max(abs(u-df.improvement))<1e-12
report['posthoc_cost_risk_decisions']=len(cost)
(ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
