"""Post-E8 comparison with the ORIGINAL E7 nonlinear CP scoring rule."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from analyze import contrast
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controlled'))
from allocation import exact_table,allocations
from experiment import favorable_types,score,choose
source=ROOT.parent/'controlled/allocation.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='d3762e348df2357cd88e09a80945e512508f145b1ca2b94315f30614e2017cfa'
logs=np.load(ROOT/'results/logs.npz');u=logs['main_0.03_19000'];v=logs['trusted_0.03_1000'];f=favorable_types();t=(f@(u+v).T).T
primary=pd.read_csv(ROOT/'primary_runs.csv.gz');rows=[];report={}
for g,b in [[.05,.45],[.45,.05]]:
 B=2*np.array([g,b]);truth=np.array([2*g*.03,-2*b*.001]);report[str(g)]={}
 for name,risks in [('uniform',np.array([.005,.005])),('influence_p1',allocations(B,.01)['influence_p1'])]:
  table=exact_table(20000,B,risks);selected=[];sc=[]
  for tt in t:
   options=[]
   for shift in [np.array([-200,200]),np.array([-200,-200]),np.array([0,0])]:
    observed=tt+shift;s=score(table,observed,200)
    # Reconstruct each score without the released table routine.
    corrected=np.maximum(0,observed-200);q=np.array([beta_dist.isf(1-risks[j],int(x),20000-int(x)+1) if x else 0 for j,x in enumerate(corrected)])
    assert np.max(abs(s-B*(q-.5)))<1e-12
    se=choose(s);options.append((se,s))
   se,ss=min(options,key=lambda x:{1:0,-1:1,0:2}[x[0]]);selected.append(se);sc.append(ss)
  se=np.array(selected);util=np.where(se<0,0,truth[np.maximum(se,0)])
  d=pd.DataFrame(dict(seed=np.arange(90000,92000),good_amplitude=g,bad_amplitude=b,method='original_E7_'+name,selected=se,improvement=util,harm=(util<0).astype(int),fallback=(se<0).astype(int)))
  rows.append(d);adapted='pooled_equal' if name=='uniform' else 'pooled_influence'
  p=primary.query('good_amplitude==@g and method==@adapted').sort_values('seed')
  report[str(g)][name]=dict(decision_differences_from_adapted_gate=int(np.sum(se!=p.selected.to_numpy())),utility_pp=float(100*util.mean()),harm=float((util<0).mean()),weighted_evidence_minus_original=contrast(primary.query('good_amplitude==@g and method=="evidence_weighted"'),d))
pd.concat(rows,ignore_index=True).to_csv(ROOT/'legacy_gate_runs.csv.gz',index=False)
report['status']='Post hoc diagnostic on the existing E8 primary seeds; original E7 scoring and allocations; independently reconstructed scores; 8000 decisions'
(ROOT/'legacy_gate_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
