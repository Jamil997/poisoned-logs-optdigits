import json,gzip
import numpy as np
import pandas as pd
from allocation import ROOT,allocations,exact_table,attack_counts,favorable_types

def run():
 logs=np.load(ROOT/'allocation_results/logs.npz');rows=[];w=[];n=20000;k=200;f=favorable_types().astype(int)
 for ga,ba in [(.05,.45),(.45,.05)]:
  B=2*np.array([ga,ba]);risks=allocations(B,.01)
  for name in ['uniform','influence_p1']:
   tab=exact_table(n,B,risks[name]);truth=[2*ga*.03,-2*ba*.001]
   for seed in range(80000,81000):
    counts=logs[f'20000_0.03_{seed}'];t=f@counts;sel,ss,ed,ta,target=attack_counts(counts,k,tab,0);u=0 if sel<0 else truth[sel]
    assert ed.sum()<=k and np.all(ed<=counts);np.testing.assert_array_equal(t+(1-2*f)@ed,ta)
    upbad=tab[1,t[1]+k];logood=tab[0,t[0]-k];lobad=tab[1,t[1]-k]
    expected=truth[1] if upbad>max(1e-9,logood) else 0. if max(logood,lobad)<=1e-9 else truth[0]
    assert abs(expected-u)<1e-12
    rows.append(dict(seed=seed,good_amplitude=ga,bad_amplitude=ba,method=name+'_nominal',selected=sel,improvement=u,harm=int(u<0),fallback=int(sel<0)))
    w.append(dict(row=len(rows)-1,edits=ed.tolist(),scores=ss.tolist()))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'nominal_control_runs.csv',index=False)
 with gzip.open(ROOT/'nominal_control_witnesses.json.gz','wt') as ff:json.dump(w,ff,separators=(',',':'))
 s=df.groupby(['good_amplitude','bad_amplitude','method'])[['improvement','harm','fallback']].mean();s.to_csv(ROOT/'nominal_control_summary.csv');print(s)
 (ROOT/'nominal_control_audit.json').write_text(json.dumps(dict(passed=True,decisions=len(rows),witnesses=len(w),exact_worst_utility_verified=True),indent=2)+'\n')
if __name__=='__main__':run()
