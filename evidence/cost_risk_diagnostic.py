"""Post-E8 diagnostic: monetary-cost alternatives at the same harm-bound ceiling.

Uses evaluator p_bad=.499 and previously used E8 seeds; not fresh confirmation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit import critical,F,pick
from analyze import contrast,summary
ROOT=Path(__file__).resolve().parent
protocol=dict(stage='E8 post hoc cost-risk diagnostic',reason='Nominal-alpha cost curves can have different actual harmful-policy risk; apply the prespecified primary matched-risk calibration also to monetary-cost controls',gamma=.001,p_bad=.499,rho=[5,10,50],seeds='Same E8 seeds 90000..91999; no fresh confirmation',weights='No retuning of the proposed evidence weight')
(ROOT/'cost_risk_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
logs=np.load(ROOT/'results/logs.npz');rows=[];contrasts={}
for g,b in [[.05,.45],[.45,.05]]:
 base=pd.read_csv(ROOT/'matched_risk_runs.csv.gz').query('method=="evidence_weighted" and good_amplitude==@g')
 contrasts[str(g)]={}
 for rho in [5,10,50]:
  nt=int((19000+rho*1000)//rho);V=logs[f'trusted_0.03_{nt}'];t=(V@F.T).T
  h=critical(0,nt,0,1,.001,.499);scores=2*np.array([g,b])[:,None]*(t-h+.5)/nt;sel=pick(scores);truth=np.array([2*g*.03,-2*b*.001]);u=np.where(sel<0,0,truth[np.maximum(sel,0)])
  df=pd.DataFrame(dict(seed=np.arange(90000,92000),good_amplitude=g,bad_amplitude=b,rho=rho,trusted_count=nt,threshold=h,method='cost_matched_trusted',selected=sel,improvement=u,harm=(u<0).astype(int),fallback=(sel<0).astype(int)))
  rows.append(df);contrasts[str(g)][str(rho)]=contrast(base,df)
d=pd.concat(rows,ignore_index=True);d.to_csv(ROOT/'cost_risk_runs.csv.gz',index=False)
summary(d,['good_amplitude','rho','trusted_count','threshold']).to_csv(ROOT/'cost_risk_summary.csv',index=False)
(ROOT/'cost_risk_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n');print(summary(d,['good_amplitude','rho'])[['good_amplitude','rho','utility_pp','harm']].to_string(index=False))
