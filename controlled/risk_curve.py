from pathlib import Path
import json,time,gzip,hashlib
import numpy as np
import pandas as pd
from scipy.stats import binom
from experiment import ROOT,probabilities,favorable_types,lower_tables,attack_counts,score,choose

def run():
 cfg=json.loads((ROOT/'risk_protocol.json').read_text());out=ROOT/'risk_results';out.mkdir(exist_ok=False);(out/'frozen_protocol.json').write_bytes((ROOT/'risk_protocol.json').read_bytes())
 n=cfg['n'];B=np.array([.1,.9]);truth=np.array([.003,-.0009]);fav=favorable_types();rows=[];w=[];logs={};risks=[];start=time.perf_counter()
 tables={alpha:lower_tables(n,B,alpha)[0] for alpha in cfg['alpha']}
 for alpha,ts in tables.items():
  for family,table in ts.items():
   threshold=int(np.flatnonzero(table[1]>1e-9)[0]);bound=float(binom.sf(threshold-1,n,.5+truth[1]/B[1]));risks.append(dict(alpha=alpha,family=family,bad_positive_count_threshold=threshold,exact_harm_upper=bound,nominal_theory_upper=alpha/2))
 for seed in range(70000,71000):
  rng=np.random.default_rng(np.random.SeedSequence([seed,n,177]));counts=rng.multinomial(n,probabilities(.03,.001));logs[str(seed)]=counts;t=fav@counts
  for eps in cfg['epsilon']:
   k=int(n*eps)
   for alpha,ts in tables.items():
    for family,table in ts.items():
     for protect in [False,True]:
      kd=k if protect else 0;sel,ss,edits,ta,target=attack_counts(counts,k,table,kd);u=0. if sel<0 else truth[sel]
      rows.append(dict(seed=seed,n=n,epsilon=eps,alpha=alpha,method=family+('_corrected' if protect else '_nominal'),selected=sel,improvement=float(u),harm=int(u<0),fallback=int(sel<0),clean_improvement=float(0 if (sc:=choose(score(table,t,kd)))<0 else truth[sc])))
      w.append(dict(row=len(rows)-1,edits=edits.tolist(),scores=ss.tolist(),favorable=ta.tolist(),target=target))
 pd.DataFrame(rows).to_csv(out/'runs.csv.gz',index=False);pd.DataFrame(risks).to_csv(ROOT/'exact_risk_bounds.csv',index=False)
 np.savez_compressed(out/'logs.npz',**logs)
 with gzip.open(out/'witnesses.json.gz','wt') as f:json.dump(w,f,separators=(',',':'))
 (out/'metadata.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,outcomes=len(rows),logs=len(logs),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),experiment_sha256=hashlib.sha256((ROOT/'experiment.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'risk_protocol.json').read_bytes()).hexdigest()),indent=2)+'\n')
 print(pd.DataFrame(rows).query('epsilon == .01').groupby(['alpha','method'])[['improvement','harm','fallback']].mean().to_string())
if __name__=='__main__':run()
