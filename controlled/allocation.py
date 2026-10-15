from pathlib import Path
import json,time,gzip,hashlib
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist,binom
from experiment import ROOT,probabilities,favorable_types,attack_counts,score,choose

def allocations(B,alpha):
 p1=alpha/B/(1/B).sum();p2=alpha/B**2/(1/B**2).sum()
 return dict(uniform=np.full(2,alpha/2),influence_p1=p1,influence_p2=p2,uniform_matched_p1=np.full(2,p1[1]))

def exact_table(n,B,risks):
 t=np.arange(n+1);table=np.zeros((2,n+1))
 for j in range(2):
  q=np.zeros(n+1);q[1:]=beta_dist.ppf(risks[j],t[1:],n-t[1:]+1);table[j]=B[j]*(q-.5)
 return table

def run():
 cfg=json.loads((ROOT/'allocation_protocol.json').read_text());out=ROOT/'allocation_results';out.mkdir(exist_ok=False);(out/'frozen_protocol.json').write_bytes((ROOT/'allocation_protocol.json').read_bytes())
 rows=[];w=[];logs={};riskrows=[];start=time.perf_counter();fav=favorable_types()
 for n in cfg['n']:
  tables={}
  for ga,ba in cfg['amplitudes']:
   B=2*np.array([ga,ba]);d=allocations(B,cfg['alpha']);tables[(ga,ba)]={name:exact_table(n,B,risk) for name,risk in d.items()}
   for name,table in tables[(ga,ba)].items():
    threshold=int(np.flatnonzero(table[1]>1e-9)[0]);risks=d[name];bound=float(binom.sf(threshold-1,n,.5-cfg['theta']))
    riskrows.append(dict(n=n,good_amplitude=ga,bad_amplitude=ba,method=name,alpha_good=risks[0],alpha_bad=risks[1],total_alpha=risks.sum(),threshold=threshold,exact_harm_upper=bound))
  for beta in cfg['beta']:
   for seed in range(80000,81000):
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),179]));counts=rng.multinomial(n,probabilities(beta,cfg['theta']));logs[f'{n}_{beta}_{seed}']=counts;t=fav@counts
    for eps in cfg['epsilon']:
     k=int(n*eps);rng_r=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),k,180]));er=rng_r.multivariate_hypergeometric(counts,k);tr=t+(1-2*fav.astype(int))@er
     for (ga,ba),ts in tables.items():
      truth=[2*ga*beta,-2*ba*cfg['theta']]
      for method,table in ts.items():
       sel,ss,edits,ta,target=attack_counts(counts,k,table,k)
       for condition,tv,ed in [('clean',t,np.zeros(16,dtype=int)),('random',tr,er),('adaptive',ta,edits)]:
        scores=score(table,tv,k);selected=choose(scores);u=0. if selected<0 else truth[selected]
        rows.append(dict(n=n,beta=beta,good_amplitude=ga,bad_amplitude=ba,epsilon=eps,seed=seed,method=method,condition=condition,selected=selected,improvement=float(u),harm=int(u<0),fallback=int(selected<0),edits=int(ed.sum())))
        if condition=='adaptive':w.append(dict(row=len(rows)-1,edits=ed.tolist(),scores=scores.tolist(),favorable=tv.tolist(),target=target))
      for condition in ['clean','random','adaptive']:rows.append(dict(n=n,beta=beta,good_amplitude=ga,bad_amplitude=ba,epsilon=eps,seed=seed,method='retain',condition=condition,selected=-1,improvement=0.,harm=0,fallback=1,edits=0))
   print('completed',n,beta,round(time.perf_counter()-start,2),flush=True)
 pd.DataFrame(rows).to_csv(out/'runs.csv.gz',index=False);pd.DataFrame(riskrows).to_csv(ROOT/'allocation_risk_bounds.csv',index=False)
 np.savez_compressed(out/'logs.npz',**logs)
 with gzip.open(out/'witnesses.json.gz','wt') as f:json.dump(w,f,separators=(',',':'))
 meta=dict(seconds=time.perf_counter()-start,outcomes=len(rows),logs=len(logs),witnesses=len(w),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),experiment_sha256=hashlib.sha256((ROOT/'experiment.py').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'allocation_protocol.json').read_bytes()).hexdigest())
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':run()
