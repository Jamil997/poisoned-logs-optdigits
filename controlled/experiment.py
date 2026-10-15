"""Exact-count robustness benchmark with independently checkable optimal attacks."""
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist
import json,itertools,time,hashlib,argparse
ROOT=Path(__file__).resolve().parent
TYPES=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])))

def probabilities(beta,theta):
 s,z,a,r=TYPES.T;mean=.5+(2*a-1)*(beta*s-theta*z)
 return np.where(r,mean,1-mean)/8

def favorable_types():
 s,z,a,r=TYPES.T
 return np.stack([(s*(2*a-1)*(2*r-1)>0),(z*(2*a-1)*(2*r-1)>0)])

def lower_tables(n,B,alpha=.05):
 t=np.arange(n+1);prob=np.zeros(n+1);prob[1:]=beta_dist.ppf(alpha/2,t[1:],n-t[1:]+1)
 exact=B[:,None]*(prob-.5)
 conf=np.log(2/alpha);v=B*B/4;aa=B*conf/(3*n);rad=aa+np.sqrt(aa**2+2*v*conf/n)
 bern=B[:,None]*(t/n-.5)-rad[:,None]
 return {'binomial':exact,'bernstein':bern},rad

def score(table,t,k):
 t=np.asarray(t);return table[np.arange(2),np.maximum(0,t-k)]

def choose(s):return int(np.argmax(s)) if max(s)>1e-9 else -1

def attack_counts(counts,k,table,kd):
 fav=favorable_types();t=fav@counts
 # These row groups simultaneously attain the coordinatewise extremizers.
 opposite=np.flatnonzero(fav[0]&~fav[1]);same=np.flatnonzero(fav[0]&fav[1])
 assert counts[opposite].sum()>=k and counts[same].sum()>=k
 options=[]
 for target,eligible,shift in [('bad',opposite,np.array([-k,k])),('retain',same,np.array([-k,-k]))]:
  left=k;edits=np.zeros(16,dtype=int)
  for i in eligible:edits[i]=min(left,int(counts[i]));left-=edits[i]
  assert left==0
  tp=t+shift;s=score(table,tp,kd);sel=choose(s)
  options.append((sel,s,edits,tp,target))
 options.append((choose(score(table,t,kd)),score(table,t,kd),np.zeros(16,dtype=int),t,'clean'))
 # Outcomes ordered bad < retain < good in the specified environment.
 rank={1:0,-1:1,0:2};return min(options,key=lambda v:rank[v[0]])

def run():
 import pandas as pd
 cfg=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=False);(out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
 fav=favorable_types();rows=[];witnesses=[];logs={};start=time.perf_counter()
 for n in cfg['n']:
  tables={amp:lower_tables(n,np.array([.1,2*amp]),cfg['alpha'])[0] for amp in cfg['bad_amplitude']}
  for beta in cfg['beta']:
   probs=probabilities(beta,cfg['theta'])
   for seed in range(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count']):
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),177]))
    counts=rng.multinomial(n,probs);logs[f'{n}_{beta}_{seed}']=counts;t=fav@counts
    for eps in cfg['epsilon']:
     k=int(eps*n);rrng=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),k,178]));random_edits=rrng.multivariate_hypergeometric(counts,k)
     random_t=t+(1-2*fav.astype(int))@random_edits
     for amp in cfg['bad_amplitude']:
      truth=np.array([.1*beta,-2*amp*cfg['theta']])
      for family,table in tables[amp].items():
       for protected in [False,True]:
        method=family+('_corrected' if protected else '_nominal');kd=k if protected else 0
        sel,ss,edits,ta,target=attack_counts(counts,k,table,kd)
        for condition,tv,ed in [('clean',t,np.zeros(16,dtype=int)),('random',random_t,random_edits),('adaptive',ta,edits)]:
         scores=score(table,tv,kd);selected=choose(scores);u=0. if selected<0 else float(truth[selected])
         rows.append(dict(n=n,beta=beta,bad_amplitude=amp,epsilon=eps,seed=seed,method=method,condition=condition,selected=selected,improvement=u,harm=int(u<0),fallback=int(selected<0),edits=int(ed.sum())))
         if condition=='adaptive':witnesses.append(dict(row=len(rows)-1,edits=ed.tolist(),scores=scores.tolist(),favorable=tv.tolist(),target=target,exact_worst_utility=u))
      for condition in cfg['conditions']:rows.append(dict(n=n,beta=beta,bad_amplitude=amp,epsilon=eps,seed=seed,method='retain',condition=condition,selected=-1,improvement=0.,harm=0,fallback=1,edits=0))
   print('completed',n,beta,'seconds',round(time.perf_counter()-start,2),flush=True)
 import gzip
 pd.DataFrame(rows).to_csv(out/'runs.csv.gz',index=False)
 with gzip.open(out/'witnesses.json.gz','wt') as f:json.dump(witnesses,f,separators=(',',':'))
 np.savez_compressed(out/'logs.npz',**logs)
 meta=dict(seconds=time.perf_counter()-start,outcomes=len(rows),logs=len(logs),witnesses=len(witnesses),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest())
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':run()
