from pathlib import Path
import json,gzip,time,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
ROOT=Path(__file__).resolve().parent

def audit():
 start=time.perf_counter();out=ROOT/'allocation_results';meta=json.loads((out/'metadata.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert meta['source_sha256']==sha(ROOT/'allocation.py') and meta['experiment_sha256']==sha(ROOT/'experiment.py') and meta['protocol_sha256']==sha(ROOT/'allocation_protocol.json')
 d=pd.read_csv(out/'runs.csv.gz');logs=np.load(out/'logs.npz')
 with gzip.open(out/'witnesses.json.gz','rt') as f:ws={w['row']:w for w in json.load(f)}
 types=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])));s,z,a,r=types.T;fav=np.array([s*(2*a-1)*(2*r-1)>0,z*(2*a-1)*(2*r-1)>0],dtype=int)
 cache={};errors=[];checks=0;logcount=0;slack=10**9
 for (n,beta,seed),part in d.groupby(['n','beta','seed'],sort=False):
  counts=logs[f'{n}_{beta}_{seed}'];rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),179]));mean=.5+(2*a-1)*(beta*s-.001*z);np.testing.assert_array_equal(counts,rng.multinomial(n,np.where(r,mean,1-mean)/8));t=fav@counts;logcount+=1
  for row in part.itertuples():
   if row.method=='retain':assert row.selected==-1 and row.improvement==0 and row.harm==0;checks+=1;continue
   B=2*np.array([row.good_amplitude,row.bad_amplitude]);ck=(n,*B,row.method)
   if ck not in cache:
    if row.method=='uniform':risk=np.full(2,.005)
    elif row.method=='uniform_matched_p1':risk=np.full(2,.01/B[1]/(1/B).sum())
    else:p=1 if row.method=='influence_p1' else 2;risk=.01*B**(-p)/sum(B**(-p))
    if row.method!='uniform_matched_p1':assert abs(sum(risk)-.01)<1e-12
    tt=np.arange(n+1);q=np.zeros((2,n+1))
    for j in range(2):q[j,1:]=beta_dist.isf(1-risk[j],tt[1:],n-tt[1:]+1)
    cache[ck]=B[:,None]*(q-.5)
   table=cache[ck];k=int(n*row.epsilon)
   if row.condition=='adaptive':
    w=ws[row.Index];ed=np.array(w['edits']);assert min(ed)>=0 and np.all(ed<=counts) and ed.sum()<=k;tv=t+(1-2*fav)@ed;np.testing.assert_array_equal(tv,w['favorable'])
   elif row.condition=='clean':tv=t
   else:
    rngr=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),k,180]));ed=rngr.multivariate_hypergeometric(counts,k);tv=t+(1-2*fav)@ed
   scores=table[np.arange(2),np.maximum(0,tv-k)];sel=int(scores.argmax()) if max(scores)>1e-9 else -1;truth=[2*row.good_amplitude*beta,-2*row.bad_amplitude*.001];u=0. if sel<0 else truth[sel]
   assert row.selected==sel and abs(row.improvement-u)<1e-12 and row.harm==int(u<0) and row.fallback==int(sel<0)
   if row.condition=='adaptive':
    err=float(max(abs(scores-w['scores'])));errors.append(err);assert err<1e-11
    groups=[counts[(fav[0]==1)&(fav[1]==0)].sum(),counts[(fav[0]==1)&(fav[1]==1)].sum()];assert min(groups)>=k;slack=min(slack,int(min(groups)-k))
    upperbad=table[1,t[1]];lowergood=table[0,max(0,t[0]-2*k)];lowerbad=table[1,max(0,t[1]-2*k)]
    worst=truth[1] if upperbad>max(1e-9,lowergood) else 0. if max(lowergood,lowerbad)<=1e-9 else truth[0]
    assert abs(u-worst)<1e-12
   checks+=1
 report=dict(passed=True,decisions=checks,logs=logcount,witnesses=len(ws),max_score_error=max(errors),exact_worst_utility_verified=True,minimum_group_slack=slack,seconds=time.perf_counter()-start)
 (ROOT/'allocation_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True);return d

def summarize(d):
 keys=['n','beta','good_amplitude','bad_amplitude','epsilon','method','condition'];rows=[]
 for key,p in d.groupby(keys):
  h=int(p.harm.sum());rows.append(dict(zip(keys,key),utility_pp=100*p.improvement.mean(),harm=h/len(p),harm_count=h,harm_upper95=float(beta_dist.ppf(.95,h+1,len(p)-h)) if h<len(p) else 1.,fallback=p.fallback.mean(),seeds=len(p)))
 s=pd.DataFrame(rows);s.to_csv(ROOT/'allocation_summary.csv',index=False)
 rng=np.random.default_rng(71318);ix=rng.integers(1000,size=(10000,1000));contrasts=[]
 for name,ga,ba in [('primary',.05,.45),('reversed',.45,.05)]:
  p=d[(d.n==20000)&(d.beta==.03)&(d.good_amplitude==ga)&(d.bad_amplitude==ba)&(d.epsilon==.01)&(d.condition=='adaptive')]
  pivot=p.pivot(index='seed',columns='method',values='improvement').sort_index()
  for base in ['uniform','uniform_matched_p1']:
   diff=100*(pivot.influence_p1-pivot[base]).to_numpy();boot=diff[ix].mean(axis=1);contrasts.append(dict(setting=name,contrast='influence_p1 minus '+base,difference_pp=float(diff.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975))))
 (ROOT/'allocation_contrasts.json').write_text(json.dumps(contrasts,indent=2)+'\n');print(contrasts,flush=True)
 return s

def figures(s):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 bounds=pd.read_csv(ROOT/'allocation_risk_bounds.csv');fig,ax=plt.subplots(1,3,figsize=(14.5,4.6),layout='constrained');colors={'uniform':'#627587','influence_p1':'#007f78','influence_p2':'#c18b36','uniform_matched_p1':'#888888'}
 for i,(ga,ba,label) in enumerate([(.05,.45,'Low-influence good policy'),(.45,.05,'Reversed influence: failure test')]):
  t=s[(s.n==20000)&(s.beta==.03)&(s.good_amplitude==ga)&(s.bad_amplitude==ba)&(s.condition=='adaptive')]
  for method in ['uniform','influence_p1','influence_p2']:
   z=t[t.method==method].sort_values('epsilon');ax[i].plot(z.epsilon*100,z.utility_pp,'o-',label=method.replace('_',' '),color=colors[method])
  ax[i].set(title=label,xlabel='Corruption budget (%)',ylabel='Mean true improvement (pp)');ax[i].axhline(0,color='gray',lw=.7);ax[i].legend(fontsize=8)
 t=s[(s.n==20000)&(s.beta==.03)&(s.good_amplitude==.05)&(s.bad_amplitude==.45)&(s.epsilon==.01)&(s.condition=='adaptive')].merge(bounds,on=['n','good_amplitude','bad_amplitude','method'])
 for row in t.itertuples():
  ax[2].scatter(100*row.exact_harm_upper,row.utility_pp,color=colors[row.method],s=50)
  labels={'uniform':'Equal allocation','influence_p1':'Proposed p=1','influence_p2':'p=2','uniform_matched_p1':'Equal allocation, matched risk'}
  offsets={'uniform':(4,5),'influence_p1':(4,-14),'influence_p2':(4,8),'uniform_matched_p1':(4,5)}
  ax[2].annotate(labels[row.method],(100*row.exact_harm_upper,row.utility_pp),xytext=offsets[row.method],textcoords='offset points',fontsize=8)
 ax[2].set(xlabel='Analytic harm probability upper bound (%)',ylabel='Mean true improvement (pp)',title='Primary: allocation changes the comparison');ax[2].margins(x=.35,y=.25)
 fig.suptitle('Influence-weighted allocation: useful in one regime, falsified as a universal rule',fontsize=13);fig.savefig(ROOT/'allocation_tradeoff.png',dpi=180);plt.close(fig)

if __name__=='__main__':d=audit();s=summarize(d);figures(s)
