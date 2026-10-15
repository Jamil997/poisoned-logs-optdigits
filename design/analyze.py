"""Independent reconstruction, paired inference, and cost accounting for E4."""
from pathlib import Path
import sys,json,gzip,time,hashlib,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import beta
from confirmation import ROOT,PARENT,all_designs,core

def tsum(z,k):
 return np.sort(z,axis=1)[:,-min(k,z.shape[1]):].sum(axis=1) if k else np.zeros(len(z))

def scores(c,off,r,k):
 return c@r/len(r)+off-tsum(c*r-np.minimum(c,0),k)/len(r)

def audit_shard(path):
 path=Path(path);meta=json.loads((path/'metadata.json').read_text());df=pd.read_csv(path/'runs.csv')
 with gzip.open(path/'witnesses.json.gz','rt') as f:ws={w['row']:w for w in json.load(f)}
 cfg=json.loads((ROOT/'confirmation_protocol.json').read_text())
 assert meta['protocol_sha256']==core.sha(ROOT/'confirmation_protocol.json')
 assert meta['source_sha256']==core.sha(ROOT/'confirmation.py')==cfg['frozen_sources']['confirmation.py']
 assert meta['core_sha256']==core.sha(core.__file__)
 assert meta['joint_source_sha256']==core.sha(ROOT/'joint_develop.py')==cfg['frozen_sources']['joint_develop.py']
 assert meta['models_sha256']==core.sha(PARENT/'frozen_models/models.npz')==cfg['models_sha256']
 data=dict(np.load(PARENT/'frozen_models/models.npz'));p=data['p'];y=data['y'];N=len(y)
 name,n,start,count,mode=meta['task'];Q=all_designs(p)[name]
 libs={z['name']:z['candidate_indices'] for z in json.loads((PARENT/'subset_protocol.json').read_text())['libraries']}
 tr=(p[1:,np.arange(N),y]-p[0,np.arange(N),y]).mean(axis=1)
 assert len(df)==count*6*2*9 and len(ws)==count*6*2*6
 errors=[];duals=0;env_cache={};score_cache={};checks=0
 for seed,group in df.groupby('seed',sort=False):
  rng=np.random.default_rng(np.random.SeedSequence([int(seed),n,71]));pairs=np.searchsorted(np.cumsum(Q.ravel()),rng.random(n),side='right');x,a=np.divmod(pairs,10);r=(a==y[x]).astype(float)
  lh=hashlib.sha256(x.astype('<i4').tobytes()+a.astype('uint8').tobytes()+r.astype('uint8').tobytes()).hexdigest()
  for lib,part in group.groupby('library',sort=False):
   inds=libs[lib];truth=tr[inds];d=(p[np.array(inds)+1]-p[0])/N;C=d/Q
   B=abs(C).max(axis=(1,2));V=(Q*C**2).sum(axis=(1,2))/4;t=np.log(3/.05)
   aa=B*t/(3*n);radius=aa+np.sqrt(aa**2+2*V*t/n)
   c=C[:,x,a];off=-c.sum(axis=1)/(2*n)-radius;base=c@r/n+off
   for index,row in part.iterrows():
    assert row.log_sha256==lh
    if row.method=='retain':
     assert row.selected==-1 and row.improvement==0 and row.fallback==1 and row.harm==0;checks+=1;continue
    w=ws[index];ids=np.array(w['indices'],dtype=int);ka=int(n*row.epsilon);kd=ka if row.method=='corrected' else 0
    assert row.declared_cap==kd and len(ids)==row.edits and len(ids)==len(set(ids)) and len(ids)<=ka
    assert len(ids)==0 or (min(ids)>=0 and max(ids)<n)
    if row.condition=='clean':assert not len(ids)
    if row.condition=='random':
     rngr=np.random.default_rng(np.random.SeedSequence([int(seed),n,ka,919]));np.testing.assert_array_equal(ids,np.sort(rngr.choice(n,ka,replace=False)))
    rp=r.copy();rp[ids]=1-rp[ids];ss=scores(c,off,rp,kd)
    err=float(np.max(abs(ss-w['scores'])));errors.append(err);assert err<1e-11
    sel=int(np.argmax(ss)) if max(ss)>1e-9 else -1;assert sel==row.selected
    u=0 if sel<0 else truth[sel];assert abs(u-row.improvement)<1e-12 and row.harm==int(u<0) and row.fallback==int(sel<0)
    key=(seed,lib,ka,kd)
    if key not in env_cache:
     lo=base-tsum(c*r-np.minimum(c,0),ka+kd)/n
     hi=base-tsum(c*r-np.minimum(c,0),kd-ka)/n if kd>=ka else base+tsum(np.maximum(c,0)-c*r,ka-kd)/n
     possible=([-1] if max(lo)<=1.1e-9 else [])+[j for j in range(3) if hi[j]>.9e-9 and hi[j]>=max(1e-9,max(np.delete(lo,j)))-1e-10]
     env_cache[key]=(lo,hi,possible)
    lo,hi,poss=env_cache[key];poss=poss.copy()
    np.testing.assert_allclose(lo,w['lower'],atol=1e-11,rtol=0);np.testing.assert_allclose(hi,w['upper'],atol=1e-11,rtol=0)
    assert np.all(ss>=lo-1e-9) and np.all(ss<=hi+1e-9)
    # Adaptive witness holds the LP certificate used on all three conditions.
    adaptive_index=part[(part.epsilon==row.epsilon)&(part.method==row.method)&(part.condition=='adaptive')].index[0]
    for sub in ws[adaptive_index]['subproblems']:
     if sub.get('kind')!='all_real_L1_dual' or sub.get('dual') is None:continue
     assert kd==0
     target=sub['target'];g=c*(1-2*r)/n;tb=0 if target<0 else base[target];tg=np.zeros(n) if target<0 else g[target]
     Az=[np.ones(n)];At=[0.];rhs=[ka]
     if target>=0:Az.append(-tg);At.append(1.);rhs.append(tb)
     for j in range(3):
      if j!=target:Az.append(g[j]-tg);At.append(1.);rhs.append(tb-base[j])
     lam=np.array(sub['dual']);assert min(lam)>=0 and abs(lam@At-1)<1e-10
     ub=float(lam@rhs+np.maximum(0,-lam@np.array(Az)).sum());assert abs(ub-sub['upper_margin'])<1e-10
     if ub<-1e-8 and target in poss:poss.remove(target)
     if row.condition=='adaptive':duals+=1
    assert sorted(poss)==sorted(w['possible']) and sel in poss
    floor=min(0 if j<0 else truth[j] for j in poss);assert abs(floor-row.utility_floor)<1e-12
    assert row.harm_excluded==int(all(j<0 or truth[j]>=0 for j in poss))
    if row.condition=='adaptive':assert abs(max(0,u-floor)-row.attack_utility_gap)<1e-12
    checks+=1
 return dict(shard=path.name,decisions=checks,witnesses=len(ws),duals=duals,max_error=max(errors),passed=True)

def audit():
 start=time.perf_counter();paths=sorted((ROOT/'results_main').glob('*/runs.csv'));out=[]
 assert len(paths)==128
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(audit_shard,p.parent) for p in paths]
  for f in as_completed(jobs):
   out.append(f.result())
   if len(out)%16==0:print('audit',len(out),'/128',round(time.perf_counter()-start,1),flush=True)
 report=dict(passed=True,decisions=sum(z['decisions'] for z in out),witnesses=sum(z['witnesses'] for z in out),duals=sum(z['duals'] for z in out),max_score_error=max(z['max_error'] for z in out),seconds=time.perf_counter()-start,shards=out)
 core.write_json(ROOT/'audit.json',report);return report

def summarize():
 paths=sorted((ROOT/'results_main').glob('*/runs.csv'));df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
 assert len(df)==345600
 keys=['design','n','seed','library','epsilon','method','condition'];assert not df.duplicated(keys).any()
 cfg=json.loads((ROOT/'confirmation_protocol.json').read_text());expected=set(itertools.product(cfg['designs'],cfg['n'],range(40000,40200),['mmp','mpm','mpp','pmm','pmp','ppm'],cfg['epsilon'],['nominal','corrected','retain'],['clean','random','adaptive']))
 assert set(df[keys].itertuples(index=False,name=None))==expected
 df.to_csv(ROOT/'all_runs.csv.gz',index=False)
 cells=[]
 for key,p in df.groupby(['design','n','library','epsilon','method','condition']):
  h=int(p.harm.sum());N=len(p)
  cells.append(dict(zip(['design','n','library','epsilon','method','condition'],key),seeds=N,harm=h/N,harm_count=h,harm_upper95=float(beta.ppf(.95,h+1,N-h)) if h<N else 1.,utility_pp=100*p.improvement.mean(),floor_pp=100*p.utility_floor.mean(),fallback=p.fallback.mean(),unresolved=int((p.attack_utility_gap>1e-12).sum()),harm_possible=int((p.harm_excluded==0).sum())))
 pd.DataFrame(cells).to_csv(ROOT/'cells.csv',index=False)
 rng=np.random.default_rng(87421);ix=rng.integers(200,size=(10000,200));rows=[]
 for key,p in df.groupby(['design','n','epsilon','method','condition']):
  d=dict(zip(['design','n','epsilon','method','condition'],key))
  for col,label,scale in [('improvement','utility',100),('utility_floor','floor',100),('harm','harm',1),('fallback','fallback',1)]:
   means=p.groupby('seed')[col].mean().sort_index().to_numpy();boot=means[ix].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);d[label]=float(means.mean()*scale);d[label+'_low']=float(lo*scale);d[label+'_high']=float(hi*scale)
  d.update(unresolved=int((p.attack_utility_gap>1e-12).sum()),harm_possible=int((p.harm_excluded==0).sum()),outcomes=len(p))
  rows.append(d)
 summary=pd.DataFrame(rows);summary.to_csv(ROOT/'summary.csv',index=False)
 data=dict(np.load(PARENT/'frozen_models/models.npz'));p=data['p'];y=data['y'];cost=[]
 for name,Q in all_designs(p).items():
  rho=Q.sum(axis=1);inc=p[0,np.arange(len(y)),y];logged=Q[np.arange(len(y)),y].sum()
  cost.append(dict(design=name,query_reward=float(logged),incumbent_reward_same_contexts=float(rho@inc),incumbent_relative_query_cost=float(rho@inc-logged),context_TV_from_uniform=float(.5*abs(rho-1/len(y)).sum()),rejection_envelope=float(max(rho)*len(y)),expected_arrivals_per_accepted_query=float(max(rho)*len(y)),max_influence=float(np.max(abs((p[1:]-p[0])/len(y)/Q)))))
 pd.DataFrame(cost).to_csv(ROOT/'collection_costs.csv',index=False)
 info=dict(decisions=len(df),independent_seed_clusters=200,distinct_logs=len(df[['design','n','seed']].drop_duplicates()),note='Logs across designs and n use common random numbers; do not count 3200 as independent replicates for aggregate inference. 200 independent seed clusters.',primary=summary[(summary.design=='joint_1')&(summary.n==20000)&(summary.epsilon==.005)&(summary.method=='corrected')&(summary.condition=='adaptive')].iloc[0].to_dict())
 core.write_json(ROOT/'inference.json',info)
 print(summary[(summary.n==20000)&(summary.condition=='adaptive')][['design','epsilon','method','utility','floor','harm','fallback','unresolved','harm_possible']].to_string(index=False))
 return summary,pd.DataFrame(cost)

def figures(s,c):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
 for n,color in [(5000,'#d1873b'),(20000,'#007f78')]:
  t=s[(s.method=='corrected')&(s.condition=='adaptive')&(s.n==n)&(s.epsilon==.005)&s.design.str.match(r'^joint_[0-9]')].copy();t['lam']=t.design.str[6:].astype(float);t=t.sort_values('lam')
  ax[0].plot(t.lam,t.floor,'o-',color=color,label=f'{n:,} queries: certified floor');ax[0].fill_between(t.lam,t.floor_low,t.floor_high,color=color,alpha=.2)
  ax[0].plot(t.lam,t.utility,'--',color=color,alpha=.65)
 ax[0].axhline(0,color='gray',lw=.8);ax[0].set(xlabel='Fraction assigned to joint disagreement sampling',ylabel='True improvement (percentage points)',title='0.5% reward corruption');ax[0].legend(fontsize=8)
 t=s[(s.method=='corrected')&(s.condition=='adaptive')&(s.n==20000)&(s.epsilon==.005)].set_index('design');names=['joint_0','action_only','uniform','joint_0.5','joint_1','joint_rms'];labels=['Original','Actions\nonly','Uniform','Joint\n50%','Joint\n100%','Joint\nRMS']
 ax[1].bar(labels,[t.loc[z,'floor'] for z in names],color=['#778899']*3+['#007f78']*3);ax[1].set(ylabel='Certified improvement floor (pp)',title='Same 20,000 queries; all six libraries');ax[1].tick_params(axis='x',labelsize=8)
 t=s[(s.method=='corrected')&(s.condition=='adaptive')&(s.n==20000)&s.design.str.match(r'^joint_[0-9]')].merge(c,on='design');t['lam']=t.design.str[6:].astype(float)
 for eps,color in [(.005,'#007f78'),(.01,'#aa4b37')]:
  z=t[t.epsilon==eps].sort_values('lam');ax[2].plot(z.incumbent_relative_query_cost*100,z.floor,'o-',color=color,label=f'{eps*100:g}% corruption')
  for r in z.itertuples():
   if eps==.005:ax[2].annotate(f'{r.lam:.0%}',(100*r.incumbent_relative_query_cost,r.floor),xytext=(3,5),textcoords='offset points',fontsize=8)
 ax[2].set(xlabel='Collection reward cost vs incumbent on\nthe same queried contexts (pp)',ylabel='Certified improvement floor (pp)',title='Benefit comes with acquisition cost');ax[2].legend(fontsize=8)
 fig.suptitle('OptDigits: prospective evidence design under reward poisoning',fontsize=14)
 fig.savefig(ROOT/'tradeoff.png',dpi=180);plt.close(fig)
if __name__=='__main__':
 if '--audit' in sys.argv:print(json.dumps(audit(),indent=2))
 else:s,c=summarize();figures(s,c)
