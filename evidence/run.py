"""E8 runner; population quality is confined to simulation, attack, evaluation."""
from pathlib import Path
import sys,json,hashlib,time,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'controlled'))
from experiment import probabilities,favorable_types
from method import design,weights_for,margins,select,damage,tail
FAV=favorable_types().astype(int)

def sample_stream(sizes,beta,theta,seeds,stream):
 sizes=sorted(set(sizes));out={n:np.zeros((len(seeds),16),dtype=np.int64) for n in sizes}
 for i,seed in enumerate(seeds):
  rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(beta*10000),202608,stream]));count=np.zeros(16,dtype=np.int64);last=0
  for n in sizes:
   if n>last:count=count+rng.multinomial(n-last,probabilities(beta,theta))
   out[n][i]=count;last=n
 return out

def extreme_choices(U,V,B,rule,k,cap=0):
 a,c,d,h,corr=rule;_,ku,kt=damage(a,c,k,cap);u=U@FAV.T;v=V@FAV.T;scores=[];sels=[]
 for shift in [np.array([-1,1]),np.array([-1,-1]),np.array([0,0])]:
  s=margins((u+ku*shift).T,(v+kt*shift).T,B,rule);scores.append(s);sels.append(select(s))
 sels=np.array(sels);ranks=np.where(sels==1,0,np.where(sels<0,1,2));option=np.argmin(ranks,axis=0)
 return sels[option,np.arange(len(U))],option,np.stack(scores)[option,:,np.arange(len(U))].T,ku,kt

def frame(sel,scores,seeds,beta,g,b,m,eps,method,condition,option,ku,kt):
 truth=np.array([2*g*beta,-2*b*.001]);utility=np.where(sel<0,0,truth[np.maximum(0,sel)])
 return pd.DataFrame(dict(seed=seeds,beta=beta,good_amplitude=g,bad_amplitude=b,trusted_size=m,epsilon=eps,method=method,condition=condition,selected=sel,improvement=utility,harm=(utility<0).astype(int),fallback=(sel<0).astype(int),attack_option=option,main_edits=ku,trusted_edits=kt,score0=scores[0],score1=scores[1]))

def run():
 cfg=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=False)
 (out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes());started=time.perf_counter();n=cfg['total_observations'];alpha=cfg['alpha']
 seeds=np.arange(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count'])
 cost_sizes={(m,rho):int((n-m+rho*m)//rho) for m in cfg['trusted_sizes'] for rho in [5,10,50]}
 trusted_sizes=set(cfg['trusted_sizes'])|set(cost_sizes.values())|{n};main_sizes={n-m for m in cfg['trusted_sizes']}
 designs={};design_rows=[]
 for m in cfg['trusted_sizes']:
  for eps in cfg['epsilon']:
   best,grid=design(n-m,m,int(n*eps),alpha);designs[f'{m}_{eps}']=best
   for r in grid:design_rows.append(dict(trusted_size=m,epsilon=eps,selected=(r==best),**r))
 pd.DataFrame(design_rows).to_csv(out/'design_grid.csv',index=False);(out/'designs.json').write_text(json.dumps(designs,indent=2)+'\n')
 hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['protocol.json','method.py','run.py']}
 logs={};riskrows=[];costrows=[];stressrows=[];rows_written=0;min_slack=10**9
 with gzip.open(out/'runs.csv.gz','wt') as output:
  for beta in cfg['beta']:
   mains=sample_stream(main_sizes,beta,cfg['theta'],seeds,1);trusted=sample_stream(trusted_sizes,beta,cfg['theta'],seeds,2)
   logs.update({f'main_{beta}_{size}':v for size,v in mains.items()});logs.update({f'trusted_{beta}_{size}':v for size,v in trusted.items()})
   for m in cfg['trusted_sizes']:
    U=mains[n-m];V=trusted[m];ut=(U@FAV.T).T;vt=(V@FAV.T).T
    for eps in cfg['epsilon']:
     k=int(n*eps)
     for mask in [(FAV[0]==1)&(FAV[1]==0),(FAV[0]==1)&(FAV[1]==1)]:
      slack=U[:,mask].sum(axis=1)-k;assert slack.min()>=0;min_slack=min(min_slack,int(slack.min()))
     random_t=[]
     for i,seed in enumerate(seeds):
      rng=np.random.default_rng(np.random.SeedSequence([int(seed),m,k,int(beta*10000),93]));edits=rng.multivariate_hypergeometric(U[i],k);random_t.append(ut[:,i]+(1-2*FAV)@edits)
     rt=np.asarray(random_t).T
     for g,b in cfg['amplitudes']:
      B=2*np.array([g,b])
      for method in cfg['methods']:
       if method=='retain':
        ss=np.zeros((2,len(seeds)));selected=np.full(len(seeds),-1)
        for cond in cfg['conditions']:
         df=frame(selected,ss,seeds,beta,g,b,m,eps,method,cond,-1,0,0);df.to_csv(output,index=False,header=rows_written==0);rows_written+=len(df)
        continue
       rule=weights_for(method,n-m,m,k,alpha,B,vt);sel,opt,ss,ku,kt=extreme_choices(U,V,B,rule,k)
       for cond,tv in [('clean',ut),('random',rt)]:
        sc=margins(tv,vt,B,rule);df=frame(select(sc),sc,seeds,beta,g,b,m,eps,method,cond,-1,k if cond=='random' else 0,0);df.to_csv(output,index=False,header=rows_written==0);rows_written+=len(df)
       df=frame(sel,ss,seeds,beta,g,b,m,eps,method,'adaptive',opt,np.where(opt==2,0,ku),np.where(opt==2,0,kt));df.to_csv(output,index=False,header=rows_written==0);rows_written+=len(df)
       a,c,d,h,corr=rule;shift=damage(a,c,k)[0]
       if method.startswith('pilot_'):risk_bound=None
       elif method=='trusted_only' and m==0:risk_bound=0.
       else:risk_bound=tail(int(h[1,0])+corr-shift,n-m,m,a,c,.5-cfg['theta'])
       if beta==cfg['beta'][0]:riskrows.append(dict(trusted_size=m,epsilon=eps,good_amplitude=g,bad_amplitude=b,method=method,exact_harm_upper=risk_bound,a=a,c=c,denominator=d,threshold_bad=int(h[1,0]),correction=corr))
       if beta==.03 and eps==.01 and m>0 and method in ['pooled_equal','trusted_only','evidence_weighted']:
        cap=int(.01*m)
        for aware in [False,True]:
         srule=weights_for(method,n-m,m,k,alpha,B,vt,cap if aware else 0);sa,sc,_,_,_=srule;_,sku,skt=damage(sa,sc,k,cap)
         for cnt,amount in [(U,sku),(V,skt)]:
          for mask in [(FAV[0]==1)&(FAV[1]==0),(FAV[0]==1)&(FAV[1]==1)]:assert np.all(cnt[:,mask].sum(axis=1)>=amount)
         se,op,ss,ku,kt=extreme_choices(U,V,B,srule,k,cap)
         sdf=frame(se,ss,seeds,beta,g,b,m,eps,method,'trust_cap_declared' if aware else 'trust_breached',op,np.where(op==2,0,ku),np.where(op==2,0,kt));sdf['allowed_trusted_edits']=cap;stressrows.append(sdf)
    for g,b in cfg['amplitudes']:
     B=2*np.array([g,b])
     for rho in [5,10,50]:
      nt=cost_sizes[m,rho];tt=(trusted[nt]@FAV.T).T;rule=weights_for('trusted_only',0,nt,0,alpha,B,tt);sc=margins(np.zeros_like(tt),tt,B,rule)
      df=frame(select(sc),sc,seeds,beta,g,b,m,0,'cost_matched_trusted','clean',-1,0,0);df['rho']=rho;df['actual_trusted_size']=nt;df['acquisition_cost']=n-m+rho*m;costrows.append(df)
   for g,b in cfg['amplitudes']:
    B=2*np.array([g,b]);tt=(trusted[n]@FAV.T).T;rule=weights_for('trusted_only',0,n,0,alpha,B,tt);sc=margins(np.zeros_like(tt),tt,B,rule)
    df=frame(select(sc),sc,seeds,beta,g,b,n,0,'all_trusted','clean',-1,0,0);df['rho']=0;df['actual_trusted_size']=n;df['acquisition_cost']=0;costrows.append(df)
   print('completed',beta,'decisions',rows_written,'seconds',round(time.perf_counter()-started,2),flush=True)
 np.savez_compressed(out/'logs.npz',**logs);pd.DataFrame(riskrows).to_csv(out/'risk_bounds.csv',index=False)
 pd.concat(costrows,ignore_index=True).to_csv(out/'cost_runs.csv.gz',index=False);pd.concat(stressrows,ignore_index=True).to_csv(out/'stress_runs.csv.gz',index=False)
 meta=dict(seconds=time.perf_counter()-started,source_hashes=hashes,decisions=rows_written,cost_decisions=sum(len(d) for d in costrows),stress_decisions=sum(len(d) for d in stressrows),independent_seed_clusters=len(seeds),source_log_arrays=len(logs),min_main_attack_group_slack=min_slack)
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':run()
