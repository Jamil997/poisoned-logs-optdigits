"""Independent verification via FFT convolution and reconstructed edit vectors.

Does not import method.py or its score, threshold, design or attack functions.
"""
from pathlib import Path
from functools import lru_cache
import hashlib,json,time,itertools
import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.signal import fftconvolve
ROOT=Path(__file__).resolve().parent
TYPES=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])))
s,z,a,r=TYPES.T
F=np.stack([s*(2*a-1)*(2*r-1)>0,z*(2*a-1)*(2*r-1)>0]).astype(int)

@lru_cache(48)
def survival(n,m,a,c,p=.5):
 def pmf(nn,ww):
  if ww==0:return np.ones(1)
  v=np.zeros(nn*ww+1);v[::ww]=binom.pmf(np.arange(nn+1),nn,p);return v
 v=fftconvolve(pmf(n,a),pmf(m,c));v=np.maximum(v,0)
 return np.r_[np.cumsum(v[::-1])[::-1],0.]

@lru_cache(None)
def critical(n,m,a,c,risk,p=.5):
 if risk==0:return a*n+c*m+1
 return int(np.flatnonzero(survival(n,m,a,c,p)+1e-12<=risk)[0])

def capacities_and_edits(counts,amount,opposite):
 mask=(F[0]==1)&(F[1]==(0 if opposite else 1));ed=np.zeros_like(counts);left=np.full(len(counts),amount)
 for ix in np.flatnonzero(mask):
  ed[:,ix]=np.minimum(left,counts[:,ix]);left-=ed[:,ix]
 assert np.all(left==0)
 assert np.all(ed>=0) and np.all(ed<=counts) and np.all(ed.sum(axis=1)==amount)
 shift=ed@(1-2*F).T
 np.testing.assert_array_equal(shift,np.tile([-amount,amount if opposite else -amount],(len(counts),1)))
 return ed

def independently_compute(method,n,m,k,B,vt,designs,cap=0):
 if method.startswith('evidence_'):
  # Designs for trust stress verified using an independently evaluated full grid.
  q=designs[(m,k,cap)];a,c,d=int(q.a),int(q.c),int(q.denominator)
 elif method=='trusted_only':a,c,d=0,1,max(m,1)
 elif method in ['main_only','pilot_select_main','pilot_weight_main']:a,c,d=1,0,n
 else:a,c,d=1,1,n+m
 risk=np.full((2,vt.shape[1]),.005)
 if method=='pooled_influence':risk[:]=(.01/B/(1/B).sum())[:,None]
 if m and method.startswith('pilot_'):
  diff=vt[0]-vt[1]
  if method=='pilot_select_main':
   risk[:]=0;risk[0,diff>=0]=.01;risk[1,diff<0]=.01
  else:
   risk[0,diff>0]=.009;risk[1,diff>0]=.001;risk[0,diff<0]=.001;risk[1,diff<0]=.009
 h=np.empty_like(risk,dtype=np.int64)
 for rr in np.unique(risk):h[risk==rr]=critical(n,m,a,c,float(rr))
 corr=a*k+max(0,c-a)*min(k,cap)
 if method=='evidence_nominal':corr=0
 return a,c,d,h,corr

def pick(score):
 return np.where(score.max(axis=0)>0,score.argmax(axis=0),-1)

def main():
 start=time.perf_counter();cfg=json.loads((ROOT/'protocol.json').read_text());meta=json.loads((ROOT/'results/metadata.json').read_text())
 for f,h in meta['source_hashes'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 assert (ROOT/'protocol.json').read_bytes()==(ROOT/'results/frozen_protocol.json').read_bytes()
 logs=np.load(ROOT/'results/logs.npz');N=cfg['total_observations'];seeds=np.arange(90000,92000)
 # Check every stored log against an independent regeneration of row probabilities.
 for beta in cfg['beta']:
  mean=.5+(2*a-1)*(beta*s-.001*z);p=np.where(r==1,mean,1-mean)/8
  for stream,name in [(1,'main'),(2,'trusted')]:
   sizes=sorted(int(x.split('_')[-1]) for x in logs.files if x.startswith(f'{name}_{beta}_'))
   for ix,seed in enumerate(seeds):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(beta*10000),202608,stream]));v=np.zeros(16,dtype=int);prev=0
    for size in sizes:
     if size>prev:v+=rng.multinomial(size-prev,p)
     assert np.array_equal(v,logs[f'{name}_{beta}_{size}'][ix]);prev=size
 # Verify all candidate weights and every selected design independently by FFT.
 grid=pd.read_csv(ROOT/'results/design_grid.csv');designs={};max_null_err=0.
 for (m,eps),sub in grid.groupby(['trusted_size','epsilon']):
  n=N-m;k=int(N*eps)
  for row in sub.itertuples():
   h=critical(n,m,row.a,row.c,.005);assert h==row.threshold
   obj=(h-.5+2*row.a*k)/row.denominator-.5
   assert abs(obj-row.objective)<1e-13
  q=sub.sort_values(['objective','fraction']).iloc[0];assert q.selected;designs[(m,k,0)]=q
  # Same grid for the separately declared trust-boundary stress.
  if m>0 and eps==.01:
   cap=int(.01*m);ss=sub.copy();ss['objective']=(ss.threshold-.5+2*(ss.a*k+np.maximum(ss.c-ss.a,0)*min(k,cap)))/ss.denominator-.5
   designs[(m,k,cap)]=ss.sort_values(['objective','fraction']).iloc[0]
 edits_archive={};checked=0;max_error=0.;adaptive=0;random_cache={};min_slack=10**9
 for fname in ['runs.csv.gz','stress_runs.csv.gz','cost_runs.csv.gz']:
  for chunk in pd.read_csv(ROOT/'results'/fname,chunksize=108000):
   keys=['beta','good_amplitude','bad_amplitude','trusted_size','epsilon','method','condition']
   if fname=='cost_runs.csv.gz':keys+=['rho','actual_trusted_size']
   for key,df in chunk.groupby(keys,sort=False):
    beta,g,b,m,eps,method,condition=key[:7];B=2*np.array([g,b]);ix=df.seed.to_numpy()-90000;nt=m;n=N-m;k=int(N*eps)
    if method=='all_trusted':nt=N;n=0
    if method=='cost_matched_trusted':nt=int(key[-1]);n=0
    V=logs[f'trusted_{beta}_{nt}'][ix];vt=(V@F.T).T
    U=np.zeros_like(V) if n==0 else logs[f'main_{beta}_{n}'][ix];ut=(U@F.T).T
    if method=='retain':sc=np.zeros_like(ut,dtype=float);sel=np.full(len(df),-1);op=np.full(len(df),-1)
    else:
     declcap=int(.01*m) if condition=='trust_cap_declared' else 0
     meth='trusted_only' if method in ['all_trusted','cost_matched_trusted'] else method
     a0,c0,d,h,corr=independently_compute(meth,n,nt,k,B,vt,designs,declcap)
     def scores(u,v):return B[:,None]*(a0*u+c0*v-h-corr+.5)/d
     if condition=='random':
      rr=[]
      for i,seed in enumerate(df.seed):
       rng=np.random.default_rng(np.random.SeedSequence([int(seed),m,k,int(beta*10000),93]));e=rng.multivariate_hypergeometric(U[i],k)
       assert e.sum()==k and np.all(e<=U[i]);rr.append(ut[:,i]+(1-2*F)@e)
      sc=scores(np.array(rr).T,vt);sel=pick(sc);op=np.full(len(df),-1)
     elif condition in ['adaptive','trust_breached','trust_cap_declared']:
      actualcap=int(.01*m) if condition.startswith('trust_') else 0
      kt=min(k,actualcap) if c0>a0 else 0;ku=k-kt
      candidates=[];cs=[]
      for opt in range(3):
       if opt==2:eu=np.zeros_like(U);ev=np.zeros_like(V)
       else:
        eu=capacities_and_edits(U,ku,opt==0);ev=capacities_and_edits(V,kt,opt==0)
       sc0=scores(ut+(eu@(1-2*F).T).T,vt+(ev@(1-2*F).T).T);cs.append(sc0);candidates.append(pick(sc0))
       if opt<2:
        assert np.all((eu.sum(axis=1)+ev.sum(axis=1))<=k)
       if opt<2 and len(df)==2000:
        edits_archive[f'{beta}_{m}_{k}_{ku}_{kt}_{opt}']=np.concatenate([eu,ev],axis=1).astype(np.int16)
      cand=np.array(candidates);rank=np.where(cand==1,0,np.where(cand<0,1,2));op=rank.argmin(axis=0)
      sc=np.stack(cs)[op,:,np.arange(len(df))].T;sel=cand[op,np.arange(len(df))]
      assert np.array_equal(df.main_edits.to_numpy(),np.where(op==2,0,ku));assert np.array_equal(df.trusted_edits.to_numpy(),np.where(op==2,0,kt));adaptive+=len(df)
     else:sc=scores(ut,vt);sel=pick(sc);op=np.full(len(df),-1)
    assert np.array_equal(sel,df.selected.to_numpy())
    assert np.array_equal(op,df.attack_option.to_numpy())
    err=np.max(np.abs(sc-df[['score0','score1']].to_numpy().T));max_error=max(max_error,float(err));assert err<1e-12
    truth=np.array([2*g*beta,-2*b*.001]);u=np.where(sel<0,0,truth[np.maximum(sel,0)])
    assert np.max(abs(u-df.improvement.to_numpy()))<1e-12
    assert np.array_equal(df.harm.to_numpy(),u<0);assert np.array_equal(df.fallback.to_numpy(),sel<0);checked+=len(df)
  print('audited',fname,checked,'elapsed',round(time.perf_counter()-start,2),flush=True)
 assert checked==meta['decisions']+meta['cost_decisions']+meta['stress_decisions']
 np.savez_compressed(ROOT/'results/attack_certificates.npz',**edits_archive)
 report=dict(passed=True,decisions_reconstructed=checked,adaptive_decisions_verified=adaptive,
 source_log_arrays_regenerated=len(logs.files),full_count_type_attack_vectors=len(edits_archive)*2000,
 max_score_error=max_error,independent_calibration='FFT convolution; runner uses conditional binomial-tail summation',
 attack_completeness='Monotone endpoints plus feasible row-type edits; no unresolved utility gaps',seconds=time.perf_counter()-start)
 (ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
