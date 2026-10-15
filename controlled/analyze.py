"""Reconstruct all grouped-row attacks and exact worst outcomes independently."""
from pathlib import Path
import json,gzip,time,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist,binom
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_tables(n,amp,alpha):
 t=np.arange(n+1);prob=np.zeros(n+1);prob[1:]=beta_dist.isf(1-alpha/2,t[1:],n-t[1:]+1)
 B=np.array([.1,2*amp]);l=np.log(2/alpha)
 rad=B*(l/(3*n)+np.sqrt((l/(3*n))**2+l/(2*n)))
 return {'binomial':B[:,None]*(prob-.5),'bernstein':B[:,None]*(t/n-.5)-rad[:,None]}

def audit(mode):
 began=time.perf_counter();directory=ROOT/('results' if mode=='main' else 'risk_results');meta=json.loads((directory/'metadata.json').read_text())
 cfgfile='protocol.json' if mode=='main' else 'risk_protocol.json';source='experiment.py' if mode=='main' else 'risk_curve.py'
 assert meta['protocol_sha256']==sha(ROOT/cfgfile) and meta['source_sha256']==sha(ROOT/source)
 if mode=='risk':assert meta['experiment_sha256']==sha(ROOT/'experiment.py')
 df=pd.read_csv(directory/'runs.csv.gz');logs=np.load(directory/'logs.npz')
 with gzip.open(directory/'witnesses.json.gz','rt') as f:ws={w['row']:w for w in json.load(f)}
 types=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])));s,z,a,r=types.T
 f=np.array([s*(2*a-1)*(2*r-1)>0,z*(2*a-1)*(2*r-1)>0],dtype=int)
 cache={};maxerr=0.;checked=0;log_count=0;min_groups=10**9
 groups=['n','beta','seed'] if mode=='main' else ['n','seed']
 for key,part in df.groupby(groups,sort=False):
  if mode=='main':n,beta,seed=key;logkey=f'{n}_{beta}_{seed}';rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),177]))
  else:n,seed=key;beta=.03;logkey=str(seed);rng=np.random.default_rng(np.random.SeedSequence([seed,n,177]))
  counts=logs[logkey];mean=.5+(2*a-1)*(beta*s-.001*z);probs=np.where(r,mean,1-mean)/8
  np.testing.assert_array_equal(counts,rng.multinomial(n,probs));t=f@counts;log_count+=1
  for row in part.itertuples():
   i=row.Index;eps=row.epsilon;k=int(n*eps);amp=row.bad_amplitude if mode=='main' else .45;alpha=.05 if mode=='main' else row.alpha
   if row.method=='retain':assert row.selected==-1 and row.improvement==0 and row.harm==0 and row.fallback==1;checked+=1;continue
   family,protection=row.method.split('_');kd=k if protection=='corrected' else 0
   ck=(n,amp,alpha)
   if ck not in cache:cache[ck]=independent_tables(n,amp,alpha)
   table=cache[ck][family]
   condition=row.condition if mode=='main' else 'adaptive'
   if condition=='adaptive':
    w=ws[i];ed=np.array(w['edits']);assert np.all(ed>=0) and np.all(ed<=counts) and ed.sum()<=k
    tv=t+(1-2*f)@ed;np.testing.assert_array_equal(tv,w['favorable'])
   elif condition=='clean':tv=t
   else:
    rngr=np.random.default_rng(np.random.SeedSequence([seed,n,int(beta*10000),k,178]));ed=rngr.multivariate_hypergeometric(counts,k);tv=t+(1-2*f)@ed
   score=table[np.arange(2),np.maximum(0,tv-kd)];sel=int(np.argmax(score)) if max(score)>1e-9 else -1
   truth=[.1*beta,-2*amp*.001];u=0. if sel<0 else truth[sel]
   assert sel==row.selected and abs(u-row.improvement)<1e-12 and row.harm==int(u<0) and row.fallback==int(sel<0)
   if condition=='adaptive':
    err=float(np.max(abs(score-w['scores'])));maxerr=max(maxerr,err);assert err<1e-11
    opposite=counts[(f[0]==1)&(f[1]==0)].sum();same=counts[(f[0]==1)&(f[1]==1)].sum();assert min(opposite,same)>=k;min_groups=min(min_groups,int(min(opposite,same)-k))
    # Independent endpoint dominance proof. Bad attains its largest possible
    # score while good attains its smallest; all-down then targets retention.
    bad_upper=table[1,min(n,max(0,t[1]+k-kd))];good_lower=table[0,max(0,t[0]-k-kd)];bad_lower=table[1,max(0,t[1]-k-kd)]
    if bad_upper>max(1e-9,good_lower):worst=truth[1]
    elif max(good_lower,bad_lower)<=1e-9:worst=0.
    else:worst=truth[0]
    assert abs(u-worst)<1e-12
   checked+=1
 report=dict(passed=True,decisions=checked,logs=log_count,witnesses=len(ws),max_score_error=maxerr,minimum_attack_group_slack=min_groups,exact_worst_utility_verified=True,seconds=time.perf_counter()-began)
 (ROOT/f'{mode}_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(mode,report,flush=True)
 return df

def summarize(df,mode):
 keys=['n','beta','bad_amplitude','epsilon','method','condition'] if mode=='main' else ['epsilon','alpha','method']
 out=[];rng=np.random.default_rng(77219);ix=rng.integers(1000,size=(10000,1000))
 for key,p in df.groupby(keys):
  p=p.sort_values('seed');d=dict(zip(keys,key));u=p.improvement.to_numpy()*100;h=int(p.harm.sum());N=len(p);boot=u[ix].mean(axis=1)
  d.update(utility_pp=float(u.mean()),utility_low=float(np.quantile(boot,.025)),utility_high=float(np.quantile(boot,.975)),harm=float(h/N),harm_count=h,harm_upper95=float(beta_dist.ppf(.95,h+1,N-h)) if h<N else 1.,fallback=float(p.fallback.mean()),seeds=N)
  out.append(d)
 summary=pd.DataFrame(out);summary.to_csv(ROOT/f'{mode}_summary.csv',index=False)
 if mode=='main':p=df[(df.n==20000)&(df.beta==.03)&(df.bad_amplitude==.45)&(df.epsilon==.01)&(df.condition=='adaptive')]
 else:p=df[(df.epsilon==.01)&(df.alpha==.01)]
 pivot=p.pivot(index='seed',columns='method',values='improvement').sort_index();diff=(pivot.binomial_corrected-pivot.bernstein_corrected).to_numpy()*100;boot=diff[ix].mean(axis=1)
 contrast=dict(contrast='corrected binomial minus corrected Bernstein',difference_pp=float(diff.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),risk_level=.05 if mode=='main' else .01)
 (ROOT/f'{mode}_primary_contrast.json').write_text(json.dumps(contrast,indent=2)+'\n');print(mode,contrast,flush=True)
 return summary

def figures():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 m=pd.read_csv(ROOT/'main_summary.csv');r=pd.read_csv(ROOT/'risk_summary.csv');b=pd.read_csv(ROOT/'exact_risk_bounds.csv')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
 colors={'bernstein_corrected':'#546a7b','binomial_corrected':'#007f78','binomial_nominal':'#b13f32','bernstein_nominal':'#cc8e3a'}
 t=m[(m.n==20000)&(m.beta==.03)&(m.bad_amplitude==.45)&(m.condition=='adaptive')]
 for method,color in colors.items():
  z=t[t.method==method].sort_values('epsilon');axes[0].plot(z.epsilon*100,z.utility_pp,'o-',label=method.replace('_',' '),color=color)
 axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xlabel='Reward corruption budget (%)',ylabel='Mean true improvement (pp)',title='Controlled benchmark: exact worst attacks');axes[0].legend(fontsize=8)
 for family,color in [('binomial','#007f78'),('bernstein','#546a7b')]:
  z=r[(r.epsilon==.01)&(r.method==family+'_corrected')].sort_values('alpha');axes[1].plot(z.harm*100,z.utility_pp,'o-',color=color,label=family)
  for q in z.itertuples():axes[1].annotate(f'{q.alpha:g}',(q.harm*100,q.utility_pp),xytext=(3,3),textcoords='offset points',fontsize=7)
 axes[1].set(xlabel='Observed harmful deployments (%)',ylabel='Mean true improvement (pp)',title='Fresh seeds: confidence–utility tradeoff');axes[1].legend(fontsize=8)
 for family,color in [('binomial','#007f78'),('bernstein','#546a7b')]:
  z=r[(r.epsilon==.01)&(r.method==family+'_corrected')].merge(b[b.family==family],on='alpha').sort_values('alpha');axes[2].plot(z.exact_harm_upper*100,z.utility_pp,'o-',color=color,label=family)
 axes[2].set(xlabel='Analytic upper bound on harm probability (%)',ylabel='Mean true improvement (pp)',title='Model-specific bound includes unseen logs');axes[2].legend(fontsize=8)
 fig.suptitle('A more useful confidence gate carries an explicit risk level',fontsize=14);fig.savefig(ROOT/'tradeoff.png',dpi=180);plt.close(fig)

if __name__=='__main__':
 import sys
 mode=sys.argv[1];df=audit(mode);summarize(df,mode)
 if (ROOT/'main_summary.csv').exists() and (ROOT/'risk_summary.csv').exists():figures()
