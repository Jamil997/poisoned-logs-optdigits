"""Independent conditional-binomial calibration and full decision reconstruction."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,itertools,json,time
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT=Path(__file__).resolve().parent
TYPE=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])))
x,z,action,reward=TYPE.T
F=np.array([x*(2*action-1)*(2*reward-1)>0,z*(2*action-1)*(2*reward-1)>0],dtype=int)

@lru_cache(512)
def marginal(n):
    lo=max(0,int(binom.ppf(1e-15,n,.5)));hi=min(n,int(binom.isf(1e-15,n,.5)))
    v=np.arange(lo,hi+1)
    return v,binom.pmf(v,n,.5),float(binom.cdf(lo-1,n,.5)+binom.sf(hi,n,.5))

@lru_cache(None)
def direct_tail(n,m,a,c,h):
    if not a:return float(binom.sf((h-1)//c,m,.5)),0.0
    if not c:return float(binom.sf((h-1)//a,n,.5)),0.0
    if m>n: n,m,a,c=m,n,c,a
    v,p,omitted=marginal(m)
    val=float(np.sum(p*binom.sf((h-1-c*v)//a,n,.5),dtype=np.longdouble))
    return val,omitted

CHECKED={}

def verified_threshold(n,m,a,c,risk,claimed=None):
    key=(n,m,a,c,float(risk))
    if key in CHECKED:
        if claimed is not None:assert claimed==CHECKED[key]
        return CHECKED[key]
    if risk==0: h=a*n+c*m+1
    elif claimed is not None: h=int(claimed)
    else:
        lo,hi=0,a*n+c*m+1
        while lo<hi:
            mid=(lo+hi)//2;v,e=direct_tail(n,m,a,c,mid)
            if v+e+1e-11<=risk:hi=mid
            else:lo=mid+1
        h=lo
    if risk:
        v,e=direct_tail(n,m,a,c,h)
        assert v+e <= risk+1e-12,(key,h,v,e)
        before,err=direct_tail(n,m,a,c,h-1)
        assert before+err+2e-11>risk,(key,h,before)
    CHECKED[key]=h
    return h

def pick(sc):return np.where(sc.max(axis=0)>0,sc.argmax(axis=0),-1).astype(np.int8)

def main(out):
    begin=time.perf_counter();cfg=json.loads((out/'frozen_protocol.json').read_text())
    for name,h in json.loads((out/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    grid=pd.read_csv(out/'design_grid.csv')
    for r in grid.itertuples():
        h=verified_threshold(r.n,r.m,r.a,r.c,r.alpha/2,r.threshold)
        k=min(r.n,200) if r.threat=='fixed_200' else (r.n+99)//100
        assert r.k==k and r.correction==r.a*k
        assert abs(r.objective-((h-.5+2*r.a*k)/r.denominator-.5))<1e-12
        assert r.actual_cost==r.rho*r.m+(r.n if r.accounting=='total_purchase' else 0)
        assert r.actual_cost<=r.budget
    keys=['accounting','budget','rho','stage','threat','alpha']
    for _,sub in grid.groupby(keys):
        chosen=sub[sub.chosen]
        assert len(chosen)==1
        assert abs(chosen.iloc[0].objective-sub.objective.min())<1e-12
        assert ((sub.a==0)&(sub.m==int(sub.budget.iloc[0]//sub.rho.iloc[0]))).any()
    print('Calibration and complete design grid verified',len(grid),flush=True)
    data=np.load(out/'outcomes.npz');archive=np.load(out/'logs.npz')
    log_names=archive.files;logs={name:archive[name] for name in log_names};seeds=data['seeds']
    for beta in cfg['beta']:
        mean=.5+(2*action-1)*(beta*x-cfg['theta']*z)
        probability=np.where(reward==1,mean,1-mean)/8
        for stream,prefix in [(1,'u'),(2,'t')]:
            sizes=sorted(int(k.split('_')[-1]) for k in log_names if k.startswith(f'{prefix}_{beta}_'))
            for i,seed in enumerate(seeds):
                rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(round(beta*10000)),20261017,stream]))
                last=0;counts=np.zeros(16,dtype=np.int64)
                for n in sizes:
                    if n>last:counts+=rng.multinomial(n-last,probability)
                    assert np.array_equal(counts,logs[f'{prefix}_{beta}_{n}'][i]);last=n
    print('All independent source streams regenerated',len(log_names),flush=True)
    groups=pd.read_csv(out/'groups.csv');witness_cache={};witness_archive={};decisions=0
    for r in groups.itertuples():
        U=logs[f'u_{r.beta}_{r.n}'];V=logs[f't_{r.beta}_{r.m}']
        if r.method=='retain':
            clean=bad=np.full(len(seeds),-1,dtype=np.int8);op=np.full(len(seeds),2,dtype=np.int8)
        else:
            u=(U@F.T).T;v=(V@F.T).T;B=2*np.array([r.g,r.b]);risks=np.full(v.shape,r.alpha/2)
            if r.method=='fixed_influence':risks[:]=(r.alpha/B/(1/B).sum())[:,None]
            if r.method.startswith('pilot_'):
                diff=v[0]-v[1]
                if r.method=='pilot_select':
                    risks[:]=0;risks[0,diff>=0]=r.alpha;risks[1,diff<0]=r.alpha
                else:
                    risks[0,diff>0]=.9*r.alpha;risks[1,diff>0]=.1*r.alpha
                    risks[0,diff<0]=.1*r.alpha;risks[1,diff<0]=.9*r.alpha
            h=np.empty(v.shape,dtype=np.int64)
            for rr in np.unique(risks):h[risks==rr]=verified_threshold(r.n,r.m,r.a,r.c,float(rr))
            def calc(uu):return B[:,None]*(r.a*uu+r.c*v-h-r.a*r.k+.5)/r.denominator
            clean=pick(calc(u))
            if r.a==0 or r.k==0:
                bad=clean.copy();op=np.full(len(seeds),2,dtype=np.int8)
            else:
                witness_key=(r.beta,r.n,r.k)
                if witness_key not in witness_cache:
                    shifts=[]
                    for bit in [0,1]:
                        remaining=np.full(len(seeds),r.k,dtype=int);ed=np.zeros_like(U)
                        for col in np.flatnonzero((F[0]==1)&(F[1]==bit)):
                            ed[:,col]=np.minimum(remaining,U[:,col]);remaining-=ed[:,col]
                        assert np.all(remaining==0) and np.all(ed<=U) and np.all(ed.sum(axis=1)==r.k)
                        shift=(ed@(1-2*F).T).T
                        expected=np.array([-r.k,r.k if bit==0 else -r.k])[:,None]
                        assert np.all(shift==expected)
                        shifts.append(shift)
                        witness_archive[f'{r.beta}_{r.n}_{r.k}_{bit}']=ed[:10].copy()
                    witness_cache[witness_key]=shifts
                candidate=np.array([pick(calc(u+sh)) for sh in witness_cache[witness_key]]+[clean])
                ranks=np.where(candidate==1,0,np.where(candidate<0,1,2));op=ranks.argmin(axis=0).astype(np.int8)
                bad=candidate[op,np.arange(len(seeds))]
            assert r.actual_cost<=r.budget
        assert np.array_equal(clean,data['clean'][r.group_id]),r
        assert np.array_equal(bad,data['adaptive'][r.group_id]),r
        assert np.array_equal(op,data['attack_option'][r.group_id]),r
        decisions+=2*len(seeds)
    np.savez_compressed(out/'witness_examples.npz',**witness_archive)
    report=dict(passed=True,seconds=time.perf_counter()-begin,decisions_reconstructed=decisions,
                calibration_cases=len(CHECKED),design_entries_checked=len(grid),source_arrays_regenerated=len(log_names),
                all_seed_witness_families=len(witness_cache),sample_witness_vectors=10*len(witness_archive),
                independent_calibration='Conditional binomial survival summation with omitted-mass bound; runner uses FFT convolution.',
                attack_completeness='All harmful and retention extrema attained by feasible row edits in every evaluated sample; same two-candidate monotone argument as E8.')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--timing',action='store_true');args=p.parse_args()
    main(ROOT/('timing' if args.timing else 'results'))
