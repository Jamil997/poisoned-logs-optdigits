"""Acquisition and testing decisions; no true policy quality enters this module."""
from functools import lru_cache
import numpy as np
from scipy.stats import binom
from scipy.signal import fftconvolve

GUARD = 1e-11

@lru_cache(512)
def short_binomial(n):
    if n == 0:
        return 0, np.ones(1), 0.0
    lo = max(0, int(binom.ppf(1e-14, n, .5)))
    hi = min(n, int(binom.isf(1e-14, n, .5)))
    return lo, binom.pmf(np.arange(lo, hi + 1), n, .5), float(binom.cdf(lo-1,n,.5)+binom.sf(hi,n,.5))

@lru_cache(192)
def null_survival(n, m, a, c):
    def array(nn, w):
        if w == 0: return 0, np.ones(1), 0.0
        lo, p, lost = short_binomial(nn)
        out = np.zeros(w*(len(p)-1)+1)
        out[::w] = p
        return w*lo, out, lost
    off_u, pu, eu = array(n,a)
    off_t, pt, et = array(m,c)
    pmf = np.maximum(fftconvolve(pu,pt), 0)
    sf = np.r_[np.cumsum(pmf[::-1])[::-1], 0.0]
    return off_u + off_t, sf, eu+et

@lru_cache(None)
def critical(n, m, a, c, risk):
    if risk == 0: return a*n+c*m+1
    off,sf,lost = null_survival(n,m,a,c)
    indexes = np.flatnonzero(sf+lost+GUARD <= risk)
    if not len(indexes): raise ArithmeticError('Insufficient numerical accuracy for requested risk')
    return off+int(indexes[0])

def edit_cap(n, threat):
    return min(n,200) if threat == 'fixed_200' else (n+99)//100

def cost(n,m,rho,accounting):
    return rho*m+(n if accounting == 'total_purchase' else 0)

def plans(cfg,budget,rho,accounting):
    rows=set()
    if accounting == 'total_purchase':
        for f in cfg['trusted_budget_shares']:
            m=int(budget*f//rho)
            n=budget-rho*m if f<1 else 0
            rows.add((n,m))
        m=budget//(19+rho)
        fixed=(19*m,m)
    else:
        n=cfg['historical_log_size']
        for f in cfg['trusted_budget_shares']:
            rows.add((n,int(budget*f//rho)))
        fixed=(n,budget//rho)
    rows.update([fixed,(0,budget//rho),(budget if accounting=='total_purchase' else cfg['historical_log_size'],0)])
    return sorted(p for p in rows if sum(p)>0),fixed

def weight_grid(cfg,n,m):
    if not n: return [(0,1)]
    if not m: return [(1,0)]
    return [tuple(x) for x in cfg['integer_source_weights']]

def design_rows(cfg,budget,rho,accounting,threat,alpha):
    pp,fixed=plans(cfg,budget,rho,accounting)
    rows=[]
    for n,m in pp:
        k=edit_cap(n,threat)
        for a,c in weight_grid(cfg,n,m):
            d=a*n+c*m
            h=critical(n,m,a,c,alpha/2)
            correction=a*k
            rows.append(dict(n=n,m=m,k=k,a=a,c=c,denominator=d,threshold=h,
                             correction=correction,objective=(h-.5+2*correction)/d-.5,
                             actual_cost=cost(n,m,rho,accounting),reference_weight=c*m/d))
    def key(r):
        return (r['objective'],r['actual_cost'],1-r['reference_weight'],r['n'],r['m'])
    chosen=min(rows,key=key)
    fixed_best=min([r for r in rows if (r['n'],r['m'])==fixed],key=key)
    return chosen,fixed_best,fixed,rows

def scores(u,v,B,a,c,d,h,correction):
    return B[:,None]*(a*u+c*v-h-correction+.5)/d

def select(score):
    return np.where(score.max(axis=0)>0,score.argmax(axis=0),-1).astype(np.int8)

def thresholds(n,m,a,c,alpha,B,trusted,method):
    risks=np.full_like(trusted,alpha/2,dtype=float)
    if method=='fixed_influence':
        risks[:]=(alpha/B/(1/B).sum())[:,None]
    if method in ('pilot_select','pilot_weight'):
        diff=trusted[0]-trusted[1]
        if method=='pilot_select':
            risks[:]=0
            risks[0,diff>=0]=alpha
            risks[1,diff<0]=alpha
        else:
            risks[0,diff>0]=.9*alpha; risks[1,diff>0]=.1*alpha
            risks[0,diff<0]=.1*alpha; risks[1,diff<0]=.9*alpha
    h=np.empty_like(trusted,dtype=np.int64)
    for risk in np.unique(risks):
        h[risks==risk]=critical(n,m,a,c,float(risk))
    return h
