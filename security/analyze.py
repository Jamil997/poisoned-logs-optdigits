"""Reconstruct every recorded decision and summarize paired finite-grid results."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, build_specs, write_json

def direct_scores(c,off,r,k):
    out=np.einsum('ji,i->j',c,r)/len(r)+off
    if k:
        damage=c*r-np.minimum(c,0)
        out-=np.sort(damage,axis=1)[:,-min(k,len(r)):].sum(axis=1)/len(r)
    return out

def direct_choice(s):
    return int(np.argmax(s)) if max(s)>1e-9 else -1

def audit(out):
    start=time.perf_counter();config=json.loads((ROOT/'protocol.json').read_text())
    df=pd.read_csv(out/'runs.csv');ws=json.loads((out/'witnesses.json').read_text())
    data=dict(np.load(ROOT/'frozen_models/models.npz'));logs=np.load(out/'logs.npz')
    truth=data['p'][:,np.arange(len(data['y'])),data['y']].mean(axis=1)
    truth=truth[1:]-truth[0];cache={};checks=0;dual_checks=0;max_score_error=0.
    byrow={w['row']:w for w in ws};max_gap=0.
    for (n,gamma,seed),part in df.groupby(['n','gamma','seed'],sort=False):
        if (n,gamma) not in cache: cache[n,gamma]=build_specs(data,n,gamma,config)
        mu,specs,_=cache[n,gamma];prefix=f'{n}_{gamma}_{seed}'
        x=logs[prefix+'_x'].astype(int);a=logs[prefix+'_a'].astype(int)
        r=(a==data['y'][x]).astype(float)
        loghash=hashlib.sha256(x.astype('<i4').tobytes()+a.astype('uint8').tobytes()+r.astype('uint8').tobytes()).hexdigest()
        gates={name:(s['c'][:,x,a],s['z0'][:,x,a].mean(axis=1)-s['rad']-s['bias']) for name,s in specs.items()}
        for index,row in part.iterrows():
            assert row.log_sha256==loghash
            if row.method=='retain':
                assert row.selected==-1 and row.improvement==0;checks+=1;continue
            w=byrow[index];ids=np.array(w['indices'],dtype=int)
            assert len(ids)==len(np.unique(ids)) and len(ids)<=int(n*row.epsilon)
            assert not len(ids) or (min(ids)>=0 and max(ids)<n)
            if row.condition=='clean': assert len(ids)==0
            rp=r.copy();rp[ids]=1-rp[ids]
            method=row.method.split('_',1)[1];c,off=gates[method]
            ss=direct_scores(c,off,rp,int(row.declared_cap));err=np.max(np.abs(ss-np.array(w['scores'])))
            max_score_error=max(max_score_error,float(err));np.testing.assert_allclose(ss,w['scores'],atol=1e-11,rtol=0)
            sel=direct_choice(ss);assert sel==row.selected
            u=0. if sel<0 else truth[sel]
            assert abs(u-row.improvement)<1e-12
            assert row.harm==int(u<0) and row.fallback==int(sel<0)
            assert np.all(ss>=np.array(w['lower'])-1e-8) and np.all(ss<=np.array(w['upper'])+1e-8)
            possible=w['possible'];assert sel in possible
            floor=min(0. if j<0 else truth[j] for j in possible)
            assert abs(floor-row.utility_floor)<1e-12
            assert row.harm_excluded==int(all(j<0 or truth[j]>=0 for j in possible))
            if row.condition=='adaptive':
                assert abs(max(0.,u-floor)-row.attack_utility_gap)<1e-12
                max_gap=max(max_gap,float(row.attack_utility_gap))
                # Rebuild each all-real relaxation's Lagrangian upper bound from
                # the saved dual multipliers; no trust in solver status required.
                for sub in w['subproblems']:
                    if sub.get('kind')!='all_real_L1_dual' or sub.get('dual') is None: continue
                    target=sub['target'];base=direct_scores(c,off,r,0);g=c*(1-2*r)/n
                    tb=0 if target<0 else base[target];tg=np.zeros(n) if target<0 else g[target]
                    Az=[np.ones(n)];At=[0.];b=[int(n*row.epsilon)]
                    if target>=0: Az.append(-tg);At.append(1.);b.append(tb)
                    for j in range(len(truth)):
                        if j!=target: Az.append(g[j]-tg);At.append(1.);b.append(tb-base[j])
                    lam=np.array(sub['dual']);assert np.min(lam)>=0 and abs(lam@At-1)<1e-10
                    ub=float(lam@b+np.maximum(0,-lam@np.array(Az)).sum())
                    assert abs(ub-sub['upper_margin'])<1e-10
                    if target not in possible: assert ub<-1e-8
                    dual_checks+=1
            checks+=1
        if len(cache) and (seed%25==0): print(f'audit n={n} gamma={gamma} seed={seed} decisions={checks}',flush=True)
    result=dict(decisions_reconstructed=checks,all_decisions=len(df),attack_witnesses=len(ws),
        dual_certificates_recomputed=dual_checks,max_score_error=max_score_error,max_unresolved_utility_gap=max_gap,
        seconds=time.perf_counter()-start,passed=True)
    write_json(out/'audit.json',result);return result

def summarize(out):
    df=pd.read_csv(out/'runs.csv');rows=[]
    for key,p in df.groupby(['n','gamma','epsilon','method','condition']):
        n,g,eps,method,condition=key;R=len(p);h=int(p.harm.sum());u=p.improvement.to_numpy()
        rad=float(t.ppf(.975,R-1)*u.std(ddof=1)/np.sqrt(R))
        rows.append(dict(n=n,gamma=g,epsilon=eps,method=method,condition=condition,seeds=R,harm_count=h,
            harm_rate=h/R,harm_upper_95=1. if h==R else float(beta.ppf(.95,h+1,R-h)),
            improvement_pp=100*u.mean(),improvement_low_pp=100*(u.mean()-rad),improvement_high_pp=100*(u.mean()+rad),
            fallback_rate=p.fallback.mean(),useful_rate=p.useful.mean(),
            certified_floor_pp=100*p.utility_floor.mean(),unresolved=int((p.attack_utility_gap>1e-12).sum()),
            all_real_harm_excluded=int(p.harm_excluded.sum())))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    adaptive=df[df.condition=='adaptive'];methods=sorted(adaptive.method.unique());aggregate=[]
    rng=np.random.default_rng(841);seeds=sorted(adaptive.seed.unique());indices=rng.integers(len(seeds),size=(10000,len(seeds)))
    paired={}
    for method in methods:
        p=adaptive[adaptive.method==method];cluster=p.groupby('seed').improvement.mean().reindex(seeds).to_numpy()
        boot=cluster[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        agg=dict(method=method,mean_improvement_pp=100*cluster.mean(),low_pp=100*lo,high_pp=100*hi,
            max_cell_harm=float(p.groupby(['n','gamma','epsilon']).harm.mean().max()),
            harmful_outcomes=int(p.harm.sum()),fallback_rate=p.fallback.mean(),
            certified_floor_pp=100*p.utility_floor.mean(),unresolved=int((p.attack_utility_gap>1e-12).sum()),
            observed_outcomes=len(p),all_real_harm_excluded=int(p.harm_excluded.sum()))
        aggregate.append(agg);paired[method]=cluster
    for a in aggregate:
        a['empirically_nondominated']=not any(b['max_cell_harm']<=a['max_cell_harm'] and b['mean_improvement_pp']>=a['mean_improvement_pp']-1e-12 and
            (b['max_cell_harm']<a['max_cell_harm'] or b['mean_improvement_pp']>a['mean_improvement_pp']+1e-12) for b in aggregate)
    pd.DataFrame(aggregate).to_csv(out/'aggregate.csv',index=False)
    comparisons=[]
    for base in ['ips_point','dr_point','ips_h','centered_dr_b','dr_b','dros_b']:
        d=paired['corrected_'+base]-paired['nominal_'+base];boot=d[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        comparisons.append(dict(base=base,contrast='corrected minus nominal, mean over all 12 grid cells',
            difference_pp=100*d.mean(),low_pp=100*lo,high_pp=100*hi))
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    information=dict(inference=f'{len(seeds)} paired seed clusters; aggregate weights all 12 grid cells equally; pointwise 95 percent intervals',
        zero_harm_pointwise_upper=1-.05**(1/len(seeds)),
        zero_harm_simultaneous_12_cells_upper=1-(.05/12)**(1/len(seeds)),
        unique_logs=df[['n','gamma','seed']].drop_duplicates().shape[0],outcomes=len(df),
        note='Bootstrap does not remove unresolved attack optimization gaps. Feasible attacks upper-bound worst-case utility; certified floors lower-bound it.')
    write_json(out/'inference.json',information)
    return summary,pd.DataFrame(aggregate)

def figures(out,summary,aggregate):
    selected=['nominal_dr_point','nominal_centered_dr_b','corrected_centered_dr_b','corrected_dr_b','corrected_dros_b','corrected_ips_h']
    labels={'nominal_dr_point':'DR point estimate','nominal_centered_dr_b':'Centered DR','corrected_centered_dr_b':'Corrected centered DR',
            'corrected_dr_b':'Corrected fitted DR','corrected_dros_b':'Corrected shrinkage DR','corrected_ips_h':'Corrected IPS / Hoeffding'}
    colors=['#8B5CF6','#CE6B29','#087F8C','#164B72','#849324','#777777']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,(n,g) in zip(axes.flat,[(5000,.2),(5000,.5),(20000,.2),(20000,.5)]):
        for m,color in zip(selected,colors):
            p=summary[(summary.n==n)&(summary.gamma==g)&(summary.method==m)&(summary.condition=='adaptive')].sort_values('epsilon')
            x=p.epsilon*100;y=p.improvement_pp
            ax.plot(x,y,'o-',label=labels[m],color=color,linewidth=1.8,markersize=4)
            ax.fill_between(x,p.certified_floor_pp,y,color=color,alpha=.10)
        ax.set_title(f'n = {n:,}; exploration = {g:.0%}');ax.axhline(0,color='#aaa',lw=.8);ax.grid(alpha=.15)
        ax.set_xlabel('Editable rewards (%)');ax.set_ylabel('True improvement (percentage points)')
    handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.02))
    fig.suptitle('OptDigits: retained improvement under each gate’s adaptive attack\nShading: remaining gap to the certified worst-case lower bound',fontsize=14)
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(out/'security_results.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,7));a=aggregate.sort_values('mean_improvement_pp')
    long_names={'ips_point':'IPS point estimate','dr_point':'DR point estimate','ips_h':'IPS / Hoeffding',
                'centered_dr_b':'centered DR / Bernstein','dr_b':'fitted DR / Bernstein','dros_b':'shrinkage DR / Bernstein'}
    names=['Always retain incumbent' if m=='retain' else
           ('Corrected ' if m.startswith('corrected_') else 'Unprotected ')+long_names[m.split('_',1)[1]] for m in a.method]
    ax.barh(names,a.mean_improvement_pp,color=['#087F8C' if m.startswith('corrected') else '#546A7B' for m in a.method])
    ax.errorbar(a.mean_improvement_pp,names,xerr=[a.mean_improvement_pp-a.low_pp,a.high_pp-a.mean_improvement_pp],fmt='none',ecolor='#222',capsize=3)
    ax.scatter(a.certified_floor_pp,names,marker='|',s=180,c='#E6A100',label='Certified worst-case lower bound')
    ax.set_xlabel('Mean true improvement (percentage points; equal weight across 12 cells)')
    ax.set_title('All methods had zero harmful selections in this grid\nBars: attained attack utility; error bars: paired 95% bootstrap intervals');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();fig.savefig(out/'comparison.png',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results_main');p.add_argument('--skip-audit',action='store_true')
    a=p.parse_args()
    if not a.skip_audit:print(json.dumps(audit(a.results),indent=2))
    summary,aggregate=summarize(a.results);figures(a.results,summary,aggregate);print(aggregate.round(5).to_string(index=False))
