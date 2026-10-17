"""All configurations retained; no selection or acquisition is fitted to outcomes."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

ROOT=Path(__file__).resolve().parent

def analyze(out):
    started=time.perf_counter();cfg=json.loads((out/'frozen_protocol.json').read_text())
    groups=pd.read_csv(out/'groups.csv');raw=np.load(out/'outcomes.npz');N=len(raw['seeds'])
    total_intervals=2*len(groups)
    tail=.05/(4*total_intervals)
    frames=[]
    for condition,key in [('clean_with_protection','clean'),('exact_adaptive','adaptive')]:
        sel=raw[key];ng=(sel==0).sum(axis=1);nb=(sel==1).sum(axis=1)
        gain=2*groups.g.to_numpy()*groups.beta.to_numpy();loss=-2*groups.b.to_numpy()*groups.theta.to_numpy()
        def bounds(count):
            lower=np.where(count==0,0,beta_distribution.ppf(tail,count,N-count+1))
            upper=np.where(count==N,1,beta_distribution.ppf(1-tail,count+1,N-count))
            return lower,upper
        gl,gu=bounds(ng);bl,bu=bounds(nb)
        df=groups.copy();df['condition']=condition;df['seeds']=N
        df['utility_pp']=100*(gain*ng/N+loss*nb/N)
        df['utility_lower_pp']=100*(gain*gl+loss*bu)
        df['utility_upper_pp']=100*(gain*gu+loss*bl)
        df['harm_percent']=100*nb/N;df['harm_upper_percent']=100*bu
        df['analytic_harm_limit_percent']=100*df.alpha
        df['fallback_percent']=100*(sel<0).mean(axis=1)
        df['best_available_gain_pp']=100*gain
        df.loc[df.method=='retain',['utility_lower_pp','utility_upper_pp','harm_upper_percent','analytic_harm_limit_percent']]=0.0
        df['source_choice']=np.where(df.method=='retain','retain',np.where(df.a==0,'trusted_only',np.where(df.c==0,'main_only','mixed')))
        frames.append(df)
    summary=pd.concat(frames,ignore_index=True)
    summary.to_csv(out/'summary.csv',index=False)
    ad=summary.query('condition=="exact_adaptive"')
    rows=[]
    keycols=['accounting','budget','rho','stage','threat','alpha','beta','g','b']
    rng=np.random.default_rng(172026)
    for key,sub in ad.query('beta==0.03 and alpha==0.01').groupby(keycols):
        ours=sub[sub.method=='joint_choice'].iloc[0];sg=raw['adaptive'][int(ours.group_id)]
        gain=2*ours.g*ours.beta;bad=-2*ours.b*ours.theta
        values=np.array([0.,gain,bad])
        for _,other in sub.iterrows():
            if other.method in ['joint_choice','retain']:continue
            sb=raw['adaptive'][int(other.group_id)]
            counts=np.bincount((sg+1)*3+(sb+1),minlength=9)
            diffs=(values[:,None]-values[None,:]).ravel()
            boot=rng.multinomial(N,counts/N,size=5000)@diffs/N*100
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(zip(keycols,key),comparator=other.method,difference_pp=float((counts@diffs)/N*100),
                             pointwise_paired_lower_pp=lo,pointwise_paired_upper_pp=hi,
                             simultaneous_difference_lower_pp=ours.utility_lower_pp-other.utility_upper_pp))
    pd.DataFrame(rows).to_csv(out/'contrasts.csv',index=False)
    minima=[]
    costkeys=['accounting','rho','threat','alpha','beta','g','b','method']
    for key,sub in ad.query('stage=="primary"').groupby(costkeys):
        sub=sub.sort_values(['actual_cost','budget'])
        targets={'absolute_0.1pp':100*cfg['primary_useful_gain'],
                 'relative_75pct':cfg['secondary_useful_fraction']*sub.best_available_gain_pp.iloc[0]}
        for name,target in targets.items():
            valid=sub[sub.utility_lower_pp>=target]
            winner=valid.iloc[0] if len(valid) else None
            minima.append(dict(zip(costkeys,key),target=name,target_pp=target,certified=winner is not None,
                               smallest_tested_cost=float(winner.actual_cost) if winner is not None else np.nan,
                               acquisition_budget=float(winner.budget) if winner is not None else np.nan,
                               utility_pp=float(winner.utility_pp) if winner is not None else np.nan,
                               utility_lower_pp=float(winner.utility_lower_pp) if winner is not None else np.nan))
    pd.DataFrame(minima).to_csv(out/'minimum_cost.csv',index=False)
    report=dict(seconds=time.perf_counter()-started,groups=len(groups),independent_seeds=N,
                simultaneous_utility_confidence=.95,utility_intervals=total_intervals,
                event_tail_probability=tail,paired_intervals='Pointwise descriptive 95% paired bootstrap, 5000 multinomial resamples.',
                source_choice_counts=ad.query('method=="joint_choice" and beta==0.03 and g==0.05').source_choice.value_counts().to_dict(),
                caution='Price ratios and acquisition models are scenarios, not measured costs. Minimum cost means smallest tested budget with simultaneous evidence; no interpolation or population-optimal claim.')
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(ad.query('method in ["joint_choice","all_trusted","all_main"] and beta==0.03 and g==0.05 and alpha==0.01 and (budget==20000 or stage=="legacy_budget")')[['accounting','budget','rho','threat','method','n','m','reference_weight','utility_pp','harm_percent','source_choice']].round(3).to_string(index=False),flush=True)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--timing',action='store_true');a=p.parse_args()
    analyze(ROOT/('timing' if a.timing else 'results'))
