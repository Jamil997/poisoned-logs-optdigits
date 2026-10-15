"""Post-run mechanism checks; these do not change policies or decisions."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT,build_specs,e2

def main():
    config=json.loads((ROOT/'protocol.json').read_text())
    data=dict(np.load(ROOT/'frozen_models/models.npz'));logs=np.load(ROOT/'results_main/logs.npz')
    rows=[]
    for n in config['n']:
        for gamma in config['gamma']:
            _,specs,_=build_specs(data,n,gamma,config)
            for seed in range(20000,20100):
                key=f'{n}_{gamma}_{seed}';x=logs[key+'_x'];a=logs[key+'_a']
                r=(a==data['y'][x]).astype(float)
                for method in config['base_methods']:
                    s=specs[method];c=s['c'][:,x,a];off=s['z0'][:,x,a].mean(axis=1)-s['rad']-s['bias']
                    for family in range(3):
                        lo,hi=2*family,2*family+1
                        diff=c[hi]-c[lo];margin=np.mean(diff*r)+off[hi]-off[lo]
                        damage=diff*r-np.minimum(diff,0)
                        for eps in config['epsilon']:
                            k=int(n*eps);lb=margin-float(e2.top_sum(damage,k))/n
                            rows.append(dict(n=n,gamma=gamma,seed=seed,epsilon=eps,method=method,
                                family=family,clean_margin=margin,minimum_all_real_margin=lb,
                                worse_sibling_excluded=int(lb>1e-9)))
    pd.DataFrame(rows).to_csv(ROOT/'sibling_certificates.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['method','family']).agg(
        all_real_order_preserved=('worse_sibling_excluded','sum'),cases=('worse_sibling_excluded','size'),
        minimum_margin=('minimum_all_real_margin','min'))
    summary.to_csv(ROOT/'sibling_summary.csv');print(summary.to_string())
    datasets={'primary':pd.read_csv(ROOT/'results_main/runs.csv')}
    if (ROOT/'subset_main_audit.json').exists():
        frames=[]
        for directory in sorted((ROOT/'subsets').iterdir()):
            d=pd.read_csv(directory/'results_main/runs.csv');d['library']=directory.name;frames.append(d)
        datasets['all_subsets']=pd.concat(frames)
    summaries=[]
    for name,df in datasets.items():
        d=df.groupby(['method','condition','epsilon']).agg(mean_harm=('harm','mean'),
            mean_improvement=('improvement','mean'),fallback_rate=('fallback','mean'),
            certified_floor=('utility_floor','mean'),unresolved=('attack_utility_gap',lambda x:int((x>1e-12).sum())),
            cases=('harm','size')).reset_index()
        d['study']=name;summaries.append(d)
    pd.concat(summaries).to_csv(ROOT/'budget_summary.csv',index=False)

if __name__=='__main__':main()
