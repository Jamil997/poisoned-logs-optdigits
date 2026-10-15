"""Reconstruct supplemental decisions from saved affine ESLB gates."""
import csv,json
import numpy as np
import pandas as pd
from experiment import ROOT,sample_log,write_json
from analyze import direct_scores,direct_choice

def main():
    config=json.loads((ROOT/'protocol.json').read_text());data=dict(np.load(ROOT/'frozen_models/models.npz'))
    values=data['p'][:,np.arange(len(data['y'])),data['y']].mean(axis=1);truth=values[1:]-values[0]
    checks=0;maxerror=0.;allrows=[]
    for cell in sorted((ROOT/'results_eslb').glob('cell_*')):
        df=pd.read_csv(cell/'runs.csv');ws=json.loads((cell/'witnesses.json').read_text());gates=np.load(cell/'gates.npz')
        for w in ws:
            row=df.iloc[w['row']];n=int(row.n);seed=int(row.seed);gamma=row.gamma
            mu=(1-gamma)*data['p'][0]+gamma/10;x,a,r=sample_log(data,mu,n,gamma,seed)
            key=f'{seed}_high' if row.mc=='high_mc' else str(seed)
            c=gates[key+'_c'];off=gates[key+'_off'];ids=np.array(w['indices'],dtype=int)
            assert len(ids)==len(np.unique(ids)) and len(ids)<=int(n*row.epsilon)
            r[ids]=1-r[ids];ss=direct_scores(c,off,r,int(row.declared_cap))
            error=np.max(np.abs(ss-w['scores']));maxerror=max(maxerror,float(error))
            np.testing.assert_allclose(ss,w['scores'],atol=1e-11,rtol=0)
            j=direct_choice(ss);assert j==row.selected;u=0. if j<0 else truth[j]
            assert abs(u-row.improvement)<1e-12 and row.harm==int(u<0)
            assert np.all(ss>=np.array(w['lower'])-1e-8) and np.all(ss<=np.array(w['upper'])+1e-8)
            floor=min(0 if j<0 else truth[j] for j in w['possible'])
            assert abs(floor-row.utility_floor)<1e-12;checks+=1
        allrows.append(df)
    allrows=pd.concat(allrows);keys=['n','gamma','epsilon','seed','method','condition']
    low=allrows[allrows.mc=='primary_mc'].set_index(keys)
    high=allrows[allrows.mc=='high_mc'].set_index(keys)
    matched=low.loc[high.index]
    assert len(high)==72
    meta=dict(passed=True,decisions_reconstructed=checks,max_score_error=maxerror,
        mc_stability_paired_decisions=len(high),mc_stability_selection_disagreements=int((matched.selected!=high.selected).sum()),
        primary_outcomes=len(low),primary_unique_logs=low.reset_index()[['n','gamma','seed']].drop_duplicates().shape[0])
    write_json(ROOT/'results_eslb/audit.json',meta)
    summary=allrows.groupby(['mc','n','gamma','epsilon','method','condition']).agg(
        seeds=('harm','size'),harm_count=('harm','sum'),mean_improvement=('improvement','mean'),fallback_rate=('fallback','mean')).reset_index()
    summary.to_csv(ROOT/'results_eslb/summary.csv',index=False);print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
