"""Audit every subset, including exact reuse of E3 logs; no selected library."""
import argparse,json
import numpy as np
import pandas as pd
import analyze
from experiment import ROOT,write_json

def main(mode):
    protocol=json.loads((ROOT/'subset_protocol.json').read_text());frames=[];audits={}
    original=np.load(ROOT/f'results_{mode}/logs.npz')
    full=np.load(ROOT/'frozen_models/models.npz')
    for item in protocol['libraries']:
        directory=ROOT/'subsets'/item['name'];out=directory/f'results_{mode}'
        local=np.load(directory/'frozen_models/models.npz')
        ix=[0]+[j+1 for j in item['candidate_indices']]
        np.testing.assert_array_equal(local['p'],full['p'][ix])
        np.testing.assert_array_equal(local['pv'],full['pv'][ix])
        logs=np.load(out/'logs.npz');assert set(logs.files)==set(original.files)
        for key in logs.files:np.testing.assert_array_equal(logs[key],original[key])
        analyze.ROOT=directory;analyze.previous.ROOT=directory
        audits[item['name']]=analyze.audit(out)
        analyze.summarize(out)
        part=pd.read_csv(out/'runs.csv');part['library']=item['name'];frames.append(part)
        print('AUDITED '+item['name'],flush=True)
    result=pd.concat(frames,ignore_index=True);ad=result[result.condition=='adaptive']
    seed_ids=sorted(result.seed.unique());rng=np.random.default_rng(841)
    draws=rng.integers(len(seed_ids),size=(10000,len(seed_ids)));rows=[]
    for method,p in ad.groupby('method'):
        cluster=p.groupby('seed').improvement.mean().reindex(seed_ids).to_numpy()
        lo,hi=np.quantile(cluster[draws].mean(axis=1),[.025,.975])
        rows.append(dict(method=method,mean_improvement_pp=100*cluster.mean(),low_pp=100*lo,high_pp=100*hi,
            max_cell_harm=p.groupby(['library','n','gamma','epsilon']).harm.mean().max(),
            mean_harm_rate=p.harm.mean(),harmful_outcomes=int(p.harm.sum()),fallback_rate=p.fallback.mean(),
            certified_floor_pp=100*p.utility_floor.mean(),unresolved=int((p.attack_utility_gap>1e-12).sum()),
            observed_outcomes=len(p),all_real_harm_excluded=int(p.harm_excluded.sum())))
    pd.DataFrame(rows).to_csv(ROOT/f'subset_{mode}_aggregate.csv',index=False)
    summaries=[]
    for item in protocol['libraries']:
        p=pd.read_csv(ROOT/'subsets'/item['name']/f'results_{mode}/summary.csv')
        p['library']=item['name'];summaries.append(p)
    pd.concat(summaries).to_csv(ROOT/f'subset_{mode}_cells.csv',index=False)
    write_json(ROOT/f'subset_{mode}_audit.json',dict(passed=True,library_audits=audits,
        decisions_reconstructed=len(result),unique_logs=400 if mode=='main' else 8,
        logs_identical_to_primary=True,frozen_probabilities_unchanged=True,
        inference='Exploratory 72-cell average; bootstrap 100 paired seed clusters in main. Libraries reuse the same logs.'))
    print(pd.DataFrame(rows).round(5).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['main','timing'],default='main')
    args=parser.parse_args();main(args.mode)
