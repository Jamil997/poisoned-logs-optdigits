"""Repartition only the slow cell; preserve all other completed main cells."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import copy,csv,json,time
import numpy as np
from experiment import ROOT,run,write_csv,write_json,sha

def worker(item):
    config,path=item;run(config,path,'main');return str(path)

def merge(paths,out):
    rows=[];witnesses=[];logs={};metas=[];tuning={}
    for path in paths:
        with open(path/'runs.csv') as f:part=list(csv.DictReader(f))
        shift=len(rows);rows.extend(part)
        for w in json.loads((path/'witnesses.json').read_text()):w['row']+=shift;witnesses.append(w)
        logs.update(dict(np.load(path/'logs.npz')))
        metas.append(json.loads((path/'metadata.json').read_text()))
        tuning.update(json.loads((path/'shrinkage_tuning.json').read_text()))
    out.mkdir(exist_ok=True);write_csv(out/'runs.csv',rows);write_json(out/'witnesses.json',witnesses)
    np.savez_compressed(out/'logs.npz',**logs);write_json(out/'shrinkage_tuning.json',tuning)
    (out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
    return dict(outcomes=len(rows),independent_logs=len(logs)//2,cells=metas,
                protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sha(ROOT/'experiment.py'))

def main():
    config=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results_main'
    old=out/'cell_5000_0.2'
    if old.exists():old.rename(out/'excluded_partial_attempt_5000_0.2')
    start=time.perf_counter();tasks=[]
    for b in range(4):
        c=copy.deepcopy(config);c['n']=[5000];c['gamma']=[.2]
        c['main_seeds']={'start':10000+25*b,'count':25}
        tasks.append((c,out/f'partition_{b}'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(worker,item) for item in tasks]):print('FINISHED PARTITION '+f.result(),flush=True)
    cellmeta=merge([path for _,path in tasks],old)
    cellmeta.update(mode='main',seconds=time.perf_counter()-start,repartitioned=True)
    write_json(old/'metadata.json',cellmeta)
    overall=merge([out/f'cell_{n}_{g}' for n in config['n'] for g in config['gamma']],out)
    rec=json.loads((out/'execution_repartition.json').read_text())
    overall.update(mode='main',wall_seconds=time.time()-rec['original_start_unix'],workers=4,
                   note='Wall time includes the excluded partial attempt before repartitioning. All 400 final logs counted once.')
    write_json(out/'metadata.json',overall);print(json.dumps(overall,indent=2),flush=True)

if __name__=='__main__':main()
