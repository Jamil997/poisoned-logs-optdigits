"""Run the four pre-specified cells in independent worker processes."""
import os
for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[var]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse,copy,csv,json,time
import numpy as np
from experiment import ROOT,run,write_csv,write_json,sha

def worker(args):
    config,out,mode,limit=args
    run(config,out,mode,limit)
    return str(out)

def main(mode,limit,workers):
    config=json.loads((ROOT/'protocol.json').read_text());out=ROOT/f'results_{mode}'
    out.mkdir(exist_ok=True);start=time.perf_counter();tasks=[]
    for n in config['n']:
      for g in config['gamma']:
        cfg=copy.deepcopy(config);cfg['n']=[n];cfg['gamma']=[g]
        tasks.append((cfg,out/f'cell_{n}_{g}',mode,limit))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            print('FINISHED '+future.result(),flush=True)
    rows=[];witnesses=[];logs={};metadata=[];tuning={}
    for _,cell,_,_ in tasks:
        with open(cell/'runs.csv') as f:part=list(csv.DictReader(f))
        shift=len(rows);rows.extend(part)
        for w in json.loads((cell/'witnesses.json').read_text()):
            w['row']+=shift;witnesses.append(w)
        logs.update(dict(np.load(cell/'logs.npz')))
        metadata.append(json.loads((cell/'metadata.json').read_text()))
        tuning.update(json.loads((cell/'shrinkage_tuning.json').read_text()))
    write_csv(out/'runs.csv',rows);write_json(out/'witnesses.json',witnesses)
    np.savez_compressed(out/'logs.npz',**logs);write_json(out/'shrinkage_tuning.json',tuning)
    total=dict(mode=mode,wall_seconds=time.perf_counter()-start,worker_seconds=sum(m['seconds'] for m in metadata),
               outcomes=len(rows),independent_logs=len(logs)//2,workers=workers,cells=metadata,
               protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sha(ROOT/'experiment.py'))
    write_json(out/'metadata.json',total)
    (out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
    print(json.dumps(total,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['timing','main'],default='main')
    p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(a.mode,a.limit,a.workers)
