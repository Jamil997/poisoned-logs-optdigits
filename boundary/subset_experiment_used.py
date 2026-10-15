"""All six balanced one-per-family libraries; preserve the full-library result."""
import os
for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[var]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,copy,csv,json,time
import numpy as np
from experiment import ROOT,e2,write_json,write_csv,sha

def prepare():
    protocol=json.loads((ROOT/'subset_protocol.json').read_text())
    original=json.loads((ROOT/'protocol.json').read_text())
    data=dict(np.load(ROOT/'frozen_models/models.npz'))
    meta=json.loads((ROOT/'frozen_models/models.json').read_text())
    for item in protocol['libraries']:
        directory=ROOT/'subsets'/item['name'];directory.mkdir(parents=True,exist_ok=True)
        if (directory/'frozen_models').exists():raise FileExistsError(directory)
        config=copy.deepcopy(original)
        config.update(stage='E3b sibling-removal diagnostic: '+item['name'],
            parent_protocol_sha256=sha(ROOT/'protocol.json'),
            subset_protocol_sha256=sha(ROOT/'subset_protocol.json'),
            retained_candidate_indices=item['candidate_indices'],
            confidence=original['confidence'].replace('six candidates','three candidates'),
            scope='Exploratory all-balanced-subset diagnostic, not the E3 primary result.')
        write_json(directory/'protocol.json',config)
        ix=[0]+[j+1 for j in item['candidate_indices']]
        arr=dict(data);arr['p']=data['p'][ix];arr['pv']=data['pv'][ix]
        (directory/'frozen_models').mkdir()
        np.savez_compressed(directory/'frozen_models/models.npz',**arr)
        local=dict(meta)
        local.update(names=[meta['names'][j] for j in ix],
                     finite_pool_values=[meta['finite_pool_values'][j] for j in ix],
                     advantages=[meta['advantages'][j] for j in item['candidate_indices']],
                     protocol_sha256=sha(directory/'protocol.json'),
                     parent_models_sha256=sha(ROOT/'frozen_models/models.npz'))
        write_json(directory/'frozen_models/models.json',local)

def worker(args):
    directory,config,path,mode=args
    e2.ROOT=directory
    e2.run(config,path,mode)
    return str(path)

def execute(mode):
    protocol=json.loads((ROOT/'subset_protocol.json').read_text());alltasks=[];start=time.perf_counter()
    bylibrary={};key='timing_seeds' if mode=='timing' else 'main_seeds'
    for item in protocol['libraries']:
        directory=ROOT/'subsets'/item['name'];config=json.loads((directory/'protocol.json').read_text())
        out=directory/f'results_{mode}';out.mkdir(exist_ok=True);tasks=[]
        chunk=25 if mode=='main' else config[key]['count']
        for n in config['n']:
            for g in config['gamma']:
                for offset in range(0,config[key]['count'],chunk):
                    cfg=copy.deepcopy(config);cfg['n']=[n];cfg['gamma']=[g]
                    cfg[key]=dict(start=config[key]['start']+offset,count=min(chunk,config[key]['count']-offset))
                    path=out/f'shard_{n}_{g}_{offset}'
                    if path.exists():raise FileExistsError(path)
                    tasks.append((directory,cfg,path,mode))
        bylibrary[item['name']]=(directory,out,tasks);alltasks.extend(tasks)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker,task) for task in alltasks]):
            print('FINISHED '+future.result(),flush=True)
    for name,(directory,out,tasks) in bylibrary.items():
        rows=[];witnesses=[];logs={};metadata=[];tuning={}
        for _,_,path,_ in tasks:
            with open(path/'runs.csv') as f:part=list(csv.DictReader(f))
            shift=len(rows);rows.extend(part)
            for w in json.loads((path/'witnesses.json').read_text()):w['row']+=shift;witnesses.append(w)
            newlogs=dict(np.load(path/'logs.npz'));assert not(set(logs)&set(newlogs));logs.update(newlogs)
            metadata.append(json.loads((path/'metadata.json').read_text()))
            tuning.update(json.loads((path/'shrinkage_tuning.json').read_text()))
        write_csv(out/'runs.csv',rows);write_json(out/'witnesses.json',witnesses)
        np.savez_compressed(out/'logs.npz',**logs);write_json(out/'shrinkage_tuning.json',tuning)
        write_json(out/'metadata.json',dict(mode=mode,library=name,outcomes=len(rows),independent_logs=len(logs)//2,
            worker_seconds=sum(m['seconds'] for m in metadata),shards=metadata,
            protocol_sha256=sha(directory/'protocol.json'),e2_core_sha256=sha(e2.__file__),
            models_sha256=sha(directory/'frozen_models/models.npz'),source_sha256=sha(__file__)))
        (out/'frozen_protocol.json').write_bytes((directory/'protocol.json').read_bytes())
    write_json(ROOT/f'subset_{mode}_execution.json',dict(wall_seconds=time.perf_counter()-start,
        libraries=list(bylibrary),total_decisions=sum(len(tasks)*chunk*117 for _,_,tasks in bylibrary.values()),
        unique_logs_across_libraries=400 if mode=='main' else 8,workers=4,
        protocol_sha256=sha(ROOT/'subset_protocol.json'),source_sha256=sha(__file__)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','timing','main']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else execute(args.mode)
