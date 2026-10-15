"""E4 confirmation: prospective corruption-aware evidence design on OptDigits.

Never modify the E2 attack implementation. Defender design/bounds use only p.
Evaluation labels are restricted to the simulated environment and diagnostics.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib,gzip,platform,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_develop import core,ROOT,PARENT,designs,spec,sample
from develop import loggers

def all_designs(p):
    out=designs(p);N=p.shape[1]
    out['action_only']=loggers(p)['mix_1']/N
    out['uniform']=np.full_like(p[0],1/(N*10))
    # Minimizes mean_j E_Q c_j^2, without a reward model or evaluation labels.
    delta=(p[1:]-p[0])/N;w=np.sqrt(np.mean(delta*delta,axis=0));base=(.8*p[0]+.02)/N
    out['joint_rms']=.99*w/w.sum()+.01*base
    return out

def loghash(x,a,r):
    return hashlib.sha256(x.astype('<i4').tobytes()+a.astype('uint8').tobytes()+r.astype('uint8').tobytes()).hexdigest()

def work(task):
    name,n,start,count,mode=task;cfg=json.loads((ROOT/'confirmation_protocol.json').read_text())
    out=ROOT/f'results_{mode}'/f'{name}_{n}_{start}';out.mkdir(parents=True,exist_ok=False)
    data=dict(np.load(PARENT/'frozen_models/models.npz'));p=data['p'];y=data['y'];Q=all_designs(p)[name]
    libs=json.loads((PARENT/'subset_protocol.json').read_text())['libraries'];rows=[];witnesses=[]
    truths=(p[1:,np.arange(len(y)),y]-p[0,np.arange(len(y)),y]).mean(axis=1)
    specs={lib['name']:spec(p[[0]+[i+1 for i in lib['candidate_indices']]],Q,n,cfg['alpha']) for lib in libs}
    began=time.perf_counter()
    for seed in range(start,start+count):
      x,a,r=sample(data,Q,n,seed);lh=loghash(x,a,r)
      for lib in libs:
        inds=lib['candidate_indices'];tr=truths[inds];s=specs[lib['name']];c=s['c'][:,x,a]
        gate=dict(c=c,off=-.5*c.mean(axis=1)-s['rad'])
        for eps in cfg['epsilon']:
          ka=int(n*eps);rng=np.random.default_rng(np.random.SeedSequence([seed,n,ka,919]))
          rand=np.sort(rng.choice(n,ka,replace=False))
          for method,kd in [('nominal',0),('corrected',ka)]:
            # A proven nonnegative floor rules out harm under every real edit.
            # Endpoint attacks and nominal LPs still optimize attained utility;
            # retain every residual utility gap instead of spending MILPs on
            # distinguishing two already-certified nonnegative outcomes.
            lower,upper=core.envelopes(gate,r,ka,kd)
            possible=core.possible_outcomes(lower,upper)
            safe=all(core.utility(j,tr)>=0 for j in possible)
            attack_cfg=cfg if not safe else dict(cfg,attack=dict(cfg['attack'],target_restarts=0))
            att=core.attack(gate,r,ka,kd,tr,attack_cfg)
            for condition in ['clean','random','adaptive']:
              ids=[] if condition=='clean' else rand.tolist() if condition=='random' else att['indices'].tolist()
              rr=r.copy();rr[ids]=1-rr[ids];scores=core.score(gate,rr,kd);sel=core.choose(scores);u=core.utility(sel,tr)
              rows.append(dict(design=name,n=n,seed=seed,library=lib['name'],epsilon=eps,method=method,condition=condition,
                selected=sel,improvement=u,harm=int(u<0),fallback=int(sel<0),edits=len(ids),declared_cap=kd,log_sha256=lh,
                utility_floor=att['floor'],attack_utility_gap=att['gap'],harm_excluded=int(all(core.utility(j,tr)>=0 for j in att['possible'])),
                solver_calls=att['calls'] if condition=='adaptive' else 0,attack_seconds=att['seconds'] if condition=='adaptive' else 0))
              witnesses.append(dict(row=len(rows)-1,indices=ids,scores=scores.tolist(),lower=att['lower'].tolist(),upper=att['upper'].tolist(),possible=att['possible'],subproblems=att['subproblems'] if condition=='adaptive' else []))
          for condition in ['clean','random','adaptive']:
            rows.append(dict(design=name,n=n,seed=seed,library=lib['name'],epsilon=eps,method='retain',condition=condition,selected=-1,improvement=0.,harm=0,fallback=1,edits=0,declared_cap=ka,log_sha256=lh,utility_floor=0.,attack_utility_gap=0.,harm_excluded=1,solver_calls=0,attack_seconds=0.))
    core.write_csv(out/'runs.csv',rows)
    with gzip.open(out/'witnesses.json.gz','wt') as f:json.dump(witnesses,f,separators=(',',':'),allow_nan=False)
    meta=dict(seconds=time.perf_counter()-began,task=task,outcomes=len(rows),logs=count,source_sha256=core.sha(__file__),
      core_sha256=core.sha(core.__file__),models_sha256=core.sha(PARENT/'frozen_models/models.npz'),
      joint_source_sha256=core.sha(ROOT/'joint_develop.py'),protocol_sha256=core.sha(ROOT/'confirmation_protocol.json'))
    core.write_json(out/'metadata.json',meta)
    return meta

def run(mode):
    cfg=json.loads((ROOT/'confirmation_protocol.json').read_text());start=time.perf_counter()
    out=ROOT/f'results_{mode}';out.mkdir(exist_ok=False)
    (out/'frozen_protocol.json').write_bytes((ROOT/'confirmation_protocol.json').read_bytes())
    seeds=cfg[f'{mode}_seeds'];shard=cfg['shard_size']
    tasks=[(d,n,seeds['start']+s,min(shard,seeds['count']-s),mode) for d in cfg['designs'] for n in cfg['n'] for s in range(0,seeds['count'],shard)]
    done=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
      futures=[pool.submit(work,t) for t in tasks]
      for future in as_completed(futures):
        m=future.result();done.append(m);print(f"{len(done)}/{len(tasks)} {m['task']} {m['seconds']:.1f}s",flush=True)
    core.write_json(out/'execution.json',dict(seconds=time.perf_counter()-start,shards=done,python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    print('Completed',mode,time.perf_counter()-start,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['timing','main']);run(ap.parse_args().mode)
