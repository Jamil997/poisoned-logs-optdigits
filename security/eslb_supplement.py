"""Unmodified upstream ESLB plus explicitly separate safe-improvement adapter.

The main grid uses analytic moment bounds. This supplement has ten fresh-log
seeds per cell and a pre-specified MC stability check. It is not pooled with
the main grid. Monte Carlo integration error is not certified.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[key]='1'
import abc, ast, enum, importlib.util, json, logging, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from experiment import ROOT, attack, choose, sample_log, score, sha, utility, write_csv, write_json

VENDOR=ROOT.parent/'vendor/offpolicy_selection_eslb'

def load_eslb():
    tree=ast.parse((VENDOR/'estimators.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Estimator','ESLBBiasType','ESLB']]
    spec=importlib.util.spec_from_file_location('eslb_utils',VENDOR/'utils.py')
    utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    ns=dict(np=np,abc=abc,enum=enum,math=math,logging=logging,utils=utils)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/'estimators.py'),'exec'),ns)
    return ns['ESLB'],ns['ESLBBiasType']

def gate_for(data,mu,x,actions,seed,config,batch):
    ESLB,Bias=load_eslb();n=len(x);M=len(data['p'])-1;parts=[]
    cfg=config['eslb_supplement']
    for j,pi in enumerate(data['p']):
        estimator=ESLB(delta=config['alpha']/(M+1),n_iterations=cfg['iterations'],n_batch_size=batch,bias_type=Bias.Bernstein)
        np.random.seed(seed+100*j)
        res=estimator(pi[x],mu[x],actions,np.zeros(n))
        w=pi[x,actions]/mu[x,actions]
        coef=res['mult_bias']*w/w.sum()
        const=-res['mult_bias']*(res['concentration']+1/np.sqrt(n))-res['concentration_of_contexts']
        parts.append(dict(coef=coef,const=const,mb=res['mult_bias'],concentration=res['concentration']))
    base=parts[0]
    gate=dict(c=n*np.array([p['coef']-base['coef'] for p in parts[1:]]),
              off=np.array([p['const']+base['const']+base['mb']-1 for p in parts[1:]]))
    return gate,parts

def validate_adapter(data,mu,x,actions,config):
    small_x=x[:128];small_a=actions[:128];seed=173
    gate,parts=gate_for(data,mu,small_x,small_a,seed,config,16)
    r=np.random.default_rng(9).random(len(small_x));ESLB,Bias=load_eslb();raw=[];clip=[]
    for j,pi in enumerate(data['p']):
        est=ESLB(delta=config['alpha']/7,n_iterations=config['eslb_supplement']['iterations'],n_batch_size=16,bias_type=Bias.Bernstein)
        np.random.seed(seed+100*j)
        result=est(pi[small_x],mu[small_x],small_a,1-r if j==0 else r)
        rr=1-r if j==0 else r
        raw.append(parts[j]['coef']@rr+parts[j]['const']);clip.append(result['lower_bound'])
        np.testing.assert_allclose(max(0,raw[-1]),clip[-1],atol=1e-12)
    direct=np.array(raw[1:])+raw[0]-1
    np.testing.assert_allclose(score(gate,r,0),direct,atol=1e-12)
    assert choose(direct)==choose(np.array(clip[1:])+clip[0]-1)
    return float(np.max(abs(score(gate,r,0)-direct)))

def cell(n,gamma):
    config=json.loads((ROOT/'protocol.json').read_text());cfg=config['eslb_supplement']
    data=dict(np.load(ROOT/'frozen_models/models.npz'))
    truth=np.array(json.loads((ROOT/'frozen_models/models.json').read_text())['advantages'])
    out=ROOT/'results_eslb'/f'cell_{n}_{gamma}';out.mkdir(parents=True,exist_ok=True)
    mu=(1-gamma)*data['p'][0]+gamma/10;rows=[];witnesses=[];caches={};stability=[];start=time.perf_counter()
    for offset in range(cfg['seeds']):
        seed=config['main_seeds']['start']+offset;x,a,r=sample_log(data,mu,n,gamma,seed)
        if offset==0: adapter_error=validate_adapter(data,mu,x,a,config)
        gate,parts=gate_for(data,mu,x,a,seed+700000,config,cfg['batch_size'])
        caches[f'{seed}_c']=gate['c'];caches[f'{seed}_off']=gate['off']
        gates=[('primary_mc',gate)]
        if offset<cfg['mc_stability_seeds_per_cell']:
            high,_=gate_for(data,mu,x,a,seed+700000,config,cfg['stability_batch_size'])
            gates.append(('high_mc',high));caches[f'{seed}_high_c']=high['c'];caches[f'{seed}_high_off']=high['off']
        for mc,actual_gate in gates:
          for eps in config['epsilon']:
            ka=int(n*eps);rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(gamma*1000),ka,51]))
            random_ids=np.sort(rng.choice(n,ka,replace=False))
            for protection in config['protection']:
                kd=ka if protection=='corrected' else 0
                result=attack(actual_gate,r,ka,kd,truth,config)
                for condition in config['conditions']:
                    ids=np.array([],dtype=int) if condition=='clean' else random_ids if condition=='random' else result['indices']
                    rp=r.copy();rp[ids]=1-rp[ids];ss=score(actual_gate,rp,kd);sel=choose(ss);u=utility(sel,truth)
                    row=dict(n=n,gamma=gamma,epsilon=eps,seed=seed,method=f'{protection}_sn_eslb_adapter',
                        condition=condition,mc=mc,selected=sel,improvement=u,harm=int(u<0),fallback=int(sel<0),
                        useful=int(u>0),edits=len(ids),declared_cap=kd,utility_floor=result['floor'],attack_utility_gap=result['gap'],
                        harm_excluded=int(all(utility(j,truth)>=0 for j in result['possible'])),
                        solver_calls=result['calls'] if condition=='adaptive' else 0)
                    rows.append(row);witnesses.append(dict(row=len(rows)-1,indices=ids.tolist(),scores=ss.tolist(),
                        lower=result['lower'].tolist(),upper=result['upper'].tolist(),possible=result['possible']))
        print(f'ESLB n={n} gamma={gamma} {offset+1}/{cfg["seeds"]} seconds={time.perf_counter()-start:.1f}',flush=True)
    write_csv(out/'runs.csv',rows);write_json(out/'witnesses.json',witnesses)
    np.savez_compressed(out/'gates.npz',**caches)
    meta=dict(seconds=time.perf_counter()-start,n=n,gamma=gamma,logs=cfg['seeds'],outcomes=len(rows),adapter_error=adapter_error,
        upstream_estimators_sha256=sha(VENDOR/'estimators.py'),source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'))
    write_json(out/'metadata.json',meta);return str(out)

def main():
    config=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'results_eslb';out.mkdir(exist_ok=True)
    start=time.perf_counter();cells=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(cell,n,g) for n in config['n'] for g in config['gamma']]):cells.append(f.result())
    import csv
    rows=[];metas=[]
    for path in sorted(cells):
        with open(Path(path)/'runs.csv') as f:rows.extend(csv.DictReader(f))
        metas.append(json.loads((Path(path)/'metadata.json').read_text()))
    write_csv(out/'runs.csv',rows);write_json(out/'metadata.json',dict(wall_seconds=time.perf_counter()-start,cells=metas))

if __name__=='__main__':
    from pathlib import Path
    main()
