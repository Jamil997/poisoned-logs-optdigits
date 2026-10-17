"""Fresh controlled confirmation. Prices and attack budgets are explicit scenarios."""
from pathlib import Path
import argparse,hashlib,itertools,json,time
import numpy as np
import pandas as pd
from method import design_rows,edit_cap,cost,scores,select,thresholds

ROOT=Path(__file__).resolve().parent
TYPES=np.array(list(itertools.product([-1,1],[-1,1],[0,1],[0,1])))
XI,ZE,AC,RE=TYPES.T
F=np.stack([XI*(2*AC-1)*(2*RE-1)>0,ZE*(2*AC-1)*(2*RE-1)>0]).astype(np.int64)

def probabilities(beta,theta):
    mean=.5+(2*AC-1)*(beta*XI-theta*ZE)
    return np.where(RE,mean,1-mean)/8

def sample_stream(sizes,beta,theta,seeds,stream):
    sizes=sorted(set(sizes)); out={n:np.zeros((len(seeds),16),dtype=np.int32) for n in sizes}
    for ix,seed in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(round(beta*10000)),20261017,stream]))
        counts=np.zeros(16,dtype=np.int32);last=0
        for n in sizes:
            if n>last: counts=counts+rng.multinomial(n-last,probabilities(beta,theta)).astype(np.int32)
            out[n][ix]=counts;last=n
    return out

def configurations(cfg):
    cells=[]
    for account,budgets in [('total_purchase',cfg['total_budgets']),('historical_sunk',cfg['sunk_budgets'])]:
        for C,rho in itertools.product(budgets,cfg['prices']): cells.append((account,C,rho,'primary'))
    for C,rho in cfg['legacy_cost_cells']:cells.append(('total_purchase',C,rho,'legacy_budget'))
    return cells

def rules_for(cfg,cell,threat,alpha):
    account,C,rho,stage=cell
    chosen,fixed_best,(nf,mf),grid=design_rows(cfg,C,rho,account,threat,alpha)
    rules={}
    for name in cfg['methods']:
        if name=='retain': n=m=a=c=0
        elif name=='joint_choice':n,m,a,c=[chosen[x] for x in ['n','m','a','c']]
        elif name=='fixed_weight':n,m,a,c=[fixed_best[x] for x in ['n','m','a','c']]
        elif name=='all_trusted':n,m,a,c=0,C//rho,0,1
        elif name=='all_main':n,m,a,c=(C if account=='total_purchase' else cfg['historical_log_size']),0,1,0
        else:
            n,m=nf,mf
            a,c=(1,0) if name.startswith('pilot_') else (1,1)
        rules[name]=dict(n=n,m=m,a=a,c=c,k=edit_cap(n,threat),denominator=max(1,a*n+c*m),
                         actual_cost=cost(n,m,rho,account),reference_weight=c*m/max(1,a*n+c*m))
    return rules,grid,chosen

def execute(cfg,out,timing=False):
    out.mkdir(exist_ok=False)
    (out/'frozen_protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['protocol.json','method.py','run.py','audit.py','analyze.py','test_method.py']}
    (out/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    started=time.perf_counter();designs=[];design_grid=[];frozen={}
    cells=configurations(cfg)
    if timing: cells=[('total_purchase',20000,5,'timing'),('historical_sunk',5000,50,'timing')]
    for cell,threat,alpha in itertools.product(cells,cfg['threat_models'],cfg['alphas']):
        rules,grid,chosen=rules_for(cfg,cell,threat,alpha)
        key=(cell,threat,alpha);frozen[key]=rules
        common=dict(accounting=cell[0],budget=cell[1],rho=cell[2],stage=cell[3],threat=threat,alpha=alpha)
        for name,r in rules.items():designs.append(dict(**common,method=name,**r))
        for r in grid:design_grid.append(dict(**common,chosen=(r==chosen),**r))
    pd.DataFrame(designs).to_csv(out/'designs.csv',index=False)
    pd.DataFrame(design_grid).to_csv(out/'design_grid.csv',index=False)
    print('Frozen acquisitions',len(frozen),'grid entries',len(design_grid),'seconds',round(time.perf_counter()-started,3),flush=True)
    n_sizes={0}|{r['n'] for dd in frozen.values() for r in dd.values()}
    m_sizes={0}|{r['m'] for dd in frozen.values() for r in dd.values()}
    seed_start=cfg['timing_seed_start'] if timing else cfg['seed_start']
    seed_count=cfg['timing_seed_count'] if timing else cfg['seed_count']
    seeds=np.arange(seed_start,seed_start+seed_count,dtype=np.int64)
    groups=[];allclean=[];allattacked=[];alloptions=[];logs={}; min_slack=10**9
    for beta in cfg['beta']:
        main=sample_stream(n_sizes,beta,cfg['theta'],seeds,1)
        trusted=sample_stream(m_sizes,beta,cfg['theta'],seeds,2)
        logs.update({f'u_{beta}_{n}':x for n,x in main.items()})
        logs.update({f't_{beta}_{m}':x for m,x in trusted.items()})
        for (cell,threat,alpha),rules in frozen.items():
            account,C,rho,stage=cell
            for g,b in cfg['amplitudes']:
                B=2*np.array([g,b])
                for name,r in rules.items():
                    n,m,a,c,k,d=[r[x] for x in ['n','m','a','c','k','denominator']]
                    if name=='retain':
                        clean=attacked=np.full(seed_count,-1,dtype=np.int8)
                        op=np.full(seed_count,2,dtype=np.int8)
                    else:
                        u=(main[n]@F.T).T;v=(trusted[m]@F.T).T
                        h=thresholds(n,m,a,c,alpha,B,v,name)
                        correction=a*k
                        clean=select(scores(u,v,B,a,c,d,h,correction))
                        if a==0 or k==0:
                            attacked=clean.copy();op=np.full(seed_count,2,dtype=np.int8)
                        else:
                            for bad_bit in [0,1]:
                                slack=main[n][:,(F[0]==1)&(F[1]==bad_bit)].sum(axis=1)-k
                                if np.any(slack<0):raise RuntimeError('Endpoint witness infeasible; exact optimizer required before reporting')
                                min_slack=min(min_slack,int(slack.min()))
                            candidates=[]
                            for shift in [np.array([-k,k]),np.array([-k,-k]),np.array([0,0])]:
                                candidates.append(select(scores(u+shift[:,None],v,B,a,c,d,h,correction)))
                            candidate=np.array(candidates)
                            rank=np.where(candidate==1,0,np.where(candidate<0,1,2))
                            op=rank.argmin(axis=0).astype(np.int8)
                            attacked=candidate[op,np.arange(seed_count)]
                    groups.append(dict(group_id=len(groups),accounting=account,budget=C,rho=rho,stage=stage,threat=threat,alpha=alpha,
                                       beta=beta,theta=cfg['theta'],g=g,b=b,method=name,**r))
                    allclean.append(clean);allattacked.append(attacked);alloptions.append(op)
        print('Simulated beta',beta,'groups',len(groups),'seconds',round(time.perf_counter()-started,3),flush=True)
    pd.DataFrame(groups).to_csv(out/'groups.csv',index=False)
    np.savez_compressed(out/'outcomes.npz',clean=np.array(allclean),adaptive=np.array(allattacked),attack_option=np.array(alloptions),seeds=seeds)
    np.savez_compressed(out/'logs.npz',**logs)
    meta=dict(seconds=time.perf_counter()-started,groups=len(groups),seeds=seed_count,decision_conditions=2,
              decisions=2*len(groups)*seed_count,log_arrays=len(logs),minimum_endpoint_capacity_slack=min_slack,
              stage='timing_only' if timing else 'fresh_confirmation',numpy=np.__version__)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--timing',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/'protocol.json').read_text())
    execute(cfg,ROOT/('timing' if args.timing else 'results'),args.timing)
