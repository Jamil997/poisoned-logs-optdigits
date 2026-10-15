"""E2: independent fitted policies, matched reward-only attacks, audited bounds.

All policy/reward fitting precedes evaluation. Test labels belong only to the
environment/evaluator. Reward-affine gates allow exact top-k corrections and
deterministic envelopes valid for arbitrary real-valued reward replacements.
"""
from pathlib import Path
import argparse, csv, hashlib, io, json, platform, time, urllib.request, zipfile
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TOL = 1e-9

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def prepare(config, out):
    out.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'optdigits.zip'
    if not archive.exists():
        urllib.request.urlretrieve(config['data_url'], archive)
    with zipfile.ZipFile(archive) as z:
        train = np.loadtxt(io.BytesIO(z.read('optdigits.tra')), delimiter=',')
        test = np.loadtxt(io.BytesIO(z.read('optdigits.tes')), delimiter=',')
        hashes = {f: hashlib.sha256(z.read(f)).hexdigest() for f in ['optdigits.tra','optdigits.tes']}
    assert train.shape == (3823,65) and test.shape == (1797,65)
    y = train[:,-1].astype(int); yt = test[:,-1].astype(int)
    fit, val = train_test_split(np.arange(len(y)), test_size=config['validation_fraction'],
                              random_state=config['training_split_seed'], stratify=y)
    scaler = StandardScaler().fit(train[fit,:-1])
    xf = scaler.transform(train[fit,:-1]); xv = scaler.transform(train[val,:-1])
    xt = scaler.transform(test[:,:-1])
    models = {}
    for C in sorted(set(config['policy_C'] + config['reward_model_C_grid'])):
        model = LogisticRegression(C=C, max_iter=5000, tol=1e-10, solver='lbfgs')
        model.fit(xf, y[fit]); assert np.max(model.n_iter_) < 5000
        models[C] = model
    losses = {C: float(log_loss(y[val], models[C].predict_proba(xv))) for C in config['reward_model_C_grid']}
    best_C = min(losses, key=lambda c:(losses[c],c))
    def probabilities(x):
        ps = [models[c].predict_proba(x) for c in config['policy_C']]
        ps.extend((1-e)*ps[0]+e/10 for e in config['uniform_mixtures'])
        return np.array(ps)
    p = probabilities(xt); pv = probabilities(xv)
    q = models[best_C].predict_proba(xt); qv = models[best_C].predict_proba(xv)
    names = ['incumbent_C0.1','C0.01','C1','C10','uniform_0.1','uniform_0.25','uniform_0.5']
    arrays = dict(p=p, pv=pv, q=q, qv=qv, y=yt, yv=y[val], fit_indices=fit, validation_indices=val,
                  scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, features=test[:,:-1])
    for C, model in models.items():
        arrays[f'coef_{C}'] = model.coef_; arrays[f'intercept_{C}'] = model.intercept_
    np.savez_compressed(out/'models.npz', **arrays)
    # These values are evaluator diagnostics, not arguments to build_specs.
    values = np.mean(p[:,np.arange(len(yt)),yt],axis=1)
    meta = dict(dataset_sha256=hashes, archive_sha256=sha(archive), training_rows=len(fit),
                validation_rows=len(val), evaluation_pool=len(yt), names=names,
                validation_log_losses=losses, reward_model_C=best_C,
                finite_pool_values=values.tolist(), advantages=(values[1:]-values[0]).tolist(),
                policy_parameter_counts={str(C): int(m.coef_.size+m.intercept_.size) for C,m in models.items()},
                protocol_sha256=sha(ROOT/'protocol.json'))
    write_json(out/'models.json',meta)
    return arrays, meta

def shrunk(w, lam):
    return w if np.isinf(lam) else lam*w/(w*w+lam)

def tune_shrinkage(pv, qv, yv, n, gamma, config):
    mu = (1-gamma)*pv[0]+gamma/10
    delta = pv[1:]-pv[0]; b = np.sum(delta*qv,axis=2)
    r = np.eye(10)[yv]; truth = np.mean(np.sum(delta*r,axis=2),axis=1)
    choices = [np.inf if x == 'infinity' else x for x in config['shrinkage_lambda_grid']]
    errors = []
    for lam in choices:
        c = shrunk(pv[1:]/mu,lam)-shrunk(pv[0]/mu,lam)
        z = b[:,:,None]+c*(r-qv)
        mean = np.mean(np.sum(mu*z,axis=2),axis=1)
        var = np.mean(np.sum(mu*z*z,axis=2),axis=1)-mean**2
        errors.append(np.maximum(0,var)/n+(mean-truth)**2)
    ix = np.argmin(errors,axis=0)
    return np.array(choices)[ix], np.array(errors)

def build_specs(data, n, gamma, config):
    """No evaluation labels are accessed here. Bounds use the known X pool."""
    p, q = data['p'],data['q']; mu = (1-gamma)*p[0]+gamma/10
    delta = p[1:]-p[0]; c = delta/mu
    M = len(delta); conf = np.log(M/config['alpha'])
    lams, mse = tune_shrinkage(data['pv'],data['qv'],data['yv'],n,gamma,config)
    specs = {}
    for name in config['base_methods']:
        bias = np.zeros(M); cj = c.copy(); qj = q
        if name.startswith('ips'): qj = np.zeros_like(q)
        elif name.startswith('centered'): qj = np.full_like(q,.5)
        b = np.sum(delta*qj,axis=2)
        if name == 'dros_b':
            cj = np.array([shrunk(p[j+1]/mu,lam)-shrunk(p[0]/mu,lam) for j,lam in enumerate(lams)])
            d = mu*cj-delta
            bias = np.maximum(-d*q, d*(1-q)).sum(axis=2).mean(axis=1)
        z0 = b[:,:,None]-cj*qj; z1 = z0+cj
        B = np.maximum(z0,z1).max(axis=(1,2))-np.minimum(z0,z1).min(axis=(1,2))
        V = (mu*np.maximum(z0*z0,z1*z1)).sum(axis=2).mean(axis=1)
        if name.endswith('_point'): rad = np.zeros(M)
        elif name == 'ips_h': rad = B*np.sqrt(conf/(2*n))
        else:
            a = B*conf/(3*n); rad = a+np.sqrt(a*a+2*V*conf/n)
        specs[name] = dict(c=cj, z0=z0, rad=rad, bias=bias, B=B, V=V)
    return mu, specs, dict(lambdas=['infinity' if np.isinf(x) else float(x) for x in lams],validation_mse=mse.tolist())

def sample_log(data, mu, n, gamma, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed,n,int(gamma*1000)]))
    x = rng.integers(len(mu),size=n)
    actions = (rng.random(n)[:,None] > np.cumsum(mu[x],axis=1)).sum(axis=1)
    actions = np.minimum(actions,9)
    r = (actions == data['y'][x]).astype(float)
    return x,actions,r

def top_sum(a, k):
    if k <= 0: return np.zeros(a.shape[:-1])
    k = min(k,a.shape[-1])
    return np.partition(a,a.shape[-1]-k,axis=-1)[...,-k:].sum(axis=-1)

def score(gate, r, kd):
    c, off = gate['c'],gate['off']
    result = np.sum(c*r,axis=1)/len(r)+off
    if kd:
        result -= top_sum(c*r-np.minimum(c,0),kd)/len(r)
    return result

def choose(scores):
    j = int(np.argmax(scores)); return j if scores[j] > TOL else -1

def utility(j, truth):
    return 0. if j < 0 else float(truth[j])

def envelopes(gate,r,ka,kd):
    c = gate['c']; clean = score(gate,r,0); n=len(r)
    down = c*r-np.minimum(c,0); up = np.maximum(c,0)-c*r
    lower = clean-top_sum(down,ka+kd)/n
    if kd >= ka: upper = clean-top_sum(down,kd-ka)/n
    else: upper = clean+top_sum(up,ka-kd)/n
    return lower,upper

def possible_outcomes(lower,upper):
    possible=[]
    if max(lower) <= TOL+1e-10: possible.append(-1)
    for j,u in enumerate(upper):
        if u > TOL-1e-10 and u >= max(TOL,np.max(np.delete(lower,j)))-1e-10: possible.append(j)
    assert possible
    return possible

def active_branch(gate,r,z,kd,j):
    """An exact active branch of the binary-reward corrected score, in flips z.

    For binary rewards top-k damage = min_lam k*lam + sum_i avail_i*(|c_i|-lam)+.
    Thus the corrected score is a maximum of affine functions of binary flips.
    This identity is NOT used as a continuous-relaxation certificate for fractional rewards.
    """
    c=gate['c'][j]; n=len(r); sign=1-2*r
    if kd==0: return float(score(gate,r,0)[j]),c*sign/n
    rp=r+sign*z; available=(c*rp-np.minimum(c,0))
    lam=float(np.partition(available,n-min(kd,n))[n-min(kd,n)])
    clipped=np.clip(c,-lam,lam)
    base=float(np.sum(clipped*r-np.minimum(c,0)+np.maximum(np.abs(c)-lam,0)*(c<0))/n+gate['off'][j]-kd*lam/n)
    # Less opaque equivalent evaluation of intercept at the clean binary vector.
    avail_clean=np.where(c>=0,r,1-r)
    base=float(np.sum(c*r)/n+gate['off'][j]-
               (kd*lam+np.sum(avail_clean*np.maximum(np.abs(c)-lam,0)))/n)
    return base,clipped*sign/n

def margin(scores,target):
    return -float(max(scores)) if target<0 else float(scores[target]-max(TOL,max(np.delete(scores,target))))

def affine_relaxation(gate,r,ka,target):
    """L1 relaxation of the L0 real-reward attack, with a checked dual bound.

    A fractional replacement on a binary clean reward is a flip of size z in
    [0,1]. L0<=k implies sum(z)<=k. Infeasibility of this larger set excludes
    the target for every allowed real-valued attack, not just binary flips.
    """
    n=len(r); base=score(gate,r,0); g=gate['c']*(1-2*r)/n
    tb=0. if target<0 else base[target];tg=np.zeros(n) if target<0 else g[target]
    mat=[np.r_[np.ones(n),0.]];rhs=[float(ka)]
    if target>=0:mat.append(np.r_[-tg,1.]);rhs.append(tb)
    for j in range(len(base)):
        if j!=target:mat.append(np.r_[g[j]-tg,1.]);rhs.append(tb-base[j])
    A=np.array(mat);b=np.array(rhs)
    result=linprog(np.r_[np.zeros(n),-1.],A_ub=A,b_ub=b,bounds=[(0,1)]*n+[(None,None)],method='highs')
    upper=None
    if result.status==0:
        lam=np.maximum(0,-result.ineqlin.marginals)
        lam/=float(lam@A[:,-1])
        upper=float(lam@b+np.maximum(0,-lam@A[:,:-1]).sum())
        result['certificate_multipliers']=lam.tolist()
    return result,upper

def attack(gate,r,ka,kd,truth,config):
    """Feasible adaptive witnesses plus independent, all-real-edit certificates."""
    start=time.perf_counter(); n=len(r); clean=score(gate,r,kd)
    best_z=np.zeros(n,dtype=np.int8); best_s=clean; best_sel=choose(clean)
    lower,upper=envelopes(gate,r,ka,kd)
    possible=possible_outcomes(lower,upper); floor=min(utility(j,truth) for j in possible)
    calls=0; unresolved=[]; subproblems=[]
    def accept(z):
        nonlocal best_z,best_s,best_sel
        z=np.asarray(z,dtype=np.int8)
        assert np.sum(z)<=ka and np.all((z==0)|(z==1))
        ss=score(gate,r+(1-2*r)*z,kd); jj=choose(ss)
        if utility(jj,truth)<utility(best_sel,truth)-1e-12:
            best_z=z.copy();best_s=ss;best_sel=jj
        return ss,jj
    if ka and utility(best_sel,truth)>floor+1e-12:
        # Endpoint attacks on target-versus-rival margins, inspected at four budgets.
        # Each is evaluated by the full, actual corrected gate.
        sign=1-2*r
        for target in sorted(possible,key=lambda j:utility(j,truth)):
            if utility(target,truth)>=utility(best_sel,truth)-1e-12: continue
            tc=np.zeros(n) if target<0 else gate['c'][target]
            rivals=[-1]+[j for j in range(len(truth)) if j!=target]
            for other in rivals:
                oc=np.zeros(n) if other<0 else gate['c'][other]
                gains=(tc-oc)*sign
                ids=np.argsort(-gains,kind='stable')[:ka]; ids=ids[gains[ids]>0]
                for frac in [.25,.5,.75,1.]:
                    z=np.zeros(n,dtype=np.int8);z[ids[:int(np.ceil(len(ids)*frac))]]=1;accept(z)
            if utility(best_sel,truth)<=floor+1e-12: break
        if kd==0:
            for target in sorted(possible.copy(),key=lambda j:utility(j,truth)):
                if utility(target,truth)>=utility(best_sel,truth)-1e-12: continue
                lp,ub=affine_relaxation(gate,r,ka,target)
                subproblems.append(dict(target=target,status=int(lp.status),upper_margin=ub,kind='all_real_L1_dual',dual=lp.get('certificate_multipliers')))
                if ub is not None and ub < -1e-8:
                    possible.remove(target)
                elif lp.x is not None:
                    ids=np.argsort(-lp.x[:n])[:ka];z=np.zeros(n,dtype=np.int8)
                    z[ids[lp.x[ids]>.5]]=1;accept(z)
                floor=min(utility(j,truth) for j in possible)
        # Targeted active-branch mixed-integer search. Exact for affine nominal
        # targets; for corrected targets, failed branch searches remain unresolved.
        for target in sorted(possible,key=lambda j:utility(j,truth)):
            if utility(target,truth)>=utility(best_sel,truth)-1e-12: continue
            target_reached=False; certified_binary_infeasible=False
            restart_vectors=[np.zeros(n,dtype=np.int8),best_z.copy()]
            if target>=0:
                gains=gate['c'][target]*(1-2*r); zz=np.zeros(n,dtype=np.int8)
                ids=np.argsort(-gains)[:ka];zz[ids[gains[ids]>0]]=1;restart_vectors.append(zz)
            for restart in restart_vectors[:config['attack']['target_restarts']]:
                ta,tv=(0.,np.zeros(n)) if target<0 else active_branch(gate,r,restart,kd,target)
                competitors=[j for j in range(len(truth)) if j!=target]
                cuts={j:[active_branch(gate,r,restart,kd,j)] for j in competitors}
                for iteration in range(config['attack']['cutting_plane_iterations']):
                    mat=[np.r_[np.ones(n),0.]]; rhs=[ka]
                    if target>=0: mat.append(np.r_[-tv,1.]);rhs.append(ta-TOL)
                    for j in competitors:
                        for ca,cv in cuts[j]: mat.append(np.r_[cv-tv,1.]);rhs.append(ta-ca)
                    objective=np.r_[np.zeros(n),-1.]
                    result=milp(objective,integrality=np.r_[np.ones(n),0.],
                        bounds=Bounds(np.r_[np.zeros(n),-np.inf],np.r_[np.ones(n),np.inf]),
                        constraints=LinearConstraint(np.array(mat),-np.inf,np.array(rhs)),
                        options={'time_limit':config['attack']['milp_time_limit_seconds'],'mip_rel_gap':0.})
                    calls+=1
                    dual=getattr(result,'mip_dual_bound',None)
                    ub=None if dual is None or not np.isfinite(dual) else -float(dual)
                    subproblems.append(dict(target=target,status=int(result.status),upper_margin=ub))
                    if result.x is None:
                        unresolved.append(target);break
                    z=np.rint(result.x[:n]).astype(np.int8)
                    if z.sum()>ka: unresolved.append(target);break
                    ss,jj=accept(z)
                    if jj==target: target_reached=True;break
                    if (kd==0 or target<0) and ub is not None and ub < -1e-8:
                        certified_binary_infeasible=True;break
                    # Corrected competitor cuts are valid on every binary vector.
                    for j in competitors:
                        ab=active_branch(gate,r,z,kd,j)
                        if all(abs(ab[0]-old[0])>1e-12 or np.max(abs(ab[1]-old[1]))>1e-12 for old in cuts[j]):
                            cuts[j].append(ab)
                    if kd==0: break
                if target_reached or certified_binary_infeasible: break
            if not target_reached and not certified_binary_infeasible: unresolved.append(target)
            if utility(best_sel,truth)<=floor+1e-12: break
    gap=max(0.,utility(best_sel,truth)-floor)
    # Every certificate applies to all real reward replacements, independent of
    # solver success. A zero gap proves optimal worst utility, not max margin.
    assert np.all(best_s>=lower-1e-8) and np.all(best_s<=upper+1e-8)
    return dict(selected=best_sel,scores=best_s,indices=np.flatnonzero(best_z),
                lower=lower,upper=upper,floor=floor,gap=gap,
                possible=possible,calls=calls,unresolved=sorted(set(unresolved)),
                subproblems=subproblems,seconds=time.perf_counter()-start)

def run(config,out,mode='main',limit=None):
    out.mkdir(parents=True,exist_ok=True)
    (out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
    model_dir=ROOT/'frozen_models'
    if not (model_dir/'models.npz').exists(): prepare(config,model_dir)
    data=dict(np.load(model_dir/'models.npz')); meta=json.loads((model_dir/'models.json').read_text())
    truth=np.array(meta['advantages']); rows=[]; witnesses=[];start=time.perf_counter()
    seed_spec=config['timing_seeds' if mode=='timing' else 'main_seeds']
    count=limit or seed_spec['count'];logs={};tuning={}
    for n in config['n']:
      for gamma in config['gamma']:
        mu,specs,tuned=build_specs(data,n,gamma,config);tuning[f'{n}_{gamma}']=tuned
        for seed in range(seed_spec['start'],seed_spec['start']+count):
            x,a,r=sample_log(data,mu,n,gamma,seed)
            logs[f'{n}_{gamma}_{seed}_x']=x.astype(np.int16)
            logs[f'{n}_{gamma}_{seed}_a']=a.astype(np.uint8)
            gates={name:dict(c=s['c'][:,x,a],off=s['z0'][:,x,a].mean(axis=1)-s['rad']-s['bias']) for name,s in specs.items()}
            loghash=hashlib.sha256(x.astype('<i4').tobytes()+a.astype('uint8').tobytes()+r.astype('uint8').tobytes()).hexdigest()
            for eps in config['epsilon']:
                ka=int(n*eps)
                rng=np.random.default_rng(np.random.SeedSequence([seed,n,int(gamma*1000),ka,51]))
                rand_ids=np.sort(rng.choice(n,ka,replace=False));rr=r.copy();rr[rand_ids]=1-rr[rand_ids]
                for name,gate in gates.items():
                  for protect in config['protection']:
                    kd=ka if protect=='corrected' else 0
                    method=f'{protect}_{name}'
                    attack_result=attack(gate,r,ka,kd,truth,config)
                    for condition in config['conditions']:
                        ids=np.array([],dtype=int) if condition=='clean' else rand_ids if condition=='random' else attack_result['indices']
                        rp=r.copy();rp[ids]=1-rp[ids]
                        ss=score(gate,rp,kd);selected=choose(ss);u=utility(selected,truth)
                        row=dict(n=n,gamma=gamma,epsilon=eps,seed=seed,method=method,condition=condition,
                            selected=selected,improvement=u,harm=int(u<-1e-12),fallback=int(selected<0),
                            useful=int(u>1e-12),edits=len(ids),declared_cap=kd,log_sha256=loghash,
                            utility_floor=attack_result['floor'],attack_utility_gap=attack_result['gap'],
                            harm_excluded=int(all(utility(j,truth)>=-1e-12 for j in attack_result['possible'])),
                            solver_calls=attack_result['calls'] if condition=='adaptive' else 0,
                            attack_seconds=attack_result['seconds'] if condition=='adaptive' else 0.)
                        rows.append(row)
                        witnesses.append(dict(row=len(rows)-1,indices=ids.tolist(),scores=ss.tolist(),
                            lower=attack_result['lower'].tolist(),upper=attack_result['upper'].tolist(),
                            possible=attack_result['possible'],subproblems=attack_result['subproblems'] if condition=='adaptive' else []))
                for condition in config['conditions']:
                    rows.append(dict(n=n,gamma=gamma,epsilon=eps,seed=seed,method='retain',condition=condition,selected=-1,
                        improvement=0.,harm=0,fallback=1,useful=0,edits=0,declared_cap=ka,log_sha256=loghash,
                        utility_floor=0.,attack_utility_gap=0.,harm_excluded=1,solver_calls=0,attack_seconds=0.))
            if (seed-seed_spec['start']+1)%10==0 or count<=2:
                write_csv(out/'runs.csv',rows)
                print(f'{mode} n={n} gamma={gamma} completed={seed-seed_spec["start"]+1}/{count} outcomes={len(rows)} seconds={time.perf_counter()-start:.1f}',flush=True)
    write_csv(out/'runs.csv',rows);write_json(out/'witnesses.json',witnesses)
    np.savez_compressed(out/'logs.npz',**logs);write_json(out/'shrinkage_tuning.json',tuning)
    metadata=dict(mode=mode,seconds=time.perf_counter()-start,outcomes=len(rows),independent_logs=len(config['n'])*len(config['gamma'])*count,
        protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sha(__file__),models_sha256=sha(model_dir/'models.npz'),
        python=platform.python_version(),numpy=np.__version__,solver_calls=sum(row['solver_calls'] for row in rows),
        all_real_attack_unresolved=sum(row['condition']=='adaptive' and row['attack_utility_gap']>1e-12 for row in rows))
    write_json(out/'metadata.json',metadata);print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','timing','main'],default='main')
    parser.add_argument('--out',type=Path);parser.add_argument('--limit',type=int)
    args=parser.parse_args();config=json.loads((ROOT/'protocol.json').read_text())
    if args.mode=='prepare':
        _,meta=prepare(config,ROOT/'frozen_models');print(json.dumps(meta,indent=2))
    else: run(config,args.out or ROOT/f'results_{args.mode}',args.mode,args.limit)
