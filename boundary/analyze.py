"""Reconstruct decisions and certificates; report all cells and paired effects."""
import argparse, importlib.util, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, build_specs, e2, write_json, sha

spec=importlib.util.spec_from_file_location('e2_analysis',ROOT.parent/'security/analyze.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def audit(out):
    result=previous.audit(out)
    config=json.loads((ROOT/'protocol.json').read_text())
    df=pd.read_csv(out/'runs.csv'); witnesses=json.loads((out/'witnesses.json').read_text())
    data=dict(np.load(ROOT/'frozen_models/models.npz')); logs=np.load(out/'logs.npz')
    byrow={w['row']:w for w in witnesses}; cache={}; checks=0
    key=['n','gamma','seed','epsilon','method','condition']
    assert not df.duplicated(key).any()
    seed_spec=config['main_seeds' if out.name=='results_main' else 'timing_seeds']
    assert len(df)==4*seed_spec['count']*3*13*3
    assert set(df.seed)==set(range(seed_spec['start'],seed_spec['start']+seed_spec['count']))
    for (n,g,seed),part in df.groupby(['n','gamma','seed'],sort=False):
        if (n,g) not in cache: cache[n,g]=build_specs(data,n,g,config)
        mu,specs,_=cache[n,g]
        prefix=f'{n}_{g}_{seed}';x=logs[prefix+'_x'].astype(int);a=logs[prefix+'_a'].astype(int)
        regenerated=e2.sample_log(data,mu,n,g,seed)
        np.testing.assert_array_equal(x,regenerated[0]);np.testing.assert_array_equal(a,regenerated[1])
        r=regenerated[2]
        for ix,row in part[part.condition=='adaptive'].iterrows():
            if row.method=='retain':continue
            s=specs[row.method.split('_',1)[1]]
            c=s['c'][:,x,a];off=s['z0'][:,x,a].mean(axis=1)-s['rad']-s['bias']
            k=int(n*row.epsilon);kd=int(row.declared_cap)
            clean=previous.direct_scores(c,off,r,0)
            down=c*r-np.minimum(c,0);up=np.maximum(c,0)-c*r
            def largest(v,k):return np.sort(v,axis=1)[:,-min(k,n):].sum(axis=1) if k else np.zeros(len(v))
            lo=clean-largest(down,k+kd)/n
            hi=clean-largest(down,kd-k)/n if kd>=k else clean+largest(up,k-kd)/n
            w=byrow[ix];np.testing.assert_allclose(w['lower'],lo,atol=1e-11)
            np.testing.assert_allclose(w['upper'],hi,atol=1e-11)
            possible=[]
            if max(lo)<=1e-9+1e-10:possible.append(-1)
            for j,u in enumerate(hi):
                if u>1e-9-1e-10 and u>=max(1e-9,max(np.delete(lo,j)))-1e-10:possible.append(j)
            for sub in w['subproblems']:
                if sub.get('kind')=='all_real_L1_dual' and sub.get('upper_margin') is not None and sub['upper_margin']<-1e-8:
                    if sub['target'] in possible:possible.remove(sub['target'])
            assert sorted(possible)==sorted(w['possible'])
            checks+=1
    # Merged records must agree with each completed shard, not an edited subset.
    parts=[]
    for shard in sorted(out.glob('shard_*')):
        meta=json.loads((shard/'metadata.json').read_text())
        assert meta['protocol_sha256']==sha(ROOT/'protocol.json')
        assert meta['models_sha256']==sha(ROOT/'frozen_models/models.npz')
        assert meta['source_sha256']==sha(e2.__file__)
        parts.append(pd.read_csv(shard/'runs.csv'))
    merged=pd.concat(parts).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(merged,df.sort_values(key).reset_index(drop=True))
    result.update(envelope_and_possible_set_checks=checks,random_logs_regenerated=True,
                  complete_frozen_grid=True,shards_match_merged=True)
    write_json(out/'audit.json',result)
    return result

def summarize(out):
    summary,agg=previous.summarize(out)
    df=pd.read_csv(out/'runs.csv');ad=df[df.condition=='adaptive'];rows=[]
    seeds=sorted(ad.seed.unique());rng=np.random.default_rng(841)
    ix=rng.integers(len(seeds),size=(10000,len(seeds)))
    for base in ['ips_point','dr_point','ips_h','centered_dr_b','dr_b','dros_b']:
        nominal=df[df.method=='nominal_'+base]
        corrected=df[df.method=='corrected_'+base]
        keys=['n','gamma','epsilon','seed','condition']
        pair=corrected.merge(nominal,on=keys,suffixes=('_c','_n'),validate='one_to_one')
        result={'base':base}
        for condition in ['adaptive','clean']:
            part=pair[pair.condition==condition].copy()
            for metric in ['harm','improvement','fallback']:
                part['difference']=part[metric+'_c']-part[metric+'_n']
                cluster=part.groupby('seed').difference.mean().reindex(seeds).to_numpy()
                lo,hi=np.quantile(cluster[ix].mean(axis=1),[.025,.975])
                result[f'{condition}_{metric}_difference_pp']=float(100*cluster.mean())
                result[f'{condition}_{metric}_low_pp']=float(100*lo)
                result[f'{condition}_{metric}_high_pp']=float(100*hi)
        rows.append(result)
    pd.DataFrame(rows).to_csv(out/'paired_safety_utility.csv',index=False)
    return summary,agg

def figures(out,summary,agg):
    selected=['nominal_dr_point','nominal_centered_dr_b','corrected_centered_dr_b','nominal_dr_b','corrected_dr_b']
    labels=['DR point estimate','Centered DR','Corrected centered DR','Fitted DR','Corrected fitted DR']
    colors=['#9467bd','#d67b30','#087f8c','#7189ad','#164b72']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
    for col,(n,g) in enumerate([(5000,.2),(5000,.5),(20000,.2),(20000,.5)]):
        for m,l,c in zip(selected,labels,colors):
            p=summary[(summary.n==n)&(summary.gamma==g)&(summary.method==m)&(summary.condition=='adaptive')].sort_values('epsilon')
            axes[0,col].plot(p.epsilon*100,p.harm_rate*100,'o-',color=c,label=l,markersize=4)
            axes[1,col].plot(p.epsilon*100,p.improvement_pp,'o-',color=c,markersize=4)
            axes[1,col].fill_between(p.epsilon*100,p.certified_floor_pp,p.improvement_pp,color=c,alpha=.12)
        axes[0,col].set_title(f'n={n:,}; exploration={g:.0%}')
        axes[1,col].set_xlabel('Editable rewards (%)');axes[1,col].axhline(0,color='#888',lw=.7)
        for row in range(2):axes[row,col].grid(alpha=.15)
    axes[0,0].set_ylabel('Harmful deployments (%)');axes[1,0].set_ylabel('True improvement (pp)')
    handles,labs=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Validation-designed OptDigits policies under adaptive reward attacks\nShading below utility: unresolved gap to the all-real-edit certificate',fontsize=14)
    fig.tight_layout(rect=[0,.09,1,.91]);fig.savefig(out/'boundary_results.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for _,r in agg.iterrows():
        color='#087f8c' if r.method.startswith('corrected') else '#a75c32'
        ax.scatter(r.max_cell_harm*100,r.mean_improvement_pp,c=color,s=50)
        ax.annotate(r.method.replace('centered_dr_b','centered').replace('ips_point','IPS point').replace('dr_point','DR point'),
            (r.max_cell_harm*100,r.mean_improvement_pp),xytext=(5,4),textcoords='offset points',fontsize=7)
    ax.set_xlabel('Maximum observed harmful-deployment rate across 12 cells (%)')
    ax.set_ylabel('Mean true improvement across 12 cells (pp)')
    ax.set_title('Finite observed safety–utility comparison\nThese points do not establish global Pareto optimality')
    ax.axhline(0,color='#888',lw=.7);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(out/'frontier.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['main','timing'],default='main')
    args=parser.parse_args();out=ROOT/f'results_{args.mode}'
    print(json.dumps(audit(out),indent=2));summary,agg=summarize(out);figures(out,summary,agg)
    print(agg.round(5).to_string(index=False))
