"""Build a code/results-only release from completed, audited experiments."""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,sha,write_json

RELEASE=ROOT.parent/'release_boundary'

def diagnostic_figure():
    frames=[]
    for directory in sorted((ROOT/'subsets').iterdir()):
        d=pd.read_csv(directory/'results_main/runs.csv');d['library']=directory.name;frames.append(d)
    df=pd.concat(frames);ad=df[df.condition=='adaptive']
    rows=[];rng=np.random.default_rng(841);draws=rng.integers(100,size=(10000,100))
    for (method,eps),part in ad.groupby(['method','epsilon']):
        row=dict(method=method,epsilon=eps,cases=len(part),
            certified_harm_upper=float(1-part.harm_excluded.mean()))
        for metric in ['harm','improvement','fallback']:
            cluster=part.groupby('seed')[metric].mean().reindex(range(20000,20100)).to_numpy()
            low,high=np.quantile(cluster[draws].mean(axis=1),[.025,.975])
            row[metric]=float(cluster.mean());row[metric+'_low']=float(low);row[metric+'_high']=float(high)
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'diagnostic_budget_intervals.csv',index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    for method,label,color in [('nominal_centered_dr_b','Unprotected centered DR','#ae4938'),
        ('corrected_centered_dr_b','Corrected centered DR','#087f8c')]:
        p=table[table.method==method].sort_values('epsilon')
        for ax,metric in zip(axes,['harm','improvement']):
            ax.plot(100*p.epsilon,100*p[metric],'o-',lw=2.3,color=color,label=label)
            ax.fill_between(100*p.epsilon,100*p[metric+'_low'],100*p[metric+'_high'],alpha=.18,color=color)
            ax.set_xlabel('Maximum editable rewards (%)');ax.grid(alpha=.15);ax.set_xticks([0,.5,1.])
    axes[0].set_ylabel('Harmful deployments (%)');axes[0].set_ylim(-3,100)
    axes[1].set_ylabel('True improvement (percentage points)');axes[1].axhline(0,color='#888',lw=.8)
    axes[0].set_title('Removing better siblings exposes harm')
    axes[1].set_title('Protection avoids harm by retaining the incumbent')
    axes[0].legend(loc='upper left',frameon=False,fontsize=10)
    fig.text(.5,.01,'Exploratory: all 6 mixed libraries and all 4 size/exploration settings. Bands: paired 95% bootstrap intervals.\n400 shared logs; 100 seed clusters. A few nominal attacks retain certified optimization gaps.',ha='center',fontsize=9,color='#555')
    fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(ROOT/'diagnostic_results.png',dpi=180);plt.close(fig)

def archive(target,paths,exclude_logs=False):
    members=[]
    for path in paths:
        members.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()])
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(members)):
            if path.name=='logs.npz' and (exclude_logs or path.parent.name.startswith('shard_')):continue
            if path.suffix=='.png':continue
            z.write(path,path.relative_to(ROOT))
        internal={name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
        z.writestr('ARCHIVE_'+target.stem+'_HASHES.json',json.dumps(internal,indent=2)+'\n')
    assert target.stat().st_size<25_000_000,(target,target.stat().st_size)

def main():
    assert json.loads((ROOT/'results_main/audit.json').read_text())['passed']
    assert json.loads((ROOT/'subset_main_audit.json').read_text())['passed']
    diagnostic_figure();dest=RELEASE/'boundary';dest.mkdir(parents=True,exist_ok=True)
    public=['README.md','METHODS.md','prepare.py','experiment.py','run_parallel.py','analyze.py',
        'test_boundary.py','subset_experiment.py','subset_experiment_used.py','analyze_subsets.py',
        'diagnostics.py','restore_logs.py','build_release.py','protocol.json','subset_protocol.json',
        'subset_main_aggregate.csv','subset_main_cells.csv','subset_main_audit.json','subset_main_execution.json',
        'sibling_summary.csv','budget_summary.csv','diagnostic_budget_intervals.csv','diagnostic_results.png']
    for name in public:shutil.copy2(ROOT/name,dest/name)
    for source,name in [('results_main/summary.csv','primary_summary.csv'),
        ('results_main/aggregate.csv','primary_aggregate.csv'),('results_main/audit.json','primary_audit.json'),
        ('results_main/paired_safety_utility.csv','primary_paired_safety_utility.csv'),
        ('results_main/metadata.json','primary_execution.json'),('frozen_models/models.json','models.json'),
        ('selection/selection.json','selection.json')]:shutil.copy2(ROOT/source,dest/name)
    archive(dest/'boundary_primary.zip',[ROOT/'selection',ROOT/'frozen_models',ROOT/'results_main',ROOT/'sibling_certificates.csv'],exclude_logs=True)
    archive(dest/'boundary_logs.zip',[ROOT/'results_main/logs.npz'])
    archive(dest/'boundary_timing.zip',[ROOT/'results_timing',ROOT/'subset_timing_audit.json',
        ROOT/'subset_timing_aggregate.csv',ROOT/'subset_timing_cells.csv',ROOT/'subset_timing_execution.json'])
    for directory in sorted((ROOT/'subsets').iterdir()):
        archive(dest/f'subset_{directory.name}.zip',[directory],exclude_logs=True)
    history=dict(primary_protocol_sha256=sha(ROOT/'protocol.json'),
        primary_candidate_selection_sha256=sha(ROOT/'selection/selection.json'),
        subset_protocol_sha256=sha(ROOT/'subset_protocol.json'),
        executed_subset_runner_sha256=sha(ROOT/'subset_experiment_used.py'),
        current_subset_runner_sha256=sha(ROOT/'subset_experiment.py'),
        postrun_change='Two descriptive lists in subset model metadata were filtered to the retained indices. Executed source is preserved. Frozen probability arrays, protocols, sampled logs, attack code, decisions and certificates did not change.',
        primary_failures='No failed or excluded main shards. An analysis command was issued before the merged main file existed; it performed no experimental computation and was rerun after completion.',
        tests='Five inherited E2 mathematical tests and three E3 construction tests passed.',
        audit_checks=327600,lp_duals_recomputed=12091,
        source_core_sha256=sha(ROOT.parent/'security/experiment.py'))
    write_json(dest/'EXECUTION_HISTORY.json',history)
    manifest=''.join(f'{sha(p)}  {p.name}\n' for p in sorted(dest.iterdir()) if p.is_file() and p.name!='BOUNDARY_MANIFEST.sha256')
    (dest/'BOUNDARY_MANIFEST.sha256').write_text(manifest)
    clone=ROOT.parent/'published'
    oldreadme=subprocess.check_output(['git','show','origin/main:README.md'],cwd=clone).decode()
    tail=oldreadme[oldreadme.index('## E0:'):]
    overview='''# OptDigits: off-policy selection and reward poisoning

This repository contains three empirical stages. **[E3: near-incumbent policies and the sibling-removal diagnostic](boundary/README.md)** is the latest follow-up: 327,600 audited decisions on 400 shared evaluation logs. It exposes harmful deployments once better sibling candidates are removed, but the confidence-based correction prevents those failures by always retaining the incumbent at the tested nonzero budgets. It does not yet demonstrate useful protected improvement under corruption.

**[E2: independently trained policies under reward poisoning](security/README.md)** evaluates the earlier, more widely separated library. That complete 400-log experiment found no harmful selections even without protection. Its source and results are preserved as a negative baseline.

**E0**, below, reproduces the authors' released clean OptDigits benchmark. The original code and result archive remain unchanged. The earlier [synthetic pilot](https://github.com/Jamil997/poisoned-logs-prior-pilot) is a separate snapshot.

---

'''
    (RELEASE/'README.md').write_text(overview+tail)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','origin/main'],cwd=clone).decode().splitlines()
    hashes={}
    for path in paths:
        if path=='MANIFEST.sha256':continue
        content=subprocess.check_output(['git','show',f'origin/main:{path}'],cwd=clone)
        hashes[path]=hashlib.sha256(content).hexdigest()
    hashes['README.md']=sha(RELEASE/'README.md')
    for path in dest.iterdir():
        if path.is_file():hashes['boundary/'+path.name]=sha(path)
    (RELEASE/'MANIFEST.sha256').write_text(''.join(f'{value}  {name}\n' for name,value in sorted(hashes.items())))
    print(json.dumps({p.name:p.stat().st_size for p in dest.glob('*.zip')},indent=2))
    print(f'Release: {dest}; {len(list(dest.iterdir()))} files')

if __name__=='__main__':main()
