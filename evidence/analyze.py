"""Prespecified E8 comparisons, matched-risk diagnostic, uncertainty and cost."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from method import weights_for,margins,select,tail,damage
from run import FAV,extreme_choices,frame
from audit import critical
ROOT=Path(__file__).resolve().parent

KEYS=['beta','good_amplitude','bad_amplitude','trusted_size','epsilon','method','condition']
def summary(d,keys):
 s=d.groupby(keys,dropna=False).agg(n=('improvement','size'),utility=('improvement','mean'),harm=('harm','mean'),fallback=('fallback','mean'),harm_count=('harm','sum')).reset_index()
 s['utility_pp']=100*s.utility;s['harm_upper_95']=beta_dist.ppf(.95,s.harm_count+1,s.n-s.harm_count);return s

def contrast(a,b):
 aa=a.set_index('seed').improvement;bb=b.set_index('seed').improvement
 assert aa.index.is_unique and bb.index.is_unique
 x=(aa-bb).dropna().to_numpy();assert len(x)==2000
 values,counts=np.unique(x,return_counts=True);rng=np.random.default_rng(98511)
 bs=rng.multinomial(len(x),counts/len(x),size=10000)@values/len(x)
 return dict(difference_pp=100*float(x.mean()),paired_bootstrap_95_pp=(100*np.quantile(bs,[.025,.975])).tolist(),seeds=len(x))

def main():
 summaries=[];primary=[];curves=[]
 for d in pd.read_csv(ROOT/'results/runs.csv.gz',chunksize=108000):
  summaries.append(summary(d,KEYS));primary.append(d.query('beta==.03 and epsilon==.01 and trusted_size==1000 and condition=="adaptive"'))
  curves.append(d.query('beta==.03 and epsilon==.01 and condition=="adaptive"'))
 s=pd.concat(summaries,ignore_index=True);assert not s.duplicated(KEYS).any();s.to_csv(ROOT/'summary.csv',index=False)
 p=pd.concat(primary,ignore_index=True);p.to_csv(ROOT/'primary_runs.csv.gz',index=False)
 curve=pd.concat(curves,ignore_index=True)
 c=pd.read_csv(ROOT/'results/cost_runs.csv.gz');summary(c,KEYS+['rho','actual_trusted_size','acquisition_cost']).to_csv(ROOT/'cost_summary.csv',index=False)
 stress=pd.read_csv(ROOT/'results/stress_runs.csv.gz');summary(stress,KEYS+['allowed_trusted_edits']).to_csv(ROOT/'trust_stress_summary.csv',index=False)
 contrasts={};costcontrasts={}
 for g in [.05,.45]:
  sub=p[p.good_amplitude==g];a=sub[sub.method=='evidence_weighted']
  contrasts[str(g)]={method:contrast(a,sub[sub.method==method]) for method in sub.method.unique() if method!='evidence_weighted'}
  cs=c.query('beta==.03 and trusted_size==1000 and good_amplitude==@g')
  costcontrasts[str(g)]={str(rho):contrast(a,cs[cs.rho==rho]) for rho in [5,10,50]}
 # Oracle matched-risk diagnostic. Weights are fixed from the deployed designs;
 # only the test threshold changes using the evaluator's p_bad=.499.
 logs=np.load(ROOT/'results/logs.npz');U=logs['main_0.03_19000'];V=logs['trusted_0.03_1000'];vt=(V@FAV.T).T;seeds=np.arange(90000,92000)
 matched=[];analytical=[]
 for g,b in [[.05,.45],[.45,.05]]:
  B=2*np.array([g,b])
  for meth in ['pooled_equal','main_only','trusted_only','evidence_weighted']:
   rule=weights_for(meth,19000,1000,200,.01,B,vt);a0,c0,d,h,corr=rule
   for mode in ['nominal_alpha','matched_actual_risk']:
    if mode=='matched_actual_risk':
     hh=critical(19000,1000,a0,c0,.001,.499);h=np.full_like(h,hh)
     # An independently summed binomial tail must agree with the FFT threshold.
     assert tail(hh,19000,1000,a0,c0,.499)+1e-12<=.001
     assert tail(hh-1,19000,1000,a0,c0,.499)+1e-12>.001
     rule=(a0,c0,d,h,corr)
    selected,option,ss,ku,kt=extreme_choices(U,V,B,rule,200)
    if mode=='matched_actual_risk':
     df=frame(selected,ss,seeds,.03,g,b,1000,.01,meth,mode,option,np.where(option==2,0,ku),0);matched.append(df)
    shift=damage(a0,c0,200)[0]
    harm=tail(int(h[1,0])+corr-shift,19000,1000,a0,c0,.499)
    good=tail(int(h[0,0])+corr+shift,19000,1000,a0,c0,.53)
    floor=2*g*.03*max(0,good-harm)-2*b*.001*harm
    analytical.append(dict(good_amplitude=g,method=meth,calibration=mode,harm_upper=harm,good_positive_under_all_attacks=good,utility_floor_pp=100*floor,threshold=int(h[0,0]),actual_null_tail=tail(int(h[0,0]),19000,1000,a0,c0,.5)))
 md=pd.concat(matched,ignore_index=True);md.to_csv(ROOT/'matched_risk_runs.csv.gz',index=False);ms=summary(md,KEYS);ms.to_csv(ROOT/'matched_risk_summary.csv',index=False)
 pd.DataFrame(analytical).to_csv(ROOT/'analytical_bounds.csv',index=False)
 matched_contrasts={}
 for g in [.05,.45]:
  sub=md[md.good_amplitude==g];aa=sub[sub.method=='evidence_weighted']
  matched_contrasts[str(g)]={meth:contrast(aa,sub[sub.method==meth]) for meth in ['pooled_equal','main_only','trusted_only']}
 report=dict(primary_contrasts=contrasts,cost_matched_contrasts=costcontrasts,matched_risk_contrasts=matched_contrasts,bootstrap='10000 paired seed resamples using multinomial frequencies of paired differences; pointwise intervals, no simultaneous claim')
 (ROOT/'inference.json').write_text(json.dumps(report,indent=2)+'\n')
 # Three panels: both policy geometries and the cost falsification.
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors={'evidence_weighted':'#087f82','pooled_equal':'#617586','trusted_only':'#b86b26','pooled_influence':'#8d66a3'}
 labels={'evidence_weighted':'Weighted evidence','pooled_equal':'Ordinary pooling','trusted_only':'Trusted data only','pooled_influence':'Influence allocation'}
 fig,axes=plt.subplots(1,3,figsize=(15,4.6))
 for ax,g,title in [(axes[0],.05,'Good policy has lower influence'),(axes[1],.45,'Good policy has higher influence')]:
  for meth in colors:
   sub=s.query('beta==.03 and epsilon==.01 and good_amplitude==@g and condition=="adaptive" and method==@meth').sort_values('trusted_size')
   ax.plot(sub.trusted_size,sub.utility_pp,'o-',color=colors[meth],label=labels[meth],markersize=4)
  ax.axhline(0,color='#999999',lw=.7);ax.set(xlabel='Trusted observations within N = 20,000',ylabel='Worst-attack mean improvement (pp)',title=title)
  ax.set_xticks([0,1000,2000,5000]);ax.grid(alpha=.15)
 ax=axes[2];g=.05
 base=float(s.query('beta==.03 and epsilon==.01 and good_amplitude==.05 and condition=="adaptive" and method=="evidence_weighted" and trusted_size==1000').utility_pp.iloc[0])
 cs=c.query('beta==.03 and trusted_size==1000 and good_amplitude==.05').groupby('rho').improvement.mean()*100
 ax.plot([5,10,50],[base]*3,'o-',color=colors['evidence_weighted'],label='19,000 main + 1,000 trusted')
 ax.plot(cs.index,cs.values,'s-',color=colors['trusted_only'],label='All trusted, same total cost')
 ax.set(xlabel='Cost of one trusted observation / main',ylabel='Mean improvement (pp)',title='Independent evidence has a price',xscale='log');ax.set_xticks([5,10,50],labels=['5×','10×','50×']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.15);ax.legend(loc='lower right',fontsize=9)
 handles,ls=axes[0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',ncol=4,bbox_to_anchor=(.43,-.005));fig.suptitle('E8: fixed 200 reward edits; 2,000 independent seed clusters',fontweight='bold')
 fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(ROOT/'evidence_tradeoff.png',dpi=180);plt.close(fig)
 print(summary(p,['good_amplitude','method'])[['good_amplitude','method','utility_pp','harm','fallback']].to_string(index=False));print('MATCHED RISK\n',ms[['good_amplitude','method','utility_pp','harm']].to_string(index=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
