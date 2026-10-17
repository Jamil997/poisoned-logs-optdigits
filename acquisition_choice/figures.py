"""Descriptive figures; these do not choose or retune an experimental method."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
s=pd.read_csv(OUT/'summary.csv')
ad=s.query('condition=="exact_adaptive" and alpha==0.01 and beta==0.03 and stage=="primary"')
palette={'all_trusted':'#2d5b9a','all_main':'#667583','fixed_weight':'#9362aa','fixed_influence':'#bd812c','joint_choice':'#087f77'}
names={'all_trusted':'Trusted only','all_main':'Main only','fixed_weight':'Fixed acquisition + weighting','fixed_influence':'Influence allocation','joint_choice':'Acquisition choice'}

fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for row,account in enumerate(['total_purchase','historical_sunk']):
    for col,threat in enumerate(['fixed_200','rate_1pct']):
        q=ad.query('method=="joint_choice" and g==0.05 and accounting==@account and threat==@threat')
        budgets=sorted(q.budget.unique());prices=sorted(q.rho.unique());arr=np.zeros((len(budgets),len(prices)))
        mapping={'main_only':0,'mixed':1,'trusted_only':2}
        ax=axes[row,col]
        for i,C in enumerate(budgets):
            for j,rho in enumerate(prices):
                r=q.query('budget==@C and rho==@rho').iloc[0]
                arr[i,j]=mapping[r.source_choice]
        ax.imshow(arr,cmap=ListedColormap(['#dbe1e5','#bde5df','#c6d6ed']),vmin=0,vmax=2,aspect='auto')
        for i,C in enumerate(budgets):
            for j,rho in enumerate(prices):
                r=q.query('budget==@C and rho==@rho').iloc[0]
                ax.text(j,i,{'main_only':'Main','mixed':'Mixed','trusted_only':'Trusted'}[r.source_choice],ha='center',va='center',fontsize=11)
        ax.set_xticks(range(len(prices)),[f'{x}x' for x in prices]);ax.set_yticks(range(len(budgets)),[f'{x:,}' for x in budgets])
        ax.set_xlabel('Trusted / main observation price');ax.set_ylabel('Available acquisition budget')
        ax.set_title(('Purchase both sources' if row==0 else 'Historical log already paid for')+'\n'+('Up to 200 reward edits' if col==0 else 'Up to 1% of main rewards edited'))
fig.suptitle('The acquisition rule can choose trusted-only, mixed, or main-only evidence\nChoices use costs, sample counts and the threat bound; no evaluation rewards',fontsize=15)
fig.savefig(OUT/'source_choices.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for row,account in enumerate(['total_purchase','historical_sunk']):
    for col,rho in enumerate([5,50]):
        q=ad.query('g==0.05 and accounting==@account and threat=="fixed_200" and rho==@rho')
        ax=axes[row,col]
        for method in ['all_trusted','all_main','fixed_influence','fixed_weight','joint_choice']:
            d=q.query('method==@method').sort_values('budget')
            ax.plot(d.budget/1000,d.utility_pp,label=names[method],color=palette[method],
                    marker='o' if method!='joint_choice' else 's',markersize=5,
                    linestyle='--' if method=='fixed_weight' else '-',linewidth=2.5 if method=='joint_choice' else 1.6)
        ax.axhline(.225,color='#b0b0b0',linestyle=':',label='75% useful-gain target' if row==0 and col==0 else None)
        ax.set_ylim(-.015,.32);ax.set_xlabel('Available budget (thousands of cost units)');ax.set_ylabel('Worst-attack improvement (pp)')
        ax.set_title(('Purchase both sources' if row==0 else 'Reuse fixed historical log')+f'; trusted price {rho}x')
        ax.grid(alpha=.18)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Acquisition cost changes which strategy is useful\nLow-influence good policy; 200-edit cap; common 1% analytic harm ceiling',fontsize=15)
fig.savefig(OUT/'cost_utility.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
methods=['all_main','all_trusted','fixed_pool','fixed_influence','pilot_select','pilot_weight','joint_choice']
labels=['Main\nonly','Trusted\nonly','Equal\npooling','Influence\nallocation','Pilot\nselection','Pilot error\nallocation','Source\nweighting']
for ax,g in zip(axes,[.05,.45]):
    q=ad.query('accounting=="historical_sunk" and budget==10000 and rho==5 and threat=="fixed_200" and g==@g').set_index('method').loc[methods]
    xx=np.arange(len(q));yy=q.utility_pp.to_numpy();error=np.vstack([yy-q.utility_lower_pp.to_numpy(),q.utility_upper_pp.to_numpy()-yy])
    colors=['#9aa8b2','#5d82b2','#abb7c7','#ca9b55','#b2a0c1','#b2a0c1','#087f77']
    ax.bar(xx,yy,color=colors);ax.errorbar(xx,yy,yerr=error,fmt='none',ecolor='#263e4f',capsize=3)
    ax.set_xticks(xx,labels,fontsize=9);ax.set_ylabel('Worst-attack improvement (percentage points)')
    ax.set_title('Good policy has '+('low' if g==.05 else 'high')+' influence')
    ax.set_ylim(0,max(q.utility_upper_pp)*1.22)
    for i,v in enumerate(yy):ax.text(i,float(q.utility_upper_pp.iloc[i])+max(yy)*.035,f'{v:.3f}',ha='center',fontsize=10)
    ax.grid(axis='y',alpha=.15)
fig.suptitle('A small protected sample improves decisions from a fixed historical log\n19,000 existing rows + 2,000 trusted rows; 200 reward edits; 2,000 fresh seeds',fontsize=15)
fig.text(.5,-.035,'Error bars: simultaneous 95% utility bounds over the full experiment. Source weighting and acquisition choice coincide in this cell.\nAll rules have a 1% analytic harm ceiling; observed harm varies. This is a synthetic mechanism study.',ha='center',fontsize=10)
fig.savefig(OUT/'fixed_log_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('Wrote source_choices.png, cost_utility.png, fixed_log_comparison.png')
