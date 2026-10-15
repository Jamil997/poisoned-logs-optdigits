"""Evaluator-only variance diagnostics; never used to fit or choose a gate."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT,build_specs

def main():
    config=json.loads((ROOT/'protocol.json').read_text())
    data=dict(np.load(ROOT/'frozen_models/models.npz'));r=np.eye(10)[data['y']];rows=[]
    for n in config['n']:
      for gamma in config['gamma']:
        mu,specs,tuning=build_specs(data,n,gamma,config)
        for method,s in specs.items():
            z=s['z0']+s['c']*r;mean=(mu*z).sum(2).mean(1)
            variance=(mu*z*z).sum(2).mean(1)-mean**2
            for j in range(6):
                rows.append(dict(n=n,gamma=gamma,method=method,candidate=j,true_variance=variance[j],
                    second_moment_bound=s['V'][j],radius_pp=100*s['rad'][j],bias_allowance_pp=100*s['bias'][j],
                    lambda_selected=tuning['lambdas'][j] if method=='dros_b' else 'not_applicable'))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'moment_diagnostics.csv',index=False)
    print(df[(df.n==20000)&(df.gamma==.2)&(df.candidate==2)].to_string(index=False))

if __name__=='__main__':main()
