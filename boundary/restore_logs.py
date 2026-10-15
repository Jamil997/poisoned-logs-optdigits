"""Restore identical shared logs omitted from archives to avoid duplication."""
from pathlib import Path
import json,shutil
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    for mode in ['main','timing']:
        source=ROOT/f'results_{mode}/logs.npz'
        if not source.exists():continue
        shared=dict(np.load(source))
        directories=[ROOT]+list((ROOT/'subsets').iterdir())
        for directory in directories:
            out=directory/f'results_{mode}'
            if not out.exists():continue
            target=out/'logs.npz'
            if not target.exists():shutil.copy2(source,target)
            for shard in out.glob('shard_*'):
                target=shard/'logs.npz'
                if target.exists():continue
                _,n,g,offset=shard.name.split('_')
                config=json.loads((directory/'protocol.json').read_text())
                seeds=config['main_seeds' if mode=='main' else 'timing_seeds']
                start=seeds['start']+int(offset)
                count=json.loads((shard/'metadata.json').read_text())['independent_logs']
                keys=[f'{n}_{g}_{seed}_{field}' for seed in range(start,start+count) for field in ['x','a']]
                np.savez_compressed(target,**{key:shared[key] for key in keys})
    print('Shared evaluation logs restored without resampling.')

if __name__=='__main__':main()
