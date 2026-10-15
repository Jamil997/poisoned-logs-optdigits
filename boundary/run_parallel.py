"""Four processes, small fixed shards; no changes based on attack outcomes."""
import os
for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[var] = '1'
import argparse, copy, csv, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from experiment import ROOT, run, write_csv, write_json, sha

def worker(args):
    config, path, mode = args
    run(config, path, mode)
    return str(path)

def main(mode, workers):
    config = json.loads((ROOT/'protocol.json').read_text()); out = ROOT/f'results_{mode}'
    out.mkdir(exist_ok=True); tasks = []; start = time.perf_counter()
    seed_key = 'main_seeds' if mode == 'main' else 'timing_seeds'
    chunk = 25 if mode == 'main' else config[seed_key]['count']
    for n in config['n']:
        for gamma in config['gamma']:
            for offset in range(0, config[seed_key]['count'], chunk):
                cfg = copy.deepcopy(config); cfg['n'] = [n]; cfg['gamma'] = [gamma]
                cfg[seed_key] = dict(start=config[seed_key]['start']+offset,
                    count=min(chunk, config[seed_key]['count']-offset))
                path = out/f'shard_{n}_{gamma}_{offset}'
                if path.exists():
                    raise FileExistsError(f'Preserve existing shard: {path}')
                tasks.append((cfg, path, mode))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(worker, task) for task in tasks]):
            print('FINISHED ' + f.result(), flush=True)
    rows = []; witnesses = []; logs = {}; metadata = []; tuning = {}
    for _, path, _ in tasks:
        with open(path/'runs.csv') as f: part = list(csv.DictReader(f))
        shift = len(rows); rows.extend(part)
        for w in json.loads((path/'witnesses.json').read_text()):
            w['row'] += shift; witnesses.append(w)
        newlogs = dict(np.load(path/'logs.npz')); assert not (set(newlogs) & set(logs))
        logs.update(newlogs); metadata.append(json.loads((path/'metadata.json').read_text()))
        tuning.update(json.loads((path/'shrinkage_tuning.json').read_text()))
    write_csv(out/'runs.csv', rows); write_json(out/'witnesses.json', witnesses)
    np.savez_compressed(out/'logs.npz', **logs); write_json(out/'shrinkage_tuning.json', tuning)
    total = dict(mode=mode, wall_seconds=time.perf_counter()-start,
        worker_seconds=sum(m['seconds'] for m in metadata), outcomes=len(rows),
        independent_logs=len(logs)//2, workers=workers, shards=metadata,
        protocol_sha256=sha(ROOT/'protocol.json'), e2_core_sha256=sha(ROOT.parent/'security/experiment.py'),
        runner_sha256=sha(__file__), models_sha256=sha(ROOT/'frozen_models/models.npz'))
    write_json(out/'metadata.json', total)
    (out/'frozen_protocol.json').write_bytes((ROOT/'protocol.json').read_bytes())
    print(json.dumps(total, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['timing','main'], default='main')
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    main(args.mode, args.workers)
