"""Freeze validation-designed mixtures before reading the evaluation split.

Run `python prepare.py freeze`, then `python prepare.py evaluate`. The first
command opens only optdigits.tra; it cannot inspect evaluation rewards.
"""
from pathlib import Path
import argparse, datetime, hashlib, io, json, zipfile
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'security' / 'optdigits.zip'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def probabilities(arrays, x, name):
    return softmax(x @ arrays['coef_' + name].T + arrays['intercept_' + name], axis=1)

def interpolate(low, high, baseline, target):
    assert high > low
    raw = (baseline + target - low) / (high - low)
    return float(np.clip(raw, 0, 1)), not (0 <= raw <= 1)

def freeze():
    out = ROOT / 'selection'
    if out.exists():
        raise FileExistsError('Preserve frozen selection; use a separate checkout to reproduce.')
    config = json.loads((ROOT / 'protocol.json').read_text())
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read('optdigits.tra')  # Deliberately never read optdigits.tes.
    train = np.loadtxt(io.BytesIO(raw), delimiter=',')
    assert train.shape == (3823, 65)
    y = train[:, -1].astype(int)
    fit, val = train_test_split(np.arange(len(y)), test_size=config['validation_fraction'],
                               random_state=config['training_split_seed'], stratify=y)
    scaler = StandardScaler().fit(train[fit, :-1])
    xf = scaler.transform(train[fit, :-1]); xv = scaler.transform(train[val, :-1])
    arr = dict(fit_indices=fit, validation_indices=val, scaler_mean=scaler.mean_,
               scaler_scale=scaler.scale_, yv=y[val])
    definitions = {f'full_C{C:g}': (C, np.arange(64)) for C in [.01, .1, 1., 10.]}
    definitions.update(top_C1=(1., np.arange(32)), bottom_C1=(1., np.arange(32, 64)))
    preds = {}; fit_info = {}
    for name, (C, mask) in definitions.items():
        model = LogisticRegression(C=C, max_iter=5000, tol=1e-10, solver='lbfgs')
        model.fit(xf[:, mask], y[fit]); assert max(model.n_iter_) < 5000
        coef = np.zeros((10, 64)); coef[:, mask] = model.coef_
        arr['coef_' + name] = coef; arr['intercept_' + name] = model.intercept_
        preds[name] = probabilities(arr, xv, name)
        fit_info[name] = dict(C=C, feature_indices=mask.tolist(),
                              iterations=int(max(model.n_iter_)))
    yy = y[val]; rows = np.arange(len(yy))
    values = {name: float(p[rows, yy].mean()) for name, p in preds.items()}
    losses = {name: float(log_loss(yy, preds[name])) for name in preds if name.startswith('full_')}
    reward_name = min(losses, key=lambda key: (losses[key], definitions[key][0]))
    incumbent = 'full_C0.1'; strong = 'full_C10'
    names = ['incumbent_C0.1']; mixtures = []; policies = [preds[incumbent]]
    for weak in ['full_C0.01', 'top_C1', 'bottom_C1']:
        for target in config['policy_construction']['target_validation_advantages']:
            weight, clipped = interpolate(values[weak], values[strong], values[incumbent], target)
            p = (1-weight)*preds[weak] + weight*preds[strong]
            name = f'{weak}_target_{target:+.3f}'
            names.append(name); policies.append(p)
            mixtures.append(dict(name=name, weak=weak, strong=strong, strong_weight=weight,
                target_validation_advantage=target, clipped=clipped,
                actual_validation_advantage=float(p[rows, yy].mean()-values[incumbent]),
                validation_mean_total_variation=float(.5*np.abs(p-preds[incumbent]).sum(axis=1).mean())))
    arr['pv'] = np.array(policies); arr['qv'] = preds[reward_name]
    out.mkdir(); np.savez_compressed(out / 'selection.npz', **arr)
    meta = dict(stage='FROZEN BEFORE E3 TEST EVALUATION',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        train_member_sha256=hashlib.sha256(raw).hexdigest(),
        protocol_sha256=sha(ROOT / 'protocol.json'), construction_source_sha256=sha(__file__),
        selection_sha256=sha(out / 'selection.npz'), training_rows=len(fit), validation_rows=len(val),
        incumbent=incumbent, names=names, definitions=fit_info, validation_parent_values=values,
        reward_model=reward_name, validation_log_losses=losses, mixtures=mixtures,
        history='E2 results on this benchmark were already inspected. This is a frozen follow-up stress test, not an untouched external validation dataset.')
    save_json(out / 'selection.json', meta)
    print(json.dumps(meta, indent=2))

def evaluate():
    selection = ROOT / 'selection'; out = ROOT / 'frozen_models'
    if out.exists():
        raise FileExistsError('Do not overwrite evaluated frozen models.')
    meta = json.loads((selection / 'selection.json').read_text())
    assert meta['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert meta['construction_source_sha256'] == sha(__file__)
    assert meta['selection_sha256'] == sha(selection / 'selection.npz')
    arr = dict(np.load(selection / 'selection.npz'))
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read('optdigits.tes')
    test = np.loadtxt(io.BytesIO(raw), delimiter=','); assert test.shape == (1797, 65)
    x = (test[:, :-1]-arr['scaler_mean']) / arr['scaler_scale']
    parents = {name: probabilities(arr, x, name) for name in meta['definitions']}
    policies = [parents[meta['incumbent']]]
    for item in meta['mixtures']:
        w = item['strong_weight']
        policies.append((1-w)*parents[item['weak']] + w*parents[item['strong']])
    arr.update(p=np.array(policies), q=parents[meta['reward_model']],
               y=test[:, -1].astype(int), features=test[:, :-1])
    values = arr['p'][:, np.arange(len(test)), arr['y']].mean(axis=1)
    out.mkdir(); np.savez_compressed(out / 'models.npz', **arr)
    result = dict(names=meta['names'], finite_pool_values=values.tolist(),
        advantages=(values[1:]-values[0]).tolist(), training_rows=meta['training_rows'],
        validation_rows=meta['validation_rows'], evaluation_pool=len(test),
        validation_advantages=[item['actual_validation_advantage'] for item in meta['mixtures']],
        test_mean_total_variation=(.5*np.abs(arr['p'][1:]-arr['p'][0]).sum(axis=2).mean(axis=1)).tolist(),
        reward_model=meta['reward_model'], protocol_sha256=sha(ROOT/'protocol.json'),
        selection_metadata_sha256=sha(selection/'selection.json'),
        selection_sha256=sha(selection/'selection.npz'), test_member_sha256=hashlib.sha256(raw).hexdigest(),
        evaluation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(out / 'models.json', result)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    freeze() if args.stage == 'freeze' else evaluate()
