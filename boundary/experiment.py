"""E3 runner reuses the published E2 estimators and attack without modification."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e2_core', ROOT.parent/'security'/'experiment.py')
e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2)
e2.ROOT = ROOT
build_specs = e2.build_specs
write_json, write_csv, sha = e2.write_json, e2.write_csv, e2.sha
score, choose, utility, attack = e2.score, e2.choose, e2.utility, e2.attack
sample_log = e2.sample_log

def run(config, out, mode='main', limit=None):
    assert (ROOT/'frozen_models/models.npz').exists(), 'Run prepare.py freeze, then evaluate first.'
    e2.run(config, out, mode, limit)
