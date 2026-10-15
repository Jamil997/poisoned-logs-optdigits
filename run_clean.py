"""OptDigits E0: instrument the pinned authors' clean benchmark without edits.

This is a reproduction of the available released code, not all Table 2 methods.
Diagnostic calls are deterministic; --verify checks against the direct upstream
entry point, including its original sequence of random-number draws.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SCIKIT_LEARN_DATA", str(ROOT / ".cache"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(ROOT / "vendor"))

import jax
import numpy as np
from absl import logging
from offpolicy_selection_eslb import estimators, experiment, policies

NAMES = ["IW-trained", "SN-trained", "label-oracle"]
CONFIG = {
    "experiment": "E0-clean-released-code-reproduction",
    "upstream_commit": "3c867e75ca22fd6eb01ac21e41fdc77f47b06ca8",
    "openml_id": 28, "n_trials": 10,
    "split_seed": 0, "log_fraction": 0.5,
    "standardization": "all feature rows, as in upstream; no labels",
    "delta_per_candidate": 0.05,
    "behavior_temperature": 0.2, "behavior_faulty_actions": [1, 2],
    "target_temperature": 0.1, "reward_flip_probability": 0.1,
    "gradient_steps": 10000, "gradient_step_size": 0.01,
    "eslb_iterations": 10, "eslb_mc_batch": 1000,
    "eslb_bias_type": "MultOneHot",
    "primary_metrics": ["selected policy expected reward on fixed test pool",
                        "simple regret to best candidate on fixed test pool",
                        "fraction with no positive score", "elapsed seconds"],
    "secondary_metrics": ["upstream sampled test reward", "selection counts",
                          "effective sample size", "candidate score decomposition"],
    "interpretation": "clean reproduction only; no adversarial reward edits",
    "expectations_fixed_before_run": {
        "qualitative": "ESLB should favor the ideal reference over inferior learned policies",
        "published_OptDigits_ESLB": 0.91,
        "published_OptDigits_lambda_IW": 0.72,
        "published_OptDigits_Cheb_SN": "no positive bound",
        "failure_action": "report discrepancy; do not tune to match the table",
    },
}


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def assert_probs(p):
    assert np.isfinite(p).all() and np.min(p) >= 0
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def make_estimators():
    return estimators.get_estimators(
        CONFIG["delta_per_candidate"], CONFIG["eslb_iterations"],
        CONFIG["eslb_mc_batch"], estimators.ESLBBiasType.MultOneHot)


def kwargs(n_trials, est):
    return dict(
        estimators=est, openml_id=28, n_trials=n_trials,
        behavior_policy_temperature=0.2, behavior_faulty_actions=[1, 2],
        reward_noise_p=0.1,
        target_policy_specs=[
            ("SoftmaxGAPolicy", dict(step_size=.01, steps=10000,
                temperature=.1, obj_type=policies.TrainedPolicyObjType.IW)),
            ("SoftmaxGAPolicy", dict(step_size=.01, steps=10000,
                temperature=.1, obj_type=policies.TrainedPolicyObjType.SNIW)),
            ("SoftmaxDataPolicy", dict(temperature=.1, faulty_actions=[]))])


class Recorder:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = []

    def __call__(self, **kw):
        start = time.perf_counter()
        result = self.wrapped(**kw)
        self.calls.append({"seconds": time.perf_counter() - start, **result})
        return result

    def get_name(self):
        return self.wrapped.get_name()

    def get_abbrev(self):
        return self.wrapped.get_abbrev()


def summarize(records, raw_rewards, reference, total_seconds):
    result = {}
    for m, name in enumerate(records[0]["methods"]):
        rows = [r["methods"][name] for r in records]
        values = [r["selected_expected_value"] for r in rows if r["positive_score"]]
        regrets = [r["simple_regret"] for r in rows if r["positive_score"]]
        per_trial = raw_rewards[m].mean(axis=0)
        finite = per_trial[np.isfinite(per_trial)]
        result[name] = {
            "positive_trials": len(values), "total_trials": len(records),
            "no_positive_score_fraction": 1 - len(values) / len(records),
            "mean_expected_reward_conditional_on_selection": np.mean(values) if values else None,
            "sd_across_trial_expected_rewards": np.std(values, ddof=1) if len(values) > 1 else None,
            "mean_simple_regret_conditional_on_selection": np.mean(regrets) if regrets else None,
            "selection_counts": {c: sum(r["selected_candidate"] == c for r in rows) for c in NAMES},
            "sampled_test_reward_mean_on_valid_trials": np.mean(finite) if len(finite) else None,
            "sampled_test_reward_trial_sd": np.std(finite, ddof=1) if len(finite) > 1 else None,
            "upstream_table_reward_mean": raw_rewards[m].mean(),
            "upstream_table_dispersion_not_SE": raw_rewards[m].std(axis=0).mean(),
            "mean_scoring_seconds": np.mean([r["scoring_seconds"] for r in rows]),
        }
    return {"n_trials": len(records), "total_seconds": total_seconds,
            "methods": result, "upstream_reference_sampled_mean": reference.mean(),
            "note": "Null means no qualifying selection or insufficient trials, not zero reward."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true",
                        help="Repeat the direct upstream run and require identical sampled outcomes.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    for rel, expected in json.loads((ROOT / "vendor_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel
    protocol = dict(CONFIG, n_trials=args.trials, seeds=list(range(args.trials)),
                    registered_utc=datetime.now(timezone.utc).isoformat())
    dump(args.out / "protocol.json", protocol)
    packages = {name: importlib.metadata.version(name) for name in
                ["numpy", "scipy", "scikit-learn", "jax", "jaxlib", "pandas", "absl-py"]}
    dump(args.out / "environment.json", {
        "platform": platform.platform(), "python": sys.version,
        "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "packages": packages, "jax_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": list(map(str, jax.devices())),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (args.out / "requirements-lock.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))

    logging.set_verbosity(logging.INFO)
    recorders = [Recorder(est) for est in make_estimators()]
    original_eval = estimators.evaluate_estimators
    records = []
    clock_start = time.perf_counter()
    last_end = [clock_start]

    def record_evaluation(contexts, actions, rewards, b_policy, t_policies, estimators, dataset):
        trial = len(records)
        result = original_eval(contexts, actions, rewards, b_policy, t_policies, estimators, dataset)
        test_rewards, winners, _ = result
        rng_before = np.random.get_state()
        bp = b_policy.get_probs(contexts)
        tp = np.stack([p.get_probs(contexts) for p in t_policies])
        ttest = np.stack([p.get_probs(dataset.contexts_test) for p in t_policies])
        for p in [bp, *tp, *ttest]:
            assert_probs(p)
        assert np.min(bp) > 0 and np.isin(rewards, [0, 1]).all()
        assert contexts.shape == (2810, 64) and dataset.contexts_test.shape == (2810, 64)
        expected_values = .1 + .8 * ttest[:, np.arange(dataset.n_test), dataset.labels_test].mean(axis=1)
        btest = b_policy.get_probs(dataset.contexts_test)
        behavior_value = .1 + .8 * btest[np.arange(dataset.n_test), dataset.labels_test].mean()
        record = {"trial_seed": trial, "behavior_expected_value": behavior_value,
                  "candidate_expected_values": dict(zip(NAMES, expected_values)), "methods": {}}
        for method_index, (est, winner) in enumerate(zip(estimators, winners)):
            calls = est.calls[-len(NAMES):]
            assert len(calls) == 3
            scores = [float(c["estimate"]) for c in calls]
            assert np.isfinite(scores).all()
            chosen = next(j for j, p in enumerate(t_policies) if p is winner)
            # Upstream stable sort resolves exact ties in favor of the last candidate.
            assert chosen == sorted(range(3), key=lambda j: scores[j])[-1]
            valid = scores[chosen] > 0
            record["methods"][est.get_abbrev()] = {
                "scores": dict(zip(NAMES, scores)),
                "score_details": dict(zip(NAMES, calls)),
                "argmax_candidate": NAMES[chosen], "positive_score": bool(valid),
                "selected_candidate": NAMES[chosen] if valid else None,
                "selected_expected_value": expected_values[chosen] if valid else None,
                "simple_regret": expected_values.max() - expected_values[chosen] if valid else None,
                "sampled_test_reward": test_rewards[method_index].mean() if valid else None,
                "scoring_seconds": sum(c["seconds"] for c in calls),
            }
        weights = tp[:, np.arange(len(actions)), actions] / bp[np.arange(len(actions)), actions]
        record["effective_sample_sizes"] = dict(zip(NAMES, weights.sum(axis=1)**2 / (weights**2).sum(axis=1)))
        record["logged_mean_reward"] = rewards.mean()
        record["logged_min_propensity"] = bp[np.arange(len(actions)), actions].min()
        if trial == 0:
            dump(args.out / "dataset.json", {
                "name": dataset.name, "openml_id": 28, "rows": dataset.size,
                "features": 64, "actions": 10, "train_rows": dataset.n_train,
                "test_rows": dataset.n_test,
                "array_sha256": {k: array_hash(getattr(dataset, k)) for k in
                    ["contexts_all", "labels_all", "contexts_train", "labels_train", "contexts_test", "labels_test"]}})
            np.savez_compressed(args.out / "dataset_split.npz", contexts_train=contexts,
                labels_train=dataset.labels_train, contexts_test=dataset.contexts_test,
                labels_test=dataset.labels_test)
        np.savez_compressed(args.out / f"trial_{trial:02d}.npz", actions=actions, rewards=rewards,
            behavior_probs=bp, target_probs=tp, target_test_probs=ttest,
            expected_values=expected_values,
            iw_parameters=np.asarray(t_policies[0].parameters),
            sn_parameters=np.asarray(t_policies[1].parameters))
        rng_after = np.random.get_state()
        assert rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1])
        assert rng_before[2:] == rng_after[2:]
        record["seconds_since_previous_trial"] = time.perf_counter() - last_end[0]
        last_end[0] = time.perf_counter()
        records.append(record)
        dump(args.out / f"trial_{trial:02d}.json", record)
        selections = {k: v["selected_candidate"] for k, v in record["methods"].items()}
        print(json.dumps({"trial_complete": trial + 1, "seconds": record["seconds_since_previous_trial"],
                          "values": safe(record["candidate_expected_values"]), "selected": selections}), flush=True)
        return result

    estimators.evaluate_estimators = record_evaluation
    try:
        rewards, reference, name = experiment.run_single_experiment(**kwargs(args.trials, recorders))
    finally:
        estimators.evaluate_estimators = original_eval
    elapsed = time.perf_counter() - clock_start
    np.savez_compressed(args.out / "upstream_outputs.npz", test_rewards=rewards, reference_rewards=reference)
    dump(args.out / "summary.json", summarize(records, rewards, reference, elapsed))
    with (args.out / "trial_metrics.csv").open("w", newline="") as f:
        fields = ["seed", "method", "selected_candidate", "positive_score", "selected_expected_value",
                  "simple_regret", "sampled_test_reward", "scoring_seconds"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            for method, row in r["methods"].items():
                writer.writerow(safe({"seed": r["trial_seed"], "method": method,
                                      **{k: row[k] for k in fields[2:]}}))
    print(f"Completed {args.trials} trials in {elapsed:.1f} seconds. Dataset: {name}", flush=True)
    if args.verify:
        t = time.perf_counter()
        direct, direct_ref, _ = experiment.run_single_experiment(**kwargs(args.trials, make_estimators()))
        np.testing.assert_array_equal(direct, rewards)
        np.testing.assert_array_equal(direct_ref, reference)
        dump(args.out / "instrumentation_verification.json", {
            "passed": True, "direct_upstream_outputs_bitwise_equal": True,
            "trials": args.trials, "verification_seconds": time.perf_counter() - t})
        print("PASS: instrumented outcomes exactly equal direct upstream outcomes.", flush=True)


if __name__ == "__main__":
    main()
