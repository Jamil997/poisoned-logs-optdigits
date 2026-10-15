"""Audit the saved trials independently and regenerate the result figure."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/clean_10"))
    parser.add_argument("--figure", type=Path, default=Path("clean_results.png"))
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    dataset = np.load(args.results / "dataset_split.npz")
    names = ["IW-trained", "SN-trained", "label-oracle"]
    values_by_method = {m: [] for m in summary["methods"]}
    n_checks = 0
    for seed in range(summary["n_trials"]):
        z = np.load(args.results / f"trial_{seed:02d}.npz")
        rec = json.loads((args.results / f"trial_{seed:02d}.json").read_text())
        actions, rewards = z["actions"], z["rewards"]
        weights = z["target_probs"][:, np.arange(len(actions)), actions] / z["behavior_probs"][np.arange(len(actions)), actions]
        iw = (weights * rewards).mean(axis=1)
        sn = (weights * rewards).sum(axis=1) / weights.sum(axis=1)
        for method, score in [("IW", iw), ("SNIW", sn)]:
            np.testing.assert_allclose(score, [rec["methods"][method]["scores"][c] for c in names], rtol=1e-12)
            n_checks += 1
        tp = z["target_test_probs"]
        exact = (.1 + .8 * tp[:, np.arange(len(dataset["labels_test"])), dataset["labels_test"]]).mean(axis=1)
        np.testing.assert_allclose(exact, z["expected_values"], atol=1e-12)
        n_checks += 1
        for m, row in rec["methods"].items():
            scores = [row["scores"][c] for c in names]
            winner = sorted(range(3), key=lambda j: scores[j])[-1]
            assert row["positive_score"] == (scores[winner] > 0)
            if row["positive_score"]:
                assert row["selected_candidate"] == names[winner]
                np.testing.assert_allclose(row["selected_expected_value"], exact[winner], atol=1e-12)
                np.testing.assert_allclose(row["simple_regret"], exact.max() - exact[winner], atol=1e-12)
                values_by_method[m].append(exact[winner])
            else:
                assert row["selected_candidate"] is None and row["selected_expected_value"] is None
            n_checks += 1
    for m, vals in values_by_method.items():
        assert len(vals) == summary["methods"][m]["positive_trials"]
        if vals:
            np.testing.assert_allclose(np.mean(vals), summary["methods"][m]["mean_expected_reward_conditional_on_selection"], atol=1e-12)

    order = ["ESLB", "Emp. Bernstein for λ-IW", "IW", "SNIW"]
    labels = ["ESLB", "λ-IW", "IW / IPS", "SN / SNIPS"]
    means = [np.mean(values_by_method[m]) for m in order]
    sds = [np.std(values_by_method[m], ddof=1) for m in order]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.8, 5.1), layout="constrained")
    ax.bar(labels, means, yerr=sds, capsize=5, width=.57,
           color=["#087f8c", "#5685a5", "#5685a5", "#91a5b2"], zorder=3)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Expected reward of selected policy")
    ax.set_title("OptDigits: clean policy selection", loc="left", weight="bold", pad=18)
    ax.axhline(.8996732540138862, ls="--", color="#57636d", lw=1)
    ax.text(3.49, .925, "Ideal reference: 0.8997", ha="right", fontsize=10)
    for j, mean in enumerate(means):
        ax.text(j, mean/2, f"{mean:.3f}", ha="center", va="center", color="white", weight="bold", fontsize=13)
    ax.grid(axis="y", alpha=.18, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("10 log seeds; one fixed data split. Bars: mean ± trial SD.\nChebyshev: no positive bound in 10/10 trials (omitted, not assigned zero reward).", fontsize=9)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    audit = {"passed": True, "trial_level_checks": n_checks, "trials": summary["n_trials"],
             "independent_recomputations": ["IW scores", "SN scores", "finite-pool expected values", "argmax and no-positive decisions", "simple regret", "summary means"],
             "limitations": "Numerical consistency audit, not a confidence-coverage or robustness proof.",
             "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.results / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
