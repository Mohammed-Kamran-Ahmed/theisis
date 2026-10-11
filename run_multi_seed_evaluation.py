"""
run_multi_seed_evaluation.py
=============================
Runs the FULL pipeline (data split → Stage 4 encoder training → Stage 5
extraction → Stage 6 evaluation) across several random seeds, and reports
mean ± std for the headline metrics.

This matters because a single 80/20 split (especially with only 35 test
patches) gives a noisy point estimate — you saw this directly: the same
pipeline produced probe accuracy anywhere from 65.7% to 77.1% and Top-5
retrieval from 82.9% to 88.6% across just three runs. Reporting a single
number without this context overstates precision your data can't support.

WARNING: this retrains the encoder from scratch for EACH seed (Stage 4's
40 epochs), so total runtime is roughly N_SEEDS × (one full pipeline run).
On CPU, budget accordingly — start with 3 seeds to gauge timing before
committing to 5+.

Usage:
    python run_multi_seed_evaluation.py --seeds 42 1 7
    python run_multi_seed_evaluation.py --seeds 42 1 7 123 2024   # more robust, slower
"""

import argparse
import json
import subprocess
import sys
import numpy as np


def run_step(cmd, label):
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Step failed: {label} (command: {' '.join(cmd)})")


def run_one_seed(seed):
    py = sys.executable  # use the same interpreter/venv already running this script

    run_step([py, "utils/data_split.py", "--seed", str(seed)], f"Data split (seed={seed})")
    run_step([py, "stage4_contrastive_learning.py"], f"Stage 4 encoder training (seed={seed})")
    run_step([py, "stage5_extract_and_evaluate.py"], f"Stage 5 embedding extraction (seed={seed})")
    run_step([py, "stage6_downstream_evaluation.py"], f"Stage 6 evaluation (seed={seed})")

    with open("./stage6_metrics_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary


def flatten_metrics(summary):
    """Pulls out the headline numbers into a flat dict for easy aggregation."""
    flat = {}
    for clf_name, metrics in summary["classification"].items():
        flat[f"probe_acc__{clf_name}"] = metrics["accuracy"]
    for k, v in summary["retrieval"].items():
        flat[f"retrieval__{k}"] = v
    flat["n_test"] = summary["n_test"]
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7],
                         help="List of random seeds to run the full pipeline with.")
    args = parser.parse_args()

    all_runs = []
    for seed in args.seeds:
        print(f"\n\n{'#'*70}\n# RUNNING FULL PIPELINE FOR SEED {seed}\n{'#'*70}")
        try:
            summary = run_one_seed(seed)
            flat = flatten_metrics(summary)
            flat["seed"] = seed
            all_runs.append(flat)
        except Exception as e:
            print(f"⚠️ Seed {seed} failed: {e} — skipping and continuing with remaining seeds.")

    if not all_runs:
        print("❌ No seeds completed successfully.")
        return

    # Aggregate
    all_keys = set()
    for run in all_runs:
        all_keys.update(k for k in run.keys() if k != "seed")

    print(f"\n\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY ({len(all_runs)} seeds: {[r['seed'] for r in all_runs]})")
    print(f"{'='*70}\n")

    agg_rows = []
    for key in sorted(all_keys):
        values = [run[key] for run in all_runs if key in run]
        mean = np.mean(values)
        std = np.std(values)
        agg_rows.append((key, mean, std, values))
        unit = "%" if "acc" in key or "top" in key else ""
        print(f"{key:40s}  mean={mean:7.3f}{unit}  std={std:6.3f}  (values: {[round(v,2) for v in values]})")

    # Save both per-seed and aggregated results for inclusion in the thesis.
    with open("./multi_seed_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "per_seed_runs": all_runs,
            "aggregated": {k: {"mean": m, "std": s} for k, m, s, _ in agg_rows},
        }, f, indent=2)

    print(f"\nSaved per-seed and aggregated results to ./multi_seed_results.json")
    print("Use the mean ± std values here in your thesis instead of a single-run point estimate.")


if __name__ == "__main__":
    main()