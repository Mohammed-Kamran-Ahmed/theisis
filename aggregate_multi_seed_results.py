"""
aggregate_multi_seed_results.py
=================================
Reads every ./multi_seed_runs/seed_*_summary.json produced by
run_multi_seed_evaluation.ps1 and reports mean ± std for the headline
metrics. Pure file I/O — no subprocess calls, so it can't hit the
nested-subprocess torch DLL issue that affected the earlier orchestrator.

Usage:
    python aggregate_multi_seed_results.py
"""

import glob
import json
import numpy as np


def flatten_metrics(summary):
    flat = {}
    for clf_name, metrics in summary["classification"].items():
        flat[f"probe_acc__{clf_name}"] = metrics["accuracy"]
    for k, v in summary["retrieval"].items():
        flat[f"retrieval__{k}"] = v
    flat["n_test"] = summary["n_test"]
    return flat


def main():
    files = sorted(glob.glob("./multi_seed_runs/seed_*_summary.json"))
    if not files:
        print("❌ No files found in ./multi_seed_runs/. Run run_multi_seed_evaluation.ps1 first.")
        return

    all_runs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
        flat = flatten_metrics(summary)
        # seed_42_summary.json -> 42
        seed = f.split("seed_")[1].split("_summary")[0]
        flat["seed"] = seed
        all_runs.append(flat)

    print(f"Loaded {len(all_runs)} seed runs: {[r['seed'] for r in all_runs]}\n")

    all_keys = set()
    for run in all_runs:
        all_keys.update(k for k in run.keys() if k != "seed")

    print(f"{'='*70}")
    print(f"  MULTI-SEED SUMMARY ({len(all_runs)} seeds)")
    print(f"{'='*70}\n")

    aggregated = {}
    for key in sorted(all_keys):
        values = [run[key] for run in all_runs if key in run]
        mean = float(np.mean(values))
        std = float(np.std(values))
        aggregated[key] = {"mean": mean, "std": std, "values": values}
        unit = "%" if ("acc" in key or "top" in key) else ""
        print(f"{key:40s}  mean={mean:7.3f}{unit}  std={std:6.3f}  (values: {[round(v,2) for v in values]})")

    with open("./multi_seed_results.json", "w", encoding="utf-8") as f:
        json.dump({"per_seed_runs": all_runs, "aggregated": aggregated}, f, indent=2)

    print(f"\nSaved to ./multi_seed_results.json")
    print("Use the mean ± std values here in your thesis instead of a single-run point estimate.")


if __name__ == "__main__":
    main()