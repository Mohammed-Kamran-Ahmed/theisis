"""
compare_runs.py - aggregate finished runs (mean +/- std over seeds) into one table.

  python -m reben_pipeline.compare_runs results/baseline_seed* results/cls_seed* \
         results/align_seed* results/cahn_seed* results/cahn_ierc_seed*
  python -m reben_pipeline.compare_runs results/* --out results/table.md

Reads lmdb_experiment_summary.json (dual encoder) or baseline_summary.json (baseline).
Rows are grouped by config["variant"]. Std is the sample std (ddof=1); with one seed it is
shown as n/a - a single seed cannot support a claim that one variant beats another.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUMMARY_NAMES = ("lmdb_experiment_summary.json", "baseline_summary.json")
ORDER = ["baseline", "cls", "align", "cahn", "cahn_ierc"]
COLUMNS = [
    ("Val AP-macro", lambda s: s["best_val_ap_macro"]),
    ("Test AP-macro", lambda s: s["test"]["fusion"]["ap_macro"]),
    ("Test AP-micro", lambda s: s["test"]["fusion"]["ap_micro"]),
    ("Test F1-macro", lambda s: s["test"]["fusion"]["f1_macro"]),
    ("Test F1-micro", lambda s: s["test"]["fusion"]["f1_micro"]),
]


def load_summaries(paths):
    out = []
    for p in paths:
        p = Path(p)
        for name in SUMMARY_NAMES:
            f = p / name if p.is_dir() else p
            if f.is_file() and f.name in SUMMARY_NAMES:
                with open(f, encoding="utf-8") as fh:
                    out.append((str(f.parent), json.load(fh)))
                break
    return out


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    mean = float(np.mean(values))
    if len(values) < 2:
        return f"{mean:.2f} (std n/a)"
    return f"{mean:.2f} ± {float(np.std(values, ddof=1)):.2f}"


def build_table(summaries):
    groups = defaultdict(list)
    for _, s in summaries:
        groups[s["config"].get("variant", "unknown")].append(s)
    names = sorted(groups, key=lambda v: ORDER.index(v) if v in ORDER else len(ORDER))

    head = "| Variant | Seeds | " + " | ".join(c for c, _ in COLUMNS) + " | Test classes present |"
    sep = "|" + "---|" * (len(COLUMNS) + 3)
    rows = [head, sep]
    for v in names:
        runs = groups[v]
        cells = [fmt([fn(r) for r in runs]) for _, fn in COLUMNS]
        present = sorted({r["test"]["n_classes_present"] for r in runs})
        rows.append(f"| {v} | {len(runs)} | " + " | ".join(cells) + f" | {present} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run directories or summary json files")
    ap.add_argument("--out", default=None, help="also write the table to this .md file")
    args = ap.parse_args(argv)
    summaries = load_summaries(args.paths)
    if not summaries:
        raise SystemExit("No summary JSON files found in the given paths.")
    table = build_table(summaries)
    print(f"Loaded {len(summaries)} runs\n")
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return table


if __name__ == "__main__":
    main()