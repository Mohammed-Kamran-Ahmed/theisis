"""
run_log.py
Import this in cca_analysis.py (or wherever you print the CCA
comparison table) and call log_cca_result(...) once per stage,
right after you compute r1 for that stage. It appends one JSON line
per call to run_history.jsonl -- so every run you ever do is kept,
timestamped, with the seed and checkpoint path that produced it.

Usage inside cca_analysis.py, right where you currently print the
comparison table row for each stage:

    from run_log import log_cca_result
    log_cca_result(
        stage="IERC fine-tuned (Stage 4b)",
        r1=r1_value,
        r_bar=r_bar_value,
        null_mean=null_mean_value,
        null_std=null_std_value,
        seed=42,
        checkpoint="sar_encoder_reclustered.pth",
    )
"""
import json
import os
import subprocess
import time

LOG_PATH = "run_history.jsonl"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def log_cca_result(stage, r1, r_bar, null_mean, null_std, seed, checkpoint):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "seed": seed,
        "checkpoint": checkpoint,
        "stage": stage,
        "r1": r1,
        "r_bar": r_bar,
        "null_mean": null_mean,
        "null_std": null_std,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[run_log] logged {stage}: r1={r1:.4f} -> {LOG_PATH}")


def show_history(stage_filter=None):
    """Print every logged run, optionally filtered to one stage name."""
    if not os.path.exists(LOG_PATH):
        print("No run_history.jsonl yet.")
        return
    with open(LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if stage_filter and stage_filter not in entry["stage"]:
                continue
            print(
                f"{entry['timestamp']} | seed={entry['seed']} | "
                f"commit={entry['git_commit']} | {entry['stage']} | "
                f"r1={entry['r1']:.4f} | ckpt={entry['checkpoint']}"
            )


if __name__ == "__main__":
    show_history()