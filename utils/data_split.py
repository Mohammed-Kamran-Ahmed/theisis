"""
utils/data_split.py
====================
Creates ONE authoritative train/test split, stratified by GMM cluster,
and writes it back into patch_metadata_clustered.csv as a `split` column
("train" / "test"). Every downstream stage (4, 5, 6, decoders) must read
this column and respect it — this is what was missing before, and it's
why the earlier numbers (90% retrieval, 0.71 SSIM, etc.) were computed on
data the models had already seen during training.

Run this ONCE, right after Stage 3 (GMM clustering), before Stage 4.

Note on a residual, smaller caveat: GMM clustering itself was fit on ALL
172 patches (including what becomes the test set) back in Stage 3. That's
a mild form of leakage too — ideally GMM would be fit on train-only data
and test patches would be assigned via .predict(). This script does not
retroactively fix that (it would require re-running Stage 3), but it's
worth stating explicitly as a limitation in your thesis methodology
section rather than leaving it unmentioned.
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

METADATA_CSV = "./patch_metadata_clustered.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                         help="Random seed for the train/test split (used by run_multi_seed_evaluation.py).")
    args = parser.parse_args()
    seed = args.seed

    if not os.path.exists(METADATA_CSV):
        print(f"❌ Could not find {METADATA_CSV}. Run Stage 3 (GMM clustering) first.")
        return

    df = pd.read_csv(METADATA_CSV)
    clean_df = df[df["valid"] == True].copy()

    label_col = "gmm_cluster" if "gmm_cluster" in clean_df.columns else "cluster"
    labels = clean_df[label_col].values

    # Stratified split needs at least 2 members per class; fall back to a
    # plain (non-stratified) split if any cluster is too small.
    _, counts = np.unique(labels, return_counts=True)
    can_stratify = counts.min() >= 2

    indices = np.arange(len(clean_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )

    clean_df["split"] = "train"
    clean_df.iloc[test_idx, clean_df.columns.get_loc("split")] = "test"

    if not can_stratify:
        print("⚠️ At least one GMM cluster has < 2 members — used a non-stratified split.")

    # clean_df kept the same row index as df (just filtered), so assign
    # back by index directly — far more robust than a column-based merge.
    df["split"] = pd.NA
    df.loc[clean_df.index, "split"] = clean_df["split"].values

    df.to_csv(METADATA_CSV, index=False)

    train_n = (df["split"] == "train").sum()
    test_n = (df["split"] == "test").sum()
    print(f"[seed={seed}] Split written to {METADATA_CSV}: {train_n} train / {test_n} test patches.")
    print("Per-cluster breakdown:")
    print(clean_df.groupby([label_col, "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()