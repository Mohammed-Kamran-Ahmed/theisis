"""
failure_mode_analysis.py
==========================
This is the core contribution script — everything before this point built
the representation-learning pipeline; this is what actually answers your
thesis question: WHERE does SAR-optical correlation hold up, and WHERE
does it break down, and WHY (tied to physical land-cover conditions)?

For the held-out TEST split only, this script:
  1. Trains a probe classifier on TRAIN embeddings, predicts per-TEST-patch,
     and records per-patch correct/incorrect — then aggregates by GMM
     cluster AND by region.
  2. Computes per-patch Top-1 retrieval correctness (same logic as Stage 6,
     but kept per-sample instead of only aggregated).
  3. Runs the trained Stage 8/9 decoders per test patch, recording
     per-patch feature MAE and image SSIM.
  4. Joins all of this with each patch's physical descriptor values
     (NDVI_mean, RVI_mean, entropy_proxy_mean, VH_VV_ratio_mean) and region,
     then aggregates everything by cluster and by region — so you can see,
     e.g., "cluster 3 (mostly Kerala backwaters, high NDVI variance) has
     40% lower retrieval accuracy than cluster 1" and tie that to a
     physical explanation (flooded-field double-bounce, cloud-driven
     acquisition mismatch, canopy backscatter saturation, etc.)

Run this AFTER stage4 → stage5 → stage6 → train_decoders → evaluate_generation
have all completed on the split you want to analyze.

Output: failure_mode_analysis.csv (per-cluster and per-region breakdown)
        failure_mode_analysis_per_patch.csv (raw per-patch data, for your
        own deeper digging / plotting)
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder
from train_decoders import ThesisRealDataset, TARGET_FEATURE_COLUMNS
from utils.metrics import compute_feature_metrics, compute_image_metrics

EMBEDDINGS_CSV = "./patch_embeddings_stage5.csv"
METADATA_CSV = "./patch_metadata_clustered.csv"
PATCH_DIR = "./downloaded_patches"
RANDOM_STATE = 42

DESCRIPTOR_COLS = ["NDVI_mean", "NDWI_mean", "EVI_mean", "RVI_mean", "VH_VV_ratio_mean", "entropy_proxy_mean"]


def load_merged():
    emb_df = pd.read_csv(EMBEDDINGS_CSV)
    meta_df = pd.read_csv(METADATA_CSV)

    if "split" not in emb_df.columns:
        raise ValueError("❌ No `split` column — run the corrected Stage 5 first.")

    meta_small = meta_df[["export_name", "region"] + [c for c in DESCRIPTOR_COLS if c in meta_df.columns]]
    merged = emb_df.merge(meta_small, left_on="patch_id", right_on="export_name", how="left")
    return merged


def per_patch_probe_and_retrieval(merged):
    feature_cols = [c for c in merged.columns if c.startswith("fused_emb_")]
    train_df = merged[merged["split"] == "train"].reset_index(drop=True)
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)

    X_train, y_train = train_df[feature_cols].values, train_df["cluster"].values
    X_test, y_test = test_df[feature_cols].values, test_df["cluster"].values

    # Probe: use Linear SVM (matches your best-performing classifier from Stage 6).
    clf = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    test_df["probe_correct"] = (preds == y_test).astype(int)

    # Retrieval: Top-1, test-only gallery, same logic as Stage 6 but per-sample.
    sim_matrix = cosine_similarity(X_test, X_test)
    np.fill_diagonal(sim_matrix, -np.inf)
    top1_correct = []
    for i in range(len(y_test)):
        best_idx = np.argmax(sim_matrix[i])
        top1_correct.append(int(y_test[i] == y_test[best_idx]))
    test_df["retrieval_top1_correct"] = top1_correct

    return test_df


def per_patch_decoder_error():
    test_dataset = ThesisRealDataset(
        embeddings_csv=EMBEDDINGS_CSV,
        metadata_csv=METADATA_CSV,
        patches_dir=PATCH_DIR,
        split="test",
    )

    feat_model = PhysicalFeatureDecoder(latent_dim=test_dataset.z_data.shape[1], num_features=len(TARGET_FEATURE_COLUMNS))
    img_model = OpticalImageDecoder(latent_dim=test_dataset.z_data.shape[1], out_channels=3)
    feat_model.load_state_dict(torch.load("feature_decoder.pth"))
    img_model.load_state_dict(torch.load("image_decoder.pth"))
    feat_model.eval()
    img_model.eval()

    rows = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            z, feat, img = test_dataset[i]
            z_b, feat_b, img_b = z.unsqueeze(0), feat.unsqueeze(0), img.unsqueeze(0)

            pred_feat = feat_model(z_b)
            pred_img = img_model(z_b)

            feat_res = compute_feature_metrics(pred_feat, feat_b)
            img_res = compute_image_metrics(pred_img, img_b)

            rows.append({
                "patch_id": test_dataset.patch_ids[i],
                "feature_mae": feat_res["feat_mae"],
                "image_ssim": img_res["mean_ssim"],
            })

    return pd.DataFrame(rows)


def aggregate_and_report(full_df, group_col):
    print(f"\n{'='*70}\n  BREAKDOWN BY {group_col.upper()}\n{'='*70}")

    agg = full_df.groupby(group_col).agg(
        n_patches=("patch_id", "count"),
        probe_accuracy=("probe_correct", "mean"),
        retrieval_top1_accuracy=("retrieval_top1_correct", "mean"),
        feature_mae=("feature_mae", "mean"),
        image_ssim=("image_ssim", "mean"),
        **{f"mean_{c}": (c, "mean") for c in DESCRIPTOR_COLS if c in full_df.columns},
    ).reset_index()

    agg["probe_accuracy"] = (agg["probe_accuracy"] * 100).round(1)
    agg["retrieval_top1_accuracy"] = (agg["retrieval_top1_accuracy"] * 100).round(1)
    agg["feature_mae"] = agg["feature_mae"].round(4)
    agg["image_ssim"] = agg["image_ssim"].round(4)

    print(agg.to_string(index=False))
    return agg


def main():
    print("Loading and merging embeddings + metadata...")
    merged = load_merged()

    print("Running probe + retrieval per test patch...")
    test_with_probe = per_patch_probe_and_retrieval(merged)

    print("Running decoders per test patch...")
    decoder_errors = per_patch_decoder_error()

    full_df = test_with_probe.merge(decoder_errors, on="patch_id", how="left")

    by_cluster = aggregate_and_report(full_df, "cluster")
    by_region = aggregate_and_report(full_df, "region")

    full_df.to_csv("./failure_mode_analysis_per_patch.csv", index=False)
    by_cluster.to_csv("./failure_mode_analysis_by_cluster.csv", index=False)
    by_region.to_csv("./failure_mode_analysis_by_region.csv", index=False)

    print("\n\nSaved:")
    print("  - failure_mode_analysis_per_patch.csv   (raw per-patch data)")
    print("  - failure_mode_analysis_by_cluster.csv  (aggregated by GMM cluster)")
    print("  - failure_mode_analysis_by_region.csv   (aggregated by region)")

    # Quick interpretive pointer — identify the weakest cluster/region as a
    # starting point for the mechanistic explanation write-up.
    weakest_cluster = by_cluster.loc[by_cluster["retrieval_top1_accuracy"].idxmin()]
    strongest_cluster = by_cluster.loc[by_cluster["retrieval_top1_accuracy"].idxmax()]
    print(f"\nWeakest cluster by retrieval accuracy: cluster {weakest_cluster['cluster']} "
          f"({weakest_cluster['retrieval_top1_accuracy']}%, n={weakest_cluster['n_patches']})")
    print(f"Strongest cluster by retrieval accuracy: cluster {strongest_cluster['cluster']} "
          f"({strongest_cluster['retrieval_top1_accuracy']}%, n={strongest_cluster['n_patches']})")
    print("\nCompare their mean_NDVI_mean / mean_RVI_mean / mean_entropy_proxy_mean rows above —")
    print("a large gap in these physical descriptors between the weakest and strongest cluster")
    print("is your starting point for the mechanistic (physics-grounded) explanation.")


if __name__ == "__main__":
    main()