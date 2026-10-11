"""
compare_baseline_vs_embedding.py
===================================
The comparison your thesis has been building toward since the very first
message of this project: does the learned SAR embedding actually predict
NDVI better than the original hand-crafted-descriptor RF/XGBoost baseline
— on the SAME held-out test patches?

Loads:
  - stage3_baseline_holdout_metrics.csv  (RF/XGBoost NDVI predictions,
    fit on train split, evaluated on test split)
  - The trained Stage 8 Physical Feature Decoder (predicts 5 descriptors
    at once; this script extracts just the NDVI_mean component so the
    comparison is truly one-to-one against the baseline's single target)

Run AFTER stage3_baseline_holdout.py and train_decoders.py have both
completed on the SAME split (i.e., don't re-run utils/data_split.py
between them, or the two won't be comparable).

Usage:
    python compare_baseline_vs_embedding.py
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.decoders import PhysicalFeatureDecoder
from train_decoders import ThesisRealDataset, TARGET_FEATURE_COLUMNS

BASELINE_CSV = "./stage3_baseline_holdout_metrics.csv"
NDVI_INDEX = TARGET_FEATURE_COLUMNS.index("NDVI_mean")  # which column in the 5-target vector is NDVI


def get_decoder_ndvi_predictions():
    test_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="test",
    )

    model = PhysicalFeatureDecoder(latent_dim=test_dataset.z_data.shape[1], num_features=len(TARGET_FEATURE_COLUMNS))
    model.load_state_dict(torch.load("feature_decoder.pth"))
    model.eval()

    with torch.no_grad():
        preds = model(test_dataset.z_data).numpy()

    y_true = test_dataset.features.numpy()[:, NDVI_INDEX]
    y_pred = preds[:, NDVI_INDEX]
    cluster_labels = pd.read_csv("patch_embeddings_stage5.csv")
    cluster_labels = cluster_labels[cluster_labels["split"] == "test"]["cluster"].values

    return y_true, y_pred, cluster_labels, test_dataset.patch_ids


def main():
    try:
        baseline_df = pd.read_csv(BASELINE_CSV)
    except FileNotFoundError:
        print(f"❌ {BASELINE_CSV} not found — run stage3_baseline_holdout.py first.")
        return

    y_true, y_pred, cluster_labels, patch_ids = get_decoder_ndvi_predictions()

    embed_r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    embed_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    embed_mae = mean_absolute_error(y_true, y_pred)

    global_row = baseline_df[baseline_df["cluster"] == "GLOBAL"].iloc[0]

    print("=" * 70)
    print("  BASELINE vs. LEARNED EMBEDDING — NDVI Prediction, Same Test Set")
    print("=" * 70)
    print(f"{'Method':30s} {'R2':>8s} {'RMSE':>8s} {'MAE':>8s}")
    print(f"{'Hand-crafted RF baseline':30s} {global_row['rf_r2']:8.3f} {global_row['rf_rmse']:8.4f} {global_row['rf_mae']:8.4f}")
    print(f"{'Hand-crafted XGBoost baseline':30s} {global_row['xgb_r2']:8.3f} {global_row['xgb_rmse']:8.4f} {global_row['xgb_mae']:8.4f}")
    print(f"{'Learned SAR embedding decoder':30s} {embed_r2:8.3f} {embed_rmse:8.4f} {embed_mae:8.4f}")

    best_baseline_mae = min(global_row['rf_mae'], global_row['xgb_mae'])
    best_baseline_r2 = max(global_row['rf_r2'], global_row['xgb_r2'])
    mae_verdict = "embedding lower (better)" if embed_mae < best_baseline_mae else "baseline lower (better)"
    r2_verdict = "embedding higher (better)" if embed_r2 > best_baseline_r2 else "baseline higher (better)"
    print(f"\nMAE comparison : {mae_verdict}  (embedding {embed_mae:.4f} vs. best baseline {best_baseline_mae:.4f})")
    print(f"R2 comparison  : {r2_verdict}  (embedding {embed_r2:.3f} vs. best baseline {best_baseline_r2:.3f})")
    if mae_verdict.startswith("embedding") != r2_verdict.startswith("embedding"):
        print("\nNOTE: MAE and R2 disagree on which method is better — this is a genuine,")
        print("reportable nuance, not a bug. Lower MAE with lower R2/higher RMSE suggests")
        print("the embedding is more accurate on TYPICAL patches but less accurate on")
        print("high-variance/outlier patches than the baseline. Report both metrics rather")
        print("than picking one to declare a clean winner.")
    verdict = f"MAE: {mae_verdict}; R2: {r2_verdict}"

    # Per-cluster comparison, where both sides have enough test patches.
    print("\n" + "=" * 70)
    print("  PER-CLUSTER COMPARISON")
    print("=" * 70)
    result_rows = [{
        "cluster": "GLOBAL", "n": len(y_true),
        "rf_mae": global_row["rf_mae"], "xgb_mae": global_row["xgb_mae"], "embedding_mae": embed_mae,
        "verdict": verdict,
    }]

    df_test = pd.DataFrame({"cluster": cluster_labels, "y_true": y_true, "y_pred": y_pred})
    for cluster_id in sorted(df_test["cluster"].unique()):
        # baseline_df's "cluster" column is read back as strings, since the
        # mixed "GLOBAL" row forces the whole column to object/string dtype
        # on CSV round-trip — compare both sides as strings to match correctly.
        c_rows = baseline_df[baseline_df["cluster"].astype(str) == str(cluster_id)]
        if c_rows.empty:
            continue  # baseline script already excluded this cluster as too small
        c_sub = df_test[df_test["cluster"] == cluster_id]
        c_embed_mae = mean_absolute_error(c_sub["y_true"], c_sub["y_pred"])
        c_rf_mae = c_rows.iloc[0]["rf_mae"]
        c_xgb_mae = c_rows.iloc[0]["xgb_mae"]
        c_best_baseline = min(c_rf_mae, c_xgb_mae)
        c_verdict = "Embedding wins" if c_embed_mae < c_best_baseline else "Baseline wins"
        print(f"Cluster {cluster_id} (n={len(c_sub)}): RF MAE={c_rf_mae:.4f}, XGB MAE={c_xgb_mae:.4f}, "
              f"Embedding MAE={c_embed_mae:.4f} -> {c_verdict}")
        result_rows.append({
            "cluster": cluster_id, "n": len(c_sub),
            "rf_mae": c_rf_mae, "xgb_mae": c_xgb_mae, "embedding_mae": c_embed_mae,
            "verdict": c_verdict,
        })

    pd.DataFrame(result_rows).to_csv("./baseline_vs_embedding_comparison.csv", index=False)
    print(f"\nSaved to ./baseline_vs_embedding_comparison.csv")


if __name__ == "__main__":
    main()