"""
stage3_baseline_holdout.py
============================
Corrected version of the Stage 3 baseline regression, fixing an evaluation
protocol mismatch with the rest of the pipeline: the original script used
5-fold cross-validation over ALL 172 patches, while every embedding-based
result (Stage 6 probing/retrieval, Stage 8 decoder) is evaluated strictly
on the 35 held-out test patches from the `split` column. Comparing a
172-patch CV score against a 35-patch held-out score is not apples-to-
apples — different sample sets, different evaluation protocols.

This version fits RF/XGBoost on the TRAIN split only and evaluates on the
TEST split only — identical protocol to the rest of the pipeline — so the
result can be directly compared against the learned-embedding decoder's
performance on the same 35 patches (see compare_baseline_vs_embedding.py).

Setup
-----
pip install numpy pandas scikit-learn xgboost
"""

import os
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

INPUT_CSV = "./patch_metadata_clustered.csv"
OUTPUT_METRICS_CSV = "./stage3_baseline_holdout_metrics.csv"

SAR_FEATURES = ["VV_mean", "VH_mean", "VH_VV_ratio_mean", "entropy_proxy_mean", "RVI_mean"]
TARGET = "NDVI_mean"
RANDOM_STATE = 42
MIN_CLUSTER_TEST_SIZE = 3   # below this, a per-cluster R2/RMSE is too noisy to report


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Could not find {INPUT_CSV}.")
        return

    df = pd.read_csv(INPUT_CSV)
    if "split" not in df.columns:
        print("❌ No `split` column — run utils/data_split.py first so this baseline "
              "uses the SAME train/test split as the rest of the pipeline.")
        return

    valid = df[df["valid"] == True].copy()
    train_df = valid[valid["split"] == "train"]
    test_df = valid[valid["split"] == "test"]
    print(f"Train: {len(train_df)} patches | Test: {len(test_df)} patches "
          f"(same split as Stages 4-9 — this is what makes the comparison fair)")

    X_train, y_train = train_df[SAR_FEATURES].values, train_df[TARGET].values
    X_test, y_test = test_df[SAR_FEATURES].values, test_df[TARGET].values

    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    test_df = test_df.copy()
    test_df["rf_pred_ndvi"] = rf_pred
    test_df["xgb_pred_ndvi"] = xgb_pred

    print("\n" + "=" * 60)
    print(" GLOBAL BASELINE — TEST SET ONLY (35 patches, held out)")
    print("=" * 60)

    def report(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print(f"{name:15s} -> R2: {r2:6.3f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        return {"R2": r2, "RMSE": rmse, "MAE": mae}

    rf_metrics = report(y_test, rf_pred, "Random Forest")
    xgb_metrics = report(y_test, xgb_pred, "XGBoost")

    print("\n" + "=" * 60)
    print(" PER-CLUSTER BREAKDOWN (TEST SET ONLY)")
    print("=" * 60)

    label_col = "gmm_cluster" if "gmm_cluster" in test_df.columns else "cluster"
    metrics_list = [{
        "cluster": "GLOBAL", "sample_count": len(test_df),
        "rf_r2": rf_metrics["R2"], "rf_rmse": rf_metrics["RMSE"], "rf_mae": rf_metrics["MAE"],
        "xgb_r2": xgb_metrics["R2"], "xgb_rmse": xgb_metrics["RMSE"], "xgb_mae": xgb_metrics["MAE"],
    }]

    for c in sorted(test_df[label_col].unique()):
        c_df = test_df[test_df[label_col] == c]
        if len(c_df) < MIN_CLUSTER_TEST_SIZE:
            print(f"Cluster {c} (N={len(c_df)}): too few test patches — skipping.")
            continue
        y_true = c_df[TARGET].values
        rf_r2 = r2_score(y_true, c_df["rf_pred_ndvi"].values) if len(np.unique(y_true)) > 1 else float("nan")
        rf_mae = mean_absolute_error(y_true, c_df["rf_pred_ndvi"].values)
        xgb_r2 = r2_score(y_true, c_df["xgb_pred_ndvi"].values) if len(np.unique(y_true)) > 1 else float("nan")
        xgb_mae = mean_absolute_error(y_true, c_df["xgb_pred_ndvi"].values)
        print(f"Cluster {c} (N={len(c_df):2d}) | RF R2: {rf_r2:6.3f} MAE: {rf_mae:.4f} | "
              f"XGB R2: {xgb_r2:6.3f} MAE: {xgb_mae:.4f}")
        metrics_list.append({
            "cluster": c, "sample_count": len(c_df),
            "rf_r2": rf_r2, "rf_rmse": np.sqrt(mean_squared_error(y_true, c_df["rf_pred_ndvi"].values)), "rf_mae": rf_mae,
            "xgb_r2": xgb_r2, "xgb_rmse": np.sqrt(mean_squared_error(y_true, c_df["xgb_pred_ndvi"].values)), "xgb_mae": xgb_mae,
        })

    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\nSaved to {OUTPUT_METRICS_CSV}")
    print("\nNote: this is a SINGLE-TARGET (NDVI only) baseline, matching the original")
    print("Stage 3 scope. The embedding-based decoder predicts 5 targets at once — use")
    print("compare_baseline_vs_embedding.py to extract its NDVI-only component for a")
    print("true apples-to-apples comparison against this baseline.")


if __name__ == "__main__":
    main()