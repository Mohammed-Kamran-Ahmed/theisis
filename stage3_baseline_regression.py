"""
Stage 3 — Baseline Regression (Hand-Crafted Descriptors -> NDVI)
==============================================================

Purpose
-------
1. Loads `patch_metadata_clustered.csv` (generated in Stage 2).
2. Trains Random Forest and XGBoost regressors using hand-crafted SAR descriptors
   (VV, VH, VH_VV_ratio, RVI, entropy_proxy) to predict target optical NDVI.
3. Evaluates models both globally and stratified across each GMM cluster (0..7).
4. Saves benchmark metrics (R2, RMSE, MAE) to `stage3_baseline_metrics.csv`
   to serve as the control condition for Stage 5 contrastive embeddings.

IMPORTANT: this script evaluates via 5-fold cross-validation over the ENTIRE
dataset (train + test patches pooled) -- it does not read or respect the
'split' column at all. This means these numbers are unaffected by whether
the train/test split is the original leaky one or the corrected spatial
one (Section 3 of the paper), but it also means they are not directly
comparable to the held-out test-split numbers reported for Stage 4+ (which
DO use 'split'). State this explicitly wherever these numbers are cited.

Setup
-----
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from seed_lock import set_seed  # <-- ADDED

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

INPUT_CSV = "./patch_metadata_clustered.csv"
OUTPUT_METRICS_CSV = "./stage3_baseline_metrics.csv"

SAR_FEATURES = [
    "VV_mean",
    "VH_mean",
    "VH_VV_ratio_mean",
    "entropy_proxy_mean",
    "RVI_mean"
]

TARGET = "NDVI_mean"
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# EVALUATION HELPER
# ----------------------------------------------------------------------

def evaluate_model_cv(X, y, model_type="rf", n_splits=5):
    """Runs cross-validation and returns out-of-fold predictions."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        elif model_type == "xgb":
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    r2 = r2_score(y, oof_preds)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)

    return oof_preds, {"R2": r2, "RMSE": rmse, "MAE": mae}


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    set_seed(RANDOM_STATE)  # <-- ADDED: fixes numpy global RNG used elsewhere in sklearn/xgboost internals

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Could not find {INPUT_CSV}. Run Stage 2 GMM script first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} validated patches for Stage 3 Baseline.")
    print("NOTE: this evaluation pools train+test via 5-fold CV and does not "
          "use the 'split' column -- see module docstring.\n")

    X = df[SAR_FEATURES].values
    y = df[TARGET].values

    print("==================================================")
    print(" 1. GLOBAL BASELINE REGRESSION (Hand-Crafted -> NDVI)")
    print("==================================================")

    rf_oof, rf_global_metrics = evaluate_model_cv(X, y, model_type="rf")
    xgb_oof, xgb_global_metrics = evaluate_model_cv(X, y, model_type="xgb")

    df["rf_pred_ndvi"] = rf_oof
    df["xgb_pred_ndvi"] = xgb_oof

    print(f"Random Forest -> R²: {rf_global_metrics['R2']:.3f} | RMSE: {rf_global_metrics['RMSE']:.4f} | MAE: {rf_global_metrics['MAE']:.4f}")
    print(f"XGBoost       -> R²: {xgb_global_metrics['R2']:.3f} | RMSE: {xgb_global_metrics['RMSE']:.4f} | MAE: {xgb_global_metrics['MAE']:.4f}")

    print("\n==================================================")
    print(" 2. PER-CLUSTER STRATIFIED BENCHMARKS (GMM Clusters 0..7)")
    print("==================================================")

    metrics_list = []

    clusters = sorted(df["gmm_cluster"].unique())
    for c in clusters:
        cluster_mask = df["gmm_cluster"] == c
        c_df = df[cluster_mask]

        if len(c_df) < 3:
            print(f"Skipping Cluster {c}: Too few samples ({len(c_df)} patches)")
            continue

        y_true = c_df[TARGET].values
        rf_p = c_df["rf_pred_ndvi"].values
        xgb_p = c_df["xgb_pred_ndvi"].values

        rf_r2 = r2_score(y_true, rf_p) if len(np.unique(y_true)) > 1 else 0.0
        rf_rmse = np.sqrt(mean_squared_error(y_true, rf_p))

        xgb_r2 = r2_score(y_true, xgb_p) if len(np.unique(y_true)) > 1 else 0.0
        xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_p))

        print(f"Cluster {c} (N={len(c_df):2d}) | RF R²: {rf_r2:6.3f}, RMSE: {rf_rmse:.4f} | XGB R²: {xgb_r2:6.3f}, RMSE: {xgb_rmse:.4f}")

        metrics_list.append({
            "cluster": c,
            "sample_count": len(c_df),
            "rf_r2": rf_r2,
            "rf_rmse": rf_rmse,
            "xgb_r2": xgb_r2,
            "xgb_rmse": xgb_rmse,
        })

    metrics_list.append({
        "cluster": "GLOBAL",
        "sample_count": len(df),
        "rf_r2": rf_global_metrics["R2"],
        "rf_rmse": rf_global_metrics["RMSE"],
        "xgb_r2": xgb_global_metrics["R2"],
        "xgb_rmse": xgb_global_metrics["RMSE"],
    })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\nSaved baseline benchmark metrics to: {OUTPUT_METRICS_CSV}")


if __name__ == "__main__":
    main()