"""
Phase 3 — GMM Clustering for Multi-Modal SAR + Optical Patches
=============================================================

Purpose
-------
1. Loads validated descriptor stats from `patch_metadata_validated.csv`.
2. Filters exclusively for the clean dual-sensor patches (valid == True).
3. Normalizes feature vectors using StandardScaler.
4. Sweeps component counts (K=2..8) and computes BIC/AIC scores to find the optimal K.
5. Trains a Gaussian Mixture Model (GMM) to uncover distinct agricultural / land-cover clusters.
6. Saves cluster assignments and membership probabilities to `patch_metadata_clustered.csv`.
7. Saves diagnostic plots (BIC/AIC curve and PCA cluster projection).

Setup
-----
pip install numpy pandas scikit-learn matplotlib seaborn
"""

import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

INPUT_CSV = "./patch_metadata_validated.csv"
OUTPUT_CSV = "./patch_metadata_clustered.csv"
PLOT_DIR = "./clustering_plots"

# Select feature columns to include in the clustering feature vector
# Using mean values of key optical, SAR, and structural descriptors
FEATURE_COLS = [
    "VV_mean", "VH_mean", "VH_VV_ratio_mean", "entropy_proxy_mean", "RVI_mean",
    "B2_mean", "B3_mean", "B4_mean", "B8_mean",
    "NDVI_mean", "NDWI_mean", "EVI_mean"
]

# Range of clusters to evaluate for BIC / AIC selection
K_RANGE = range(2, 9)
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: Could not find {INPUT_CSV}. Run Phase 2 validation first.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded total metadata records: {len(df)}")

    # Filter for valid dual-sensor patches
    clean_df = df[df["valid"] == True].copy().reset_index(drop=True)
    print(f"Filtered for clean multi-modal patches: {len(clean_df)} / {len(df)}")

    if len(clean_df) == 0:
        print("❌ Error: No valid patches found for clustering.")
        return

    # Ensure all feature columns exist
    missing_feats = [c for c in FEATURE_COLS if c not in clean_df.columns]
    if missing_feats:
        print(f"❌ Error: Missing required feature columns in CSV: {missing_feats}")
        return

    X_raw = clean_df[FEATURE_COLS].values

    # 2. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Model Selection via BIC / AIC Score Sweep
    bics = []
    aics = []
    models = {}

    print("\n--- Sweeping Gaussian Mixture Models (K=2..8) ---")
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)
        
        bics.append(bic)
        aics.append(aic)
        models[k] = gmm
        print(f"  K = {k} | BIC = {bic:.2f} | AIC = {aic:.2f}")

    # Optimal K has the lowest BIC
    best_k = K_RANGE[np.argmin(bics)]
    print(f"\nOptimal cluster count selected by BIC: K = {best_k}")

    # 4. Plot BIC / AIC Selection Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_RANGE, bics, label="BIC (Bayesian Info Criterion)", marker="o", color="crimson")
    plt.plot(K_RANGE, aics, label="AIC (Akaike Info Criterion)", marker="s", color="royalblue")
    plt.axvline(best_k, linestyle="--", color="gray", label=f"Optimal K={best_k}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Information Score (Lower is better)")
    plt.title("GMM Model Selection: BIC / AIC Curve")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    bic_plot_path = os.path.join(PLOT_DIR, "gmm_bic_aic_curve.png")
    plt.savefig(bic_plot_path, dpi=120)
    plt.close()
    print(f"Saved model selection plot: {bic_plot_path}")

    # 5. Fit Final Model
    best_gmm = models[best_k]
    cluster_labels = best_gmm.predict(X_scaled)
    probs = best_gmm.predict_proba(X_scaled)

    # Attach results to dataframe
    clean_df["gmm_cluster"] = cluster_labels
    for i in range(best_k):
        clean_df[f"cluster_{i}_prob"] = probs[:, i]

    # Save to CSV
    clean_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved clustered results to: {OUTPUT_CSV}")

    # 6. Cluster Diagnostics & Summary
    print(f"\n--- Cluster Distribution (K={best_k}) ---")
    print(clean_df["gmm_cluster"].value_counts().sort_index())

    print("\n--- Cluster Mean Feature Profile ---")
    profile = clean_df.groupby("gmm_cluster")[FEATURE_COLS].mean().T
    print(profile.round(3))

    # 7. PCA Visual Scatter Plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=clean_df["gmm_cluster"],
        palette="Set1", style=clean_df["gmm_cluster"],
        s=70, alpha=0.85
    )
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} Var)")
    plt.title(f"GMM Agricultural Clusters (K={best_k}) Projected on PCA Space")
    plt.grid(True, linestyle=":", alpha=0.5)
    pca_plot_path = os.path.join(PLOT_DIR, "gmm_pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=120)
    plt.close()
    print(f"Saved PCA cluster plot: {pca_plot_path}")

if __name__ == "__main__":
    main()