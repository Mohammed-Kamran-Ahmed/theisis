"""
cca_analysis.py
=================
Novelty 4: Canonical Correlation Analysis (CCA) Cross-Modal Latent Alignment Benchmark
"""

import sys
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from stage4_contrastive_learning import ModalityEncoder, MultiModalSAROptDataset
from seed_lock import set_seed          # <-- ADDED
from run_log import log_cca_result      # <-- ADDED

METADATA_CSV   = "./patch_metadata_reclustered.csv"
PATCH_DIR      = "./downloaded_patches"
MODEL_DIR      = "./saved_models"

N_COMPONENTS    = 3
SAMPLES_PER_DIM = 5
N_PERMUTATIONS  = 200
SEED            = 42                    # <-- ADDED: single source of truth for this run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_projections_from_models(sar_enc, opt_enc, df, patch_dir):
    """Extracts SAR and Optical projection vectors using current encoders."""
    dataset = MultiModalSAROptDataset(df, patch_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    sar_enc.eval()
    opt_enc.eval()

    z_sar_list, z_opt_list = [], []
    with torch.no_grad():
        for sar_b, opt_b, _ in loader:
            sar_b = sar_b.to(DEVICE)
            opt_b = opt_b.to(DEVICE)
            z_s = sar_enc(sar_b)
            z_o = opt_enc(opt_b)
            z_sar_list.append(z_s.cpu().numpy())
            z_opt_list.append(z_o.cpu().numpy())

    return np.vstack(z_sar_list), np.vstack(z_opt_list)


def load_current_encoders(checkpoint_path, device):
    """Loads sar_enc and opt_enc from unified stage4/4b checkpoint dictionaries."""
    sar_enc = ModalityEncoder(proj_dim=64).to(device)
    opt_enc = ModalityEncoder(proj_dim=64).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Using random initialization.")
        return sar_enc, opt_enc

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sar_enc.load_state_dict(checkpoint["sar_state"])
    opt_enc.load_state_dict(checkpoint["opt_state"])
    return sar_enc, opt_enc


def reduce_dims_if_needed(X, n_samples):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    target = min(X.shape[1], n_samples - 1)
    if X.shape[1] > target:
        pca = PCA(n_components=target, random_state=SEED)
        return pca.fit_transform(X), target
    return X, X.shape[1]


def permutation_null_correlation(X, Y, n_components, n_perm=N_PERMUTATIONS):
    rng = np.random.RandomState(SEED)
    null_corrs = []
    for _ in range(n_perm):
        Y_shuffled = Y[rng.permutation(Y.shape[0])]
        try:
            cca = CCA(n_components=n_components, max_iter=500)
            X_c, Y_c = cca.fit_transform(X, Y_shuffled)
            null_corrs.append(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
        except Exception:
            continue
    return (
        float(np.mean(null_corrs)) if len(null_corrs) > 0 else 0.0,
        float(np.std(null_corrs)) if len(null_corrs) > 0 else 0.0,
        null_corrs,
    )


def run_cca(X, Y, n_components, label, apply_pca=True):
    n_samples = X.shape[0]
    if apply_pca:
        X, _ = reduce_dims_if_needed(X, n_samples)
        Y, _ = reduce_dims_if_needed(Y, n_samples)

    safe_components = max(1, n_samples // SAMPLES_PER_DIM)
    n_components = min(n_components, X.shape[1], Y.shape[1], safe_components, n_samples - 1)
    if n_components < 1:
        return None

    X_std = StandardScaler().fit_transform(X)
    Y_std = StandardScaler().fit_transform(Y)

    try:
        cca = CCA(n_components=n_components, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X_std, Y_std)
        correlations = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_components)]
        null_mean, null_std, null_corrs = permutation_null_correlation(X_std, Y_std, n_components)
        gap = correlations[0] - null_mean

        # --- FIXED: was a hardcoded "gap > 0.15" threshold, unrelated to the
        # permutation null's own spread. Use the null-referenced z-score
        # (Delta_null / sigma_null) and the add-one permutation p-value
        # instead, matching what the paper actually claims to compute.
        exceed = sum(1 for r in null_corrs if r >= correlations[0])
        p_value = (1 + exceed) / (len(null_corrs) + 1) if null_corrs else 1.0
        z_score = gap / null_std if null_std > 1e-8 else float("inf")
        verdict = "SIGNIFICANT" if z_score >= 3.0 else "MARGINAL"

        return {
            "correlations": correlations,
            "null_mean": null_mean,
            "null_std": null_std,
            "gap": gap,
            "z_score": z_score,
            "p_value": p_value,
            "verdict": verdict,
        }
    except Exception as e:
        print(f"[{label}] CCA error: {e}")
        return None


def run_regime(name, checkpoint_path, is_random=False):
    input_csv = METADATA_CSV if os.path.exists(METADATA_CSV) else "./patch_metadata_clustered.csv"
    df = pd.read_csv(input_csv)
    clean_df = df[df.get("valid", True) == True].copy().reset_index(drop=True)

    if is_random:
        sar_enc = ModalityEncoder(proj_dim=64).to(DEVICE)
        opt_enc = ModalityEncoder(proj_dim=64).to(DEVICE)
    else:
        sar_enc, opt_enc = load_current_encoders(checkpoint_path, DEVICE)

    X_sar, Y_opt = extract_projections_from_models(sar_enc, opt_enc, clean_df, PATCH_DIR)
    return run_cca(X_sar, Y_opt, N_COMPONENTS, name, apply_pca=True)


def main():
    set_seed(SEED)  # <-- ADDED: fix PCA / permutation RNGs for this run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["before_supcon", "post_supcon", "post_recluster", "full_comparison"],
        default="full_comparison",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("   NOVELTY 4: CROSS-MODAL CANONICAL CORRELATION ANALYSIS (CCA)")
    print("=" * 70)

    regimes = {}
    checkpoints = {}
    stage4_ckpt = os.path.join(MODEL_DIR, "stage4_best.pth")
    stage4b_ckpt = os.path.join(MODEL_DIR, "stage4b_ierc_best.pth")

    if args.mode in ["before_supcon", "full_comparison"]:
        print("\n--- 1. Untrained Baseline (Random Weights) ---")
        regimes["1. Untrained Encoder (Random Baseline)"] = run_regime("Untrained", "", is_random=True)
        checkpoints["1. Untrained Encoder (Random Baseline)"] = "random_init"

    if args.mode in ["post_supcon", "full_comparison"]:
        print("\n--- 2. Post-SupCon (Novelty 1: CAHN + CrossModal Alignment) ---")
        regimes["2. Stage 4 SupCon (CAHN + InfoNCE)"] = run_regime("Post-SupCon", stage4_ckpt, is_random=False)
        checkpoints["2. Stage 4 SupCon (CAHN + InfoNCE)"] = stage4_ckpt

    if args.mode in ["post_recluster", "full_comparison"]:
        print("\n--- 3. Post-IERC (Novelty 2: Iterative Re-Clustering) ---")
        regimes["3. Stage 4b Re-Clustered (IERC Fine-Tuned)"] = run_regime("Post-IERC", stage4b_ckpt, is_random=False)
        checkpoints["3. Stage 4b Re-Clustered (IERC Fine-Tuned)"] = stage4b_ckpt

    rows = []
    for regime_name, res in regimes.items():
        if res is not None:
            r1 = res["correlations"][0] if len(res["correlations"]) > 0 else np.nan
            r_mean = float(np.mean(res["correlations"])) if len(res["correlations"]) > 0 else np.nan
            rows.append(
                {
                    "Evaluation Regime": regime_name,
                    "Canonical Corr (r₁) (↑)": round(r1, 4),
                    "Mean Corr (r̄) (↑)": round(r_mean, 4),
                    "Permutation Null (Mean±Std)": f"{res['null_mean']:.3f} ± {res['null_std']:.3f}",
                    "Alignment Gap (Δ_null)": round(res["gap"], 4),
                    "z-score (Δ_null/σ_null)": round(res["z_score"], 2),
                    "p-value": round(res["p_value"], 5),
                    "Significance": res["verdict"],
                }
            )

            # <-- ADDED: log this run so it can be compared against future ones
            log_cca_result(
                stage=regime_name,
                r1=r1,
                r_bar=r_mean,
                null_mean=res["null_mean"],
                null_std=res["null_std"],
                seed=SEED,
                checkpoint=checkpoints[regime_name],
            )

    if rows:
        results_df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("   CROSS-MODAL LATENT ALIGNMENT COMPARISON (CCA BENCHMARK)")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        results_df.to_csv("cca_comparison_results.csv", index=False)
        print("\nSaved CCA benchmark results to `cca_comparison_results.csv`")


if __name__ == "__main__":
    main()