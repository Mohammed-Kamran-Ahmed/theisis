"""
cca_analysis.py - Canonical Correlation Analysis (CCA) Benchmark with Permutation Null Test
===========================================================================================
Quantifies linear cross-modal alignment between SAR embeddings (z_sar) and Optical embeddings (z_opt).
Calculates:
- First canonical correlation coefficient (r_1)
- Mean canonical correlation over top-k components (r_bar)
- Null distribution via 200 random permutations
- Alignment gain over null: Delta_null = r_1 - mean(r_1_null)
- Empirical p-value
"""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA


def run_cca_analysis(
    z_sar: np.ndarray,
    z_opt: np.ndarray,
    n_components: int = 8,
    n_permutations: int = 200,
    seed: int = 42
) -> dict:
    """
    Args:
        z_sar: (N, D) SAR feature embeddings.
        z_opt: (N, D) Optical feature embeddings.
        n_components: Number of CCA dimensions to compute.
        n_permutations: Number of permutation runs for null distribution.
        seed: Random seed for reproducibility.
    Returns:
        dict with r1, r_bar, r_null_mean, delta_null, p_value, and all canonical correlations.
    """
    np.random.seed(seed)
    N, D = z_sar.shape

    # ── PCA pre-reduction ──────────────────────────────────────────────────────
    # CCA requires N >> d.  Rule of thumb: N ≥ 10 × cca_dim.
    # We cap cca_dim so that condition holds, then reduce both views with PCA
    # before passing to sklearn's CCA.  This prevents degenerate r₁ = 1.0
    # when the embedding dim (128) is close to N (375 test samples).
    cca_dim = max(1, min(n_components, D, N // 10))
    if D > cca_dim:
        pca_sar = PCA(n_components=cca_dim, random_state=seed)
        pca_opt = PCA(n_components=cca_dim, random_state=seed)
        z_sar_red = pca_sar.fit_transform(z_sar)
        z_opt_red = pca_opt.fit_transform(z_opt)
    else:
        z_sar_red = z_sar.copy()
        z_opt_red = z_opt.copy()
        cca_dim = min(n_components, D, N - 1)

    # Standardize inputs
    z_sar_norm = (z_sar_red - np.mean(z_sar_red, axis=0, keepdims=True)) / (np.std(z_sar_red, axis=0, keepdims=True) + 1e-8)
    z_opt_norm = (z_opt_red - np.mean(z_opt_red, axis=0, keepdims=True)) / (np.std(z_opt_red, axis=0, keepdims=True) + 1e-8)

    # Fit true CCA
    cca = CCA(n_components=cca_dim, max_iter=2000)
    sar_c, opt_c = cca.fit_transform(z_sar_norm, z_opt_norm)

    # Compute canonical correlations (up to the actual fitted cca_dim)
    correlations = []
    for i in range(cca_dim):
        corr = np.corrcoef(sar_c[:, i], opt_c[:, i])[0, 1]
        correlations.append(float(corr))

    r1 = correlations[0]
    r_bar = float(np.mean(correlations))

    # Permutation test for null distribution (uses already-reduced+standardized views)
    null_r1_list = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(N)
        z_opt_perm = z_opt_norm[perm_idx]

        cca_null = CCA(n_components=1, max_iter=1000)
        s_c_null, o_c_null = cca_null.fit_transform(z_sar_norm, z_opt_perm)
        null_r1 = np.corrcoef(s_c_null[:, 0], o_c_null[:, 0])[0, 1]
        null_r1_list.append(float(null_r1))

    r1_null_mean = float(np.mean(null_r1_list))
    r1_null_std = float(np.std(null_r1_list))
    delta_null = r1 - r1_null_mean
    p_value = float(np.sum(np.array(null_r1_list) >= r1) + 1) / (n_permutations + 1)

    results = {
        "r1": r1,
        "r_bar": r_bar,
        "r1_null_mean": r1_null_mean,
        "r1_null_std": r1_null_std,
        "delta_null": delta_null,
        "p_value": p_value,
        "all_canonical_correlations": correlations,
        "n_samples": N,
        "cca_dim_used": cca_dim,          # actual dim after PCA reduction
        "n_components_requested": n_components,
        "n_permutations": n_permutations
    }
    return results

