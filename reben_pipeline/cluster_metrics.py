"""
cluster_metrics.py - Latent Clustering & Embedding Quality Metrics
==================================================================
Computes:
1. Silhouette Coefficient (higher is better, in [-1, 1])
2. Davies-Bouldin Index (lower is better, >= 0)
3. Calinski-Harabasz Index / Variance Ratio Criterion (higher is better)
4. Adjusted Rand Index (ARI) against ground truth or initial pseudo-labels
5. Normalized Mutual Information (NMI)
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def compute_clustering_metrics(embeddings: np.ndarray, cluster_labels: np.ndarray, reference_labels: np.ndarray = None) -> dict:
    """
    Args:
        embeddings: (N, D) latent feature vectors.
        cluster_labels: (N,) predicted/assigned cluster IDs.
        reference_labels: (N,) optional ground-truth or baseline cluster IDs for ARI/NMI.
    Returns:
        dict with silhouette, davies_bouldin, calinski_harabasz, ari, nmi.
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2 or n_clusters >= len(embeddings):
        return {
            "silhouette": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
            "n_clusters": n_clusters,
            "ari": 0.0,
            "nmi": 0.0
        }

    # Clean any NaNs or infinities
    X = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Subsample for silhouette if dataset is very large to maintain fast evaluation
    if len(X) > 10000:
        sample_idx = np.random.choice(len(X), size=10000, replace=False)
        sil = float(silhouette_score(X[sample_idx], cluster_labels[sample_idx]))
    else:
        sil = float(silhouette_score(X, cluster_labels))

    db_score = float(davies_bouldin_score(X, cluster_labels))
    ch_score = float(calinski_harabasz_score(X, cluster_labels))

    res = {
        "silhouette": sil,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
        "n_clusters": int(n_clusters),
    }

    if reference_labels is not None:
        ref_clean = reference_labels
        if ref_clean.ndim > 1:
            # If multi-label binary matrix, convert to dominant single label / argmax or string signature
            ref_clean = np.argmax(ref_clean, axis=1)
        res["ari"] = float(adjusted_rand_score(ref_clean, cluster_labels))
        res["nmi"] = float(normalized_mutual_info_score(ref_clean, cluster_labels))

    return res
