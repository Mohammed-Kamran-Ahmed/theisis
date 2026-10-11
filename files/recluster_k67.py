"""
Fix the cluster-imbalance problem (Table 1: 22.2x imbalance, Cluster 3 n=5).

Two complementary strategies, both implemented below:
  A) Refit GMM at K=6 and K=7 and check whether the BIC/AIC elbow argument
     (Fig. 1 in the paper) still holds, and whether smallest-cluster size improves.
  B) Post-hoc MERGE: keep K=8 if you want, but iteratively merge any cluster
     below `min_samples` into its nearest neighbor by centroid distance, so
     no surviving cluster has fewer than 20-30 samples. This is often better
     than just lowering K blindly, since it targets the actual problem
     cluster (Cluster 3) instead of changing global cluster count.

Recommended: run both, compare silhouette/CH/DB (Eq 19-21) and re-run the
downstream SupCon + CAHN + IERC pipeline on whichever gives the best balance
of (a) no cluster < 20 samples, (b) BIC/AIC still reasonable, (c) silhouette
on the hand-crafted feature space doesn't collapse.

Usage:
    python recluster_k67.py --features handcrafted_sar_features.npy --out_dir ./recluster_results
"""
import argparse
import json
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist


def fit_gmm(features: np.ndarray, k: int, random_state: int = 42):
    gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=10, covariance_type="full")
    labels = gmm.fit_predict(features)
    bic = gmm.bic(features)
    aic = gmm.aic(features)
    return gmm, labels, bic, aic


def cluster_report(features: np.ndarray, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    report = {
        "n_clusters": int(len(sizes)),
        "cluster_sizes": sizes.tolist(),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "imbalance_ratio": float(sizes.max() / max(sizes.min(), 1)),
    }
    if len(sizes) > 1 and len(set(labels)) > 1:
        report["silhouette"] = float(silhouette_score(features, labels))
        report["calinski_harabasz"] = float(calinski_harabasz_score(features, labels))
        report["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    return report


def strategy_a_lower_k(features: np.ndarray, ks=(6, 7, 8)) -> dict:
    """Refit at lower K, compare against paper's original K=8 result."""
    results = {}
    for k in ks:
        gmm, labels, bic, aic = fit_gmm(features, k)
        rep = cluster_report(features, labels)
        rep["bic"] = float(bic)
        rep["aic"] = float(aic)
        results[f"K={k}"] = rep
        results[f"K={k}"]["labels"] = labels.tolist()
    return results


def strategy_b_merge_small_clusters(features: np.ndarray, labels: np.ndarray,
                                     min_samples: int = 20) -> dict:
    """Iteratively merge the smallest cluster into its nearest-centroid neighbor
    until every surviving cluster has >= min_samples."""
    labels = labels.copy()
    merge_log = []

    while True:
        unique, counts = np.unique(labels, return_counts=True)
        if counts.min() >= min_samples or len(unique) <= 2:
            break
        small_cluster = unique[np.argmin(counts)]

        # compute centroids
        centroids = {c: features[labels == c].mean(axis=0) for c in unique}
        small_centroid = centroids[small_cluster].reshape(1, -1)
        other_ids = [c for c in unique if c != small_cluster]
        other_centroids = np.stack([centroids[c] for c in other_ids])
        dists = cdist(small_centroid, other_centroids)[0]
        nearest = other_ids[int(np.argmin(dists))]

        merge_log.append({
            "merged_cluster": int(small_cluster),
            "merged_size": int(counts.min()),
            "into_cluster": int(nearest),
            "distance": float(dists.min()),
        })
        labels[labels == small_cluster] = nearest

    # relabel to contiguous 0..K-1
    unique_final = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique_final)}
    labels = np.array([remap[l] for l in labels])

    report = cluster_report(features, labels)
    report["merge_log"] = merge_log
    report["labels"] = labels.tolist()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                    help="npy (N,5) hand-crafted SAR features: VV, VH, VH/VV, texture, RVI")
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--out_dir", default="./recluster_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    features = np.load(args.features)

    print("=== Strategy A: lower K ===")
    result_a = strategy_a_lower_k(features, ks=(6, 7, 8))
    for k, rep in result_a.items():
        print(f"{k}: sizes={rep['cluster_sizes']} imbalance={rep['imbalance_ratio']:.1f}x "
              f"silhouette={rep.get('silhouette', float('nan')):.4f} BIC={rep['bic']:.1f}")

    print("\n=== Strategy B: merge small clusters (from original K=8 fit) ===")
    gmm8, labels8, _, _ = fit_gmm(features, k=8)
    result_b = strategy_b_merge_small_clusters(features, labels8, min_samples=args.min_samples)
    print(f"Merged to {result_b['n_clusters']} clusters: sizes={result_b['cluster_sizes']} "
          f"imbalance={result_b['imbalance_ratio']:.1f}x silhouette={result_b.get('silhouette', float('nan')):.4f}")
    print("Merge log:", json.dumps(result_b["merge_log"], indent=2))

    with open(os.path.join(args.out_dir, "strategy_a_lower_k.json"), "w") as f:
        json.dump(result_a, f, indent=2)
    with open(os.path.join(args.out_dir, "strategy_b_merge.json"), "w") as f:
        json.dump(result_b, f, indent=2)

    print(f"\nResults written to {args.out_dir}/. "
          f"Recommendation: pick whichever strategy gives min_cluster_size >= {args.min_samples} "
          f"with the least silhouette degradation vs the original K=8 fit, then re-run "
          f"Stage B (SupCon) + CAHN + IERC with the new pseudo-labels as the supervisory signal.")


if __name__ == "__main__":
    main()
