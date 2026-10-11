"""
ARI/NMI against REAL ground-truth labels (DFC2020), not self-consistency.

This is the upgrade over compute_ari.py's original use case: that script compares
your hand-crafted-GMM labels against your embedding-GMM labels (internal
consistency -- do two of YOUR OWN clusterings agree with each other). This
script instead compares your cluster assignments against DFC2020's real,
human-verified 8-class labels (external validity -- does your unsupervised
structure actually correspond to real land-cover classes).

Usage:
    python ari_vs_ground_truth.py \
        --metadata patch_metadata.csv \
        --cluster_assignments your_gmm_clusters.npy \
        --export_names_order export_names_in_cluster_order.npy

Where `your_gmm_clusters.npy` is the array of cluster ids your GMM (hand-crafted
or embedding-space/IERC) assigned, and `export_names_order.npy` is a parallel
array of the `export_name` string for each row, so this script can align your
cluster ids to the right patch_metadata.csv row's `true_label` even if your
training dataloader's iteration order doesn't match the CSV's row order.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix


def align_and_score(metadata_csv: str, cluster_assignments: np.ndarray, export_names: np.ndarray) -> dict:
    meta = pd.read_csv(metadata_csv)
    meta = meta[meta["true_label"].notna()].copy()
    meta["true_label"] = meta["true_label"].astype(int)
    label_lookup = dict(zip(meta["export_name"], meta["true_label"]))

    aligned_pred, aligned_true, dropped = [], [], 0
    for cluster_id, name in zip(cluster_assignments, export_names):
        if name in label_lookup:
            aligned_pred.append(int(cluster_id))
            aligned_true.append(label_lookup[name])
        else:
            dropped += 1  # patch not in metadata (e.g. a GEE-collected patch with no true_label)

    aligned_pred = np.array(aligned_pred)
    aligned_true = np.array(aligned_true)

    if len(aligned_pred) == 0:
        raise ValueError(
            "No overlap between cluster_assignments' export_names and metadata_csv's "
            "export_name column with a non-null true_label. Check that export_names_order.npy "
            "actually corresponds to DFC2020 patches, not your GEE-collected ones (which have "
            "no true_label)."
        )

    ari = adjusted_rand_score(aligned_true, aligned_pred)
    nmi = normalized_mutual_info_score(aligned_true, aligned_pred)
    cm = confusion_matrix(aligned_true, aligned_pred)

    n_true_classes = len(np.unique(aligned_true))
    n_pred_clusters = len(np.unique(aligned_pred))

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "n_samples_scored": int(len(aligned_pred)),
        "n_samples_dropped_no_ground_truth": int(dropped),
        "n_true_classes": int(n_true_classes),
        "n_pred_clusters": int(n_pred_clusters),
        "confusion_matrix": cm.tolist(),
        "interpretation": _interpret(ari, n_true_classes, n_pred_clusters),
    }


def _interpret(ari: float, n_true: int, n_pred: int) -> str:
    note = ""
    if n_true != n_pred:
        note = (f" Note: you're comparing {n_pred} learned clusters against {n_true} real "
                f"DFC2020 classes -- a mismatch in cluster count is expected (your GMM's K "
                f"doesn't have to equal DFC2020's 8 classes) and ARI already accounts for this, "
                f"but it's worth reporting both numbers alongside ARI so a reader isn't confused.")
    if ari > 0.5:
        return ("Strong external validity: your unsupervised clusters correspond well to real "
                "land-cover classes. This is a much stronger claim than internal consistency "
                "alone -- report this as direct evidence the embedding space captures genuine "
                "land-cover structure, not just an arbitrary partition." + note)
    if ari > 0.25:
        return ("Moderate external validity: meaningful but partial correspondence to real "
                "classes -- reasonable for an unsupervised method (supervised classifiers on "
                "DFC2020 typically score well above this, but they see the labels during "
                "training; your GMM never does)." + note)
    return ("Weak external validity by this measure. This is not unusual for K-means/GMM-style "
            "clustering compared against fine-grained real classes -- consider also reporting "
            "per-class purity/recall from the confusion matrix, since ARI penalizes cluster-count "
            "mismatches harshly and a few real classes (e.g. Wetlands, Barren, per the DFC2020 "
            "paper) are known to be hard to separate even for supervised models." + note)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="patch_metadata.csv from dfc2020_loader.py")
    p.add_argument("--cluster_assignments", required=True, help="npy (N,) of your GMM cluster ids")
    p.add_argument("--export_names_order", required=True,
                    help="npy (N,) of export_name strings, same order as cluster_assignments")
    p.add_argument("--out_dir", default="./ari_vs_ground_truth_results")
    args = p.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    cluster_assignments = np.load(args.cluster_assignments, allow_pickle=True)
    export_names = np.load(args.export_names_order, allow_pickle=True)

    result = align_and_score(args.metadata, cluster_assignments, export_names)

    out_path = os.path.join(args.out_dir, "ari_vs_ground_truth_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"ARI vs real DFC2020 labels = {result['ari']:.4f}")
    print(f"NMI vs real DFC2020 labels = {result['nmi']:.4f}")
    print(f"Scored {result['n_samples_scored']} patches "
          f"({result['n_samples_dropped_no_ground_truth']} dropped: no ground truth available)")
    print(result["interpretation"])
    print(f"Full result -> {out_path}")


if __name__ == "__main__":
    main()
