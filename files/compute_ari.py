"""
Adjusted Rand Index (ARI) consistency check.

Paper Section 9, item (ii): "computation of the Adjusted Rand Index between
hand-crafted-feature GMM clusters and embedding-space GMM clusters (IERC's
originally proposed pseudo-label consistency check, target ARI > 0.4)."

This is cheap to run — no training, just clustering + one metric.

Usage:
    python compute_ari.py --handcrafted_labels handcrafted_gmm_labels.npy \
        --embedding_labels embedding_gmm_labels_ierc.npy \
        --out_dir ./ari_results

If you haven't saved the label arrays yet, use `refit_and_compare()` below —
point it at your original hand-crafted SAR feature matrix and your IERC'd
64-dim projection embeddings, and it refits both GMMs and computes ARI in
one call.

NOTE: this script compares your OWN two clusterings against each other
(internal self-consistency). For comparison against DFC2020's real ground
truth labels, use ari_vs_ground_truth.py instead — that's the stronger claim.
"""
import argparse
import json
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)


def refit_and_compare(handcrafted_features: np.ndarray,
                       projection_embeddings: np.ndarray,
                       k: int = 8,
                       random_state: int = 42) -> dict:
    """Refit both GMMs fresh (same K, same seed policy) and compare cluster assignments.

    handcrafted_features   : (N, 5)  -- VV, VH, VH/VV, texture/entropy, RVI (Section 3)
    projection_embeddings  : (N, 64) -- post-IERC z embeddings (Section 5.2)
    """
    gmm_hc = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
    labels_hc = gmm_hc.fit_predict(handcrafted_features)

    gmm_emb = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
    labels_emb = gmm_emb.fit_predict(projection_embeddings)

    return _score(labels_hc, labels_emb)


def _score(labels_hc: np.ndarray, labels_emb: np.ndarray) -> dict:
    ari = adjusted_rand_score(labels_hc, labels_emb)
    nmi = normalized_mutual_info_score(labels_hc, labels_emb)
    cm = confusion_matrix(labels_hc, labels_emb)

    result = {
        "ari": float(ari),
        "nmi": float(nmi),
        "n_samples": int(len(labels_hc)),
        "confusion_matrix": cm.tolist(),
        "target_met": bool(ari > 0.4),
        "interpretation": _interpret(ari),
    }
    return result


def _interpret(ari: float) -> str:
    if ari > 0.6:
        return ("Strong consistency: IERC's re-clustering largely preserves the "
                "original hand-crafted cluster structure while sharpening it — "
                "supports the claim that IERC refines rather than replaces the "
                "pseudo-label geometry.")
    if ari > 0.4:
        return ("Moderate-to-strong consistency: clears the paper's stated target "
                "(ARI > 0.4). Report this as confirming the pseudo-label "
                "consistency check.")
    if ari > 0.2:
        return ("Weak-to-moderate consistency: IERC has meaningfully reorganized "
                "the clusters relative to the hand-crafted GMM. This is not "
                "necessarily bad — it's consistent with your own hypothesis (§5.2) "
                "that the hand-crafted feature space doesn't align with the "
                "geometry the encoder actually learns — but you should report it "
                "as a genuine re-organization, not a 'consistency confirmation', "
                "and discuss it honestly rather than treating >0.4 as pass/fail.")
    return ("Low consistency: the two clusterings are close to unrelated. Worth "
            "sanity-checking cluster count K and random_state/n_init stability "
            "before reporting this number — rerun refit_and_compare with a few "
            "different random_state seeds and report the mean +/- std, since GMM "
            "fits can be seed-sensitive with K=8 on only 454 samples.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--handcrafted_labels", default=None,
                    help="npy of pre-computed hand-crafted GMM cluster labels (N,)")
    p.add_argument("--embedding_labels", default=None,
                    help="npy of pre-computed IERC projection-space GMM cluster labels (N,)")
    p.add_argument("--handcrafted_features", default=None,
                    help="npy of raw hand-crafted SAR features (N,5) to refit fresh instead")
    p.add_argument("--projection_embeddings", default=None,
                    help="npy of raw IERC 64-dim embeddings (N,64) to refit fresh instead")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out_dir", default="./ari_results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.handcrafted_labels and args.embedding_labels:
        labels_hc = np.load(args.handcrafted_labels)
        labels_emb = np.load(args.embedding_labels)
        result = _score(labels_hc, labels_emb)
    elif args.handcrafted_features and args.projection_embeddings:
        hc = np.load(args.handcrafted_features)
        emb = np.load(args.projection_embeddings)
        result = refit_and_compare(hc, emb, k=args.k)
    else:
        raise ValueError(
            "Provide either (--handcrafted_labels, --embedding_labels) if you already "
            "have saved cluster assignments, or (--handcrafted_features, "
            "--projection_embeddings) to refit both GMMs fresh."
        )

    out_path = os.path.join(args.out_dir, "ari_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"ARI = {result['ari']:.4f}  (target > 0.4: {'PASS' if result['target_met'] else 'FAIL'})")
    print(f"NMI = {result['nmi']:.4f}")
    print(result["interpretation"])
    print(f"Full result written to {out_path}")


if __name__ == "__main__":
    main()
