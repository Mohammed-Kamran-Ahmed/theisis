"""
Adjusted Rand Index (ARI) consistency check between:
  (a) hand-crafted-feature GMM clusters (Stage A, Section 3.1/3.2 of the paper)
  (b) embedding-space GMM clusters after IERC (Section 5.2)

This is the pending item referenced in Table 2 / Section 9 ("ARI consistency
check pending", target ARI > 0.4).

USAGE
-----
You already have both label sets sitting around from earlier stages — you do
NOT need to retrain anything. You just need to save them to disk once and
point this script at them.

    from sklearn.metrics import adjusted_rand_score
    # right after you fit the Stage-A GMM on hand-crafted features:
    np.save("handcrafted_labels.npy", handcrafted_gmm.predict(X_handcrafted))
    # right after IERC re-fits the GMM on the 64-d projection embeddings z:
    np.save("embedding_labels.npy", ierc_gmm.predict(Z_embeddings))

Then run:
    python compute_ari.py --handcrafted handcrafted_labels.npy --embedding embedding_labels.npy

If you evaluate train and test splits separately (recommended, to match the
rest of the paper's train/test reporting convention), run it twice with
--split train / --split test and report both numbers in a small ARI table
next to Table 3.
"""

import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def compute_ari_report(labels_a: np.ndarray, labels_b: np.ndarray, split_name: str = ""):
    assert labels_a.shape[0] == labels_b.shape[0], (
        f"Label arrays must be same length (got {labels_a.shape[0]} vs {labels_b.shape[0]}). "
        "Make sure both label sets are aligned to the same patch ordering."
    )

    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    cm = contingency_matrix(labels_a, labels_b)

    print(f"{'=== ' + split_name + ' split ===' if split_name else '=== ARI report ==='}")
    print(f"N patches            : {labels_a.shape[0]}")
    print(f"Hand-crafted clusters : {len(np.unique(labels_a))}")
    print(f"Embedding clusters    : {len(np.unique(labels_b))}")
    print(f"Adjusted Rand Index   : {ari:.4f}   (target > 0.4 per paper's stated goal)")
    print(f"Normalized MI         : {nmi:.4f}   (supplementary, not required by paper)")
    print(f"Verdict               : {'CONSISTENT' if ari > 0.4 else 'INCONSISTENT / weak agreement'} "
          f"with hand-crafted pseudo-labels")
    print()
    print("Contingency matrix (rows=hand-crafted cluster, cols=embedding cluster):")
    print(cm)
    print()
    return ari, nmi, cm


def main():
    parser = argparse.ArgumentParser(description="Compute ARI between hand-crafted and embedding-space GMM clusters")
    parser.add_argument("--handcrafted", required=True, help="Path to .npy file of hand-crafted GMM cluster labels")
    parser.add_argument("--embedding", required=True, help="Path to .npy file of embedding-space (IERC) GMM cluster labels")
    parser.add_argument("--split", default="", help="Optional split name for logging, e.g. 'train' or 'test'")
    args = parser.parse_args()

    labels_a = np.load(args.handcrafted)
    labels_b = np.load(args.embedding)
    compute_ari_report(labels_a, labels_b, split_name=args.split)


if __name__ == "__main__":
    main()