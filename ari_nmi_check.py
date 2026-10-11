"""
ari_nmi_check.py
=================
Compares the ORIGINAL hand-crafted-feature GMM clusters (Section 3 of the
paper, computed before any contrastive training) against the FINAL
IERC embedding-space clusters (Section 5.2), on the same patches.

This is the check the paper's Limitations section calls "pending."
It tells you whether IERC's re-clustering discovered a meaningfully
different/better structure, or just relabeled the same 454+114
patches into different, equally-arbitrary groups.

Run this AFTER stage4b_recluster_and_finetune.py has produced
patch_metadata_reclustered.csv.

    python ari_nmi_check.py
"""
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RECLUSTERED_CSV = "./patch_metadata_reclustered.csv"

# The original hand-crafted-feature GMM label column name may differ in
# your CSV -- check with `df.columns.tolist()` if this KeyErrors, and
# adjust ORIGINAL_COL below. Common candidates: "gmm_cluster", "original_pseudo_label".
ORIGINAL_COL = "gmm_cluster"
IERC_COL = "pseudo_label"


def main():
    df = pd.read_csv(RECLUSTERED_CSV)
    print(f"Loaded {len(df)} patches from {RECLUSTERED_CSV}")
    print(f"Columns available: {df.columns.tolist()}\n")

    if ORIGINAL_COL not in df.columns:
        print(f"❌ Column '{ORIGINAL_COL}' not found. Open the CSV and check the "
              f"actual column name for the ORIGINAL hand-crafted GMM label, "
              f"then edit ORIGINAL_COL at the top of this script.")
        return
    if IERC_COL not in df.columns:
        print(f"❌ Column '{IERC_COL}' not found for the IERC label.")
        return

    clean = df.dropna(subset=[ORIGINAL_COL, IERC_COL])
    original_labels = clean[ORIGINAL_COL].astype(int).values
    ierc_labels = clean[IERC_COL].astype(int).values

    ari = adjusted_rand_score(original_labels, ierc_labels)
    nmi = normalized_mutual_info_score(original_labels, ierc_labels)

    print("=" * 60)
    print("   ARI / NMI: Original Hand-Crafted GMM vs IERC Clusters")
    print("=" * 60)
    print(f"n patches compared : {len(clean)}")
    print(f"Adjusted Rand Index: {ari:.4f}   (target > 0.4 per paper's Sec 9)")
    print(f"Normalized Mutual Info: {nmi:.4f}")
    print("=" * 60)

    if ari < 0.1:
        verdict = ("ARI is near zero: IERC's clusters are essentially "
                    "UNRELATED to the original hand-crafted clusters. This "
                    "supports the circularity concern -- IERC is producing a "
                    "different labeling, not obviously a more meaningful one, "
                    "at least relative to the original physical-feature clusters.")
    elif ari < 0.4:
        verdict = ("ARI shows weak-to-moderate agreement. IERC has partially "
                    "reorganized the clusters relative to the original GMM. "
                    "Report this honestly rather than rounding up to \"validated\".")
    else:
        verdict = ("ARI clears the paper's own >0.4 target: IERC's re-clustering "
                    "retains meaningful correspondence with the original "
                    "physical-feature clusters while presumably improving "
                    "cross-modal structure. This is evidence IERC is doing "
                    "more than relabeling noise.")

    print(f"\n{verdict}")

    # Save for citing exact numbers in the paper.
    with open("ari_nmi_result.txt", "w") as f:
        f.write(f"n={len(clean)}\nARI={ari:.4f}\nNMI={nmi:.4f}\n{verdict}\n")
    print("\nSaved to ari_nmi_result.txt")


if __name__ == "__main__":
    main()