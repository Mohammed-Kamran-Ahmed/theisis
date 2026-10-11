"""
generate_output_products.py
=============================
Produces TWO real visual deliverables from the trained decoders,
evaluated on the held-out test split:

1. multi_patch_comparison.png
   A grid of rows, each row = one test patch:
   [SAR input | Generated optical | Ground-truth optical | feature bar chart]
   Deliberately includes patches from your BEST cluster (Cluster 1, sparse
   canopy) and WORST cluster (Cluster 0, dense canopy) side by side, so the
   figure itself visually demonstrates the cluster-conditional finding --
   not just one cherry-picked patch.

2. prediction_accuracy_scatter.png
   Predicted vs. actual NDVI_mean across ALL 35 test patches, points
   colored by GMM cluster. This is the figure that shows WHERE the
   decoder is reliable and where it isn't, in one glance -- the real
   "output product" for your defense, not a single error number.

Run AFTER train_decoders.py has produced feature_decoder.pth,
image_decoder.pth, and image_norm_stats.json.
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train_decoders import ThesisRealDataset, NORM_STATS_PATH, TARGET_FEATURE_COLUMNS
from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder

N_PATCHES_TO_SHOW = 6  # how many rows in the comparison grid


def load_models(z_dim, n_features):
    feat_model = PhysicalFeatureDecoder(latent_dim=z_dim, num_features=n_features)
    img_model = OpticalImageDecoder(latent_dim=z_dim, out_channels=3)
    feat_model.load_state_dict(torch.load("feature_decoder.pth"))
    img_model.load_state_dict(torch.load("image_decoder.pth"))
    feat_model.eval()
    img_model.eval()
    return feat_model, img_model


def get_cluster_labels(test_dataset, metadata_csv):
    meta_df = pd.read_csv(metadata_csv)
    meta_indexed = meta_df.set_index("export_name")
    if "cluster" not in meta_indexed.columns:
        print("[warn] no `cluster` column found in metadata -- comparison grid "
              "will not be able to pick 'best' vs 'worst' cluster patches, "
              "falling back to the first N test patches instead.")
        return None
    return meta_indexed.loc[test_dataset.patch_ids, "cluster"].values


def build_comparison_grid(test_dataset, feat_model, img_model, clusters,
                           best_cluster=1, worst_cluster=0):
    with torch.no_grad():
        z_all = test_dataset.z_data
        pred_feats_all = feat_model(z_all).numpy()
        pred_imgs_all = img_model(z_all).numpy()

    true_feats_all = test_dataset.features.numpy()

    # Pick a mix: half from the best cluster, half from the worst, if available.
    indices = list(range(len(test_dataset)))
    if clusters is not None:
        best_idx = [i for i in indices if clusters[i] == best_cluster]
        worst_idx = [i for i in indices if clusters[i] == worst_cluster]
        half = N_PATCHES_TO_SHOW // 2
        chosen = best_idx[:half] + worst_idx[:N_PATCHES_TO_SHOW - half]
        if len(chosen) < N_PATCHES_TO_SHOW:
            remaining = [i for i in indices if i not in chosen]
            chosen += remaining[: N_PATCHES_TO_SHOW - len(chosen)]
    else:
        chosen = indices[:N_PATCHES_TO_SHOW]

    n_rows = len(chosen)
    fig = plt.figure(figsize=(16, 3.2 * n_rows))
    gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 1, 1, 1.4])

    for row, idx in enumerate(chosen):
        z, feat, sar_img_dummy = test_dataset[idx]  # sar_img_dummy unused; we don't have SAR image directly here
        gen_img = np.transpose(pred_imgs_all[idx], (1, 2, 0))
        true_img = np.transpose(test_dataset[idx][2].numpy(), (1, 2, 0))

        cluster_label = f"cluster {clusters[idx]}" if clusters is not None else f"patch {idx}"
        tag = " (BEST cluster)" if clusters is not None and clusters[idx] == best_cluster else \
              (" (WORST cluster)" if clusters is not None and clusters[idx] == worst_cluster else "")

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_title(f"{cluster_label}{tag}\nPatch id: {test_dataset.patch_ids[idx]}", fontsize=9)
        ax0.axis("off")
        ax0.text(0.5, 0.5, "SAR input\n(see stage-5 patch viewer\nfor VV/VH raster)",
                  ha="center", va="center", fontsize=8)

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(np.clip(gen_img, 0, 1))
        ax1.set_title("Generated optical", fontsize=9)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(np.clip(true_img, 0, 1))
        ax2.set_title("Ground-truth optical", fontsize=9)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[row, 3])
        x = np.arange(len(TARGET_FEATURE_COLUMNS))
        width = 0.35
        ax3.bar(x - width/2, true_feats_all[idx], width, label="Actual", color="#4C72B0")
        ax3.bar(x + width/2, pred_feats_all[idx], width, label="Predicted", color="#DD8452")
        ax3.set_xticks(x)
        ax3.set_xticklabels([c.replace("_mean", "") for c in TARGET_FEATURE_COLUMNS], fontsize=7, rotation=30)
        ax3.set_title("Physical features: actual vs predicted", fontsize=9)
        if row == 0:
            ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("multi_patch_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved multi_patch_comparison.png")
    plt.close()


def build_accuracy_scatter(test_dataset, feat_model, clusters, target_idx=0):
    """target_idx picks which of TARGET_FEATURE_COLUMNS to plot (0 = NDVI_mean)."""
    with torch.no_grad():
        pred_feats_all = feat_model(test_dataset.z_data).numpy()
    true_feats_all = test_dataset.features.numpy()

    true_vals = true_feats_all[:, target_idx]
    pred_vals = pred_feats_all[:, target_idx]
    feature_name = TARGET_FEATURE_COLUMNS[target_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    if clusters is not None:
        scatter = ax.scatter(true_vals, pred_vals, c=clusters, cmap="tab10", s=90, edgecolor="k", alpha=0.85)
        legend1 = ax.legend(*scatter.legend_elements(), title="GMM Cluster", loc="upper left", fontsize=8)
        ax.add_artist(legend1)
    else:
        ax.scatter(true_vals, pred_vals, s=90, edgecolor="k", alpha=0.85)

    lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"Actual {feature_name}")
    ax.set_ylabel(f"Predicted {feature_name}")
    ax.set_title(f"Decoder prediction accuracy across all {len(test_dataset)} test patches\n"
                 f"(points near the dashed line = accurate; scattered = unreliable)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("prediction_accuracy_scatter.png", dpi=150, bbox_inches="tight")
    print("Saved prediction_accuracy_scatter.png")
    plt.close()


def main():
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    test_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="test",
        norm_stats=norm_stats,
    )

    feat_model, img_model = load_models(
        z_dim=test_dataset.z_data.shape[1],
        n_features=test_dataset.features.shape[1],
    )

    clusters = get_cluster_labels(test_dataset, "patch_metadata_clustered.csv")

    build_comparison_grid(test_dataset, feat_model, img_model, clusters)
    build_accuracy_scatter(test_dataset, feat_model, clusters, target_idx=0)

    print("\nDone. Two output products created:")
    print("  1. multi_patch_comparison.png    -- visual SAR->optical + feature comparison, best vs worst cluster")
    print("  2. prediction_accuracy_scatter.png -- where the decoder is reliable vs not, across all test patches")


if __name__ == "__main__":
    main()