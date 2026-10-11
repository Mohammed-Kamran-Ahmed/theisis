"""
visualize_spatial_features.py
================================
Produces a "spatial feature showcase" figure for the paper: for several
test patches, shows what the frozen SAR encoder's intermediate layers
(f1..f4) actually represent spatially, via PCA-projected false-color maps.

This is COMPLETELY INDEPENDENT of the reconstruction decoder pipeline
(train_spatial_decoder.py / SpatialUNetDecoder) -- it doesn't need that
decoder to work well, or at all. It only needs the already-trained,
frozen sar_encoder.pth, which you already have.

Why this is useful even though the reconstruction attempt underperformed:
it lets you SHOW, not just claim, what each layer is doing -- e.g. early
layers (f1) tend to capture fine texture/edges, later layers (f4) tend to
capture coarse, semantic, low-spatial-resolution patterns. If f4 looks
blocky/low-detail in the figure, that's direct visual evidence supporting
the written explanation for why the reconstruction decoder struggled --
turning a negative result into a properly diagnosed, illustrated one.

Method: for each layer, mean-center the channel vectors pooled across
many spatial locations from several TRAIN patches, take the top-3
principal components via SVD (implemented with plain numpy, no sklearn
dependency required), then project each pixel's feature vector onto
those 3 components and min-max normalize to produce an RGB-like image.
This is a standard feature-visualization technique, not a new proposal.

Usage:
    python visualize_spatial_features.py
    python visualize_spatial_features.py --n_patches 6
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import numpy as np
import pandas as pd
import rasterio
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train_spatial_decoder import INPUT_CSV, PATCH_DIR, SAR_ENCODER_PATH, SAR_BAND_INDICES
from models.sar_encoder_spatial import load_frozen_sar_encoder
import torch.nn.functional as F

TARGET_SIZE = (64, 64)
LAYER_NAMES = ["f1 (early -- fine texture)", "f2", "f3", "f4 (late -- coarse/semantic)"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(row, patch_folder):
    for col in ["local_path", "filepath"]:
        if col in row and pd.notna(row[col]):
            import os
            if os.path.exists(str(row[col])):
                return str(row[col])
    import os
    if "export_name" in row and pd.notna(row["export_name"]):
        candidate = os.path.join(patch_folder, f"{row['export_name']}.tif")
        if os.path.exists(candidate):
            return candidate
    return None


def standardize(arr):
    tensor = torch.from_numpy(arr).float()
    for c in range(tensor.shape[0]):
        std = tensor[c].std()
        mean = tensor[c].mean()
        if std > 1e-6:
            tensor[c] = (tensor[c] - mean) / std
        else:
            tensor[c] = tensor[c] - mean
    return tensor


def load_sar_tensor(path):
    with rasterio.open(path) as src:
        img = np.nan_to_num(src.read(), nan=0.0, posinf=0.0, neginf=0.0)
    sar_arr = img[SAR_BAND_INDICES, :, :]
    vv_band = img[0, :, :]  # raw VV, for display
    sar_tensor = standardize(sar_arr)
    sar_tensor = F.interpolate(sar_tensor.unsqueeze(0), size=TARGET_SIZE,
                                mode="bilinear", align_corners=False).squeeze(0)
    return sar_tensor, vv_band


def vv_display(vv_band):
    lo, hi = np.percentile(vv_band, [2, 98])
    disp = np.clip((vv_band - lo) / max(hi - lo, 1e-6), 0, 1)
    return disp


def fit_pca_per_layer(sar_encoder, sample_paths, n_components=3):
    """Fits a 3-component PCA (plain SVD) per layer, using channel vectors
    pooled across every spatial location of several sample patches."""
    layer_vectors = {0: [], 1: [], 2: [], 3: []}

    with torch.no_grad():
        for path in sample_paths:
            sar_tensor, _ = load_sar_tensor(path)
            f1, f2, f3, f4, _ = sar_encoder.forward_spatial(sar_tensor.unsqueeze(0).to(DEVICE))
            for i, f in enumerate([f1, f2, f3, f4]):
                f = f.squeeze(0).cpu().numpy()  # (C, H, W)
                c, h, w = f.shape
                vecs = f.reshape(c, h * w).T  # (H*W, C)
                layer_vectors[i].append(vecs)

    pca_components = {}
    pca_means = {}
    for i in range(4):
        all_vecs = np.concatenate(layer_vectors[i], axis=0)  # (N, C)
        mean = all_vecs.mean(axis=0)
        centered = all_vecs - mean
        # top-3 principal directions via SVD
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]  # (3, C)
        pca_components[i] = components
        pca_means[i] = mean

    return pca_components, pca_means


def project_to_rgb(feature_map, components, mean):
    """feature_map: (C, H, W) -> returns (H, W, 3) normalized to [0,1]."""
    c, h, w = feature_map.shape
    vecs = feature_map.reshape(c, h * w).T  # (H*W, C)
    centered = vecs - mean
    projected = centered @ components.T  # (H*W, 3)
    projected = projected.reshape(h, w, 3)
    for k in range(3):
        ch = projected[:, :, k]
        lo, hi = np.percentile(ch, [2, 98])
        projected[:, :, k] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    return projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patches", type=int, default=5, help="how many test patches to show")
    parser.add_argument("--n_fit_patches", type=int, default=20, help="how many train patches to fit PCA on")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    train_df = df[(df["valid"] == True) & (df["split"] == "train")].reset_index(drop=True)
    test_df = df[(df["valid"] == True) & (df["split"] == "test")].reset_index(drop=True)

    print(f"Loading frozen SAR encoder from {SAR_ENCODER_PATH} ...")
    sar_encoder = load_frozen_sar_encoder(SAR_ENCODER_PATH, in_channels=len(SAR_BAND_INDICES), device=DEVICE)

    print(f"Fitting per-layer PCA on {args.n_fit_patches} train patches ...")
    fit_paths = []
    for _, row in train_df.head(args.n_fit_patches).iterrows():
        p = resolve_path(row, PATCH_DIR)
        if p:
            fit_paths.append(p)
    components, means = fit_pca_per_layer(sar_encoder, fit_paths)

    show_rows = test_df.head(args.n_patches)
    n_rows = len(show_rows)

    fig = plt.figure(figsize=(18, 3.2 * n_rows))
    gs = gridspec.GridSpec(n_rows, 5, wspace=0.15, hspace=0.35)

    for row_idx, (_, row) in enumerate(show_rows.iterrows()):
        path = resolve_path(row, PATCH_DIR)
        if path is None:
            continue
        sar_tensor, vv_band = load_sar_tensor(path)

        with torch.no_grad():
            f1, f2, f3, f4, _ = sar_encoder.forward_spatial(sar_tensor.unsqueeze(0).to(DEVICE))

        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(vv_display(vv_band), cmap="gray")
        ax0.set_title("SAR VV input" if row_idx == 0 else "", fontsize=10)
        ax0.set_ylabel(row.get("export_name", f"patch {row_idx}"), fontsize=8)
        ax0.set_xticks([]); ax0.set_yticks([])

        for i, f in enumerate([f1, f2, f3, f4]):
            f_np = f.squeeze(0).cpu().numpy()
            rgb = project_to_rgb(f_np, components[i], means[i])
            ax = fig.add_subplot(gs[row_idx, i + 1])
            ax.imshow(rgb)
            if row_idx == 0:
                ax.set_title(LAYER_NAMES[i], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Spatial Feature Showcase: PCA-projected SAR encoder activations by layer\n"
                  "(Left: raw SAR input. Right: how spatial detail coarsens from f1 -> f4)",
                  fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("spatial_feature_showcase.png", dpi=150, bbox_inches="tight")
    print("Saved spatial_feature_showcase.png")
    plt.show()


if __name__ == "__main__":
    main()