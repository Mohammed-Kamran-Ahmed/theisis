"""
evaluate_spatial_decoder.py
==============================
Novelty 3 Evaluation: Baseline vs. Proposed Spatially-Conditioned Decoder (SCD)
-------------------------------------------------------------------------------
Evaluates both decoders on the strict, held-out TEST split:
1. Baseline: `OpticalImageDecoder` (global 64-dim bottleneck -> ConvTranspose)
2. Proposed: `SpatiallyConditionedDecoder` (FiLM modulation + U-Net multi-scale skip connections)

Metrics:
- Structural Similarity (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

Outputs:
- Console table ready for LaTeX / thesis inclusion
- `spatial_decoder_benchmark_results.csv`
- Publication-quality side-by-side comparison plot: `spatial_decoder_comparison_figure.png`
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_spatial_decoder import (
    SpatialDecoderDataset,
    SAR_BAND_INDICES,
    RGB_TARGET_INDICES,
    NORM_STATS_PATH,
    PATCH_DIR,
    MODEL_DIR,
)
from models.spatial_decoder import SpatiallyConditionedDecoder, SAREcoderWithSkips
from models.decoders import OpticalImageDecoder
from stage4_contrastive_learning import MultiModalEncoder

try:
    from pytorch_msssim import ssim as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[warn] pytorch-msssim not installed.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAR_ENCODER_PATH = os.path.join(MODEL_DIR, "sar_encoder_reclustered.pth") if os.path.exists(os.path.join(MODEL_DIR, "sar_encoder_reclustered.pth")) else (
    os.path.join(MODEL_DIR, "sar_encoder_best.pth") if os.path.exists(os.path.join(MODEL_DIR, "sar_encoder_best.pth")) else os.path.join(MODEL_DIR, "sar_encoder.pth")
)
SPATIAL_DECODER_PATH = "spatial_image_decoder.pth"
BASELINE_DECODER_PATH = "image_decoder.pth"


def compute_metrics(pred, target):
    """pred, target: [B, 3, H, W] in range [0, 1]"""
    if HAVE_SSIM:
        ssim_val = ssim_fn(pred, target, data_range=1.0).item()
    else:
        ssim_val = float("nan")

    mse_val = torch.mean((pred - target) ** 2).item()
    mae_val = torch.mean(torch.abs(pred - target)).item()
    psnr_val = 10.0 * np.log10(1.0 / max(mse_val, 1e-10))
    return ssim_val, psnr_val, mse_val, mae_val


def evaluate():
    print("=" * 70)
    print("  EVALUATION: BASELINE VS. PROPOSED SPATIALLY-CONDITIONED DECODER")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    csv_path = "./patch_metadata_reclustered.csv" if os.path.exists("./patch_metadata_reclustered.csv") else "./patch_metadata_clustered.csv"
    df = pd.read_csv(csv_path)
    test_df = df[(df["valid"] == True) & (df["split"] == "test")].copy().reset_index(drop=True)
    print(f"Evaluating strictly on {len(test_df)} held-out TEST patches.")

    if not os.path.exists(NORM_STATS_PATH):
        print("❌ Normalization stats missing. Run train_spatial_decoder.py first.")
        return

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    test_dataset = SpatialDecoderDataset(test_df, PATCH_DIR, norm_stats, augment=False)
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 1. Load SAR Encoder
    base_encoder = MultiModalEncoder(in_channels=len(SAR_BAND_INDICES), embed_dim=128, proj_dim=64, pretrained=False)
    base_encoder.load_state_dict(torch.load(SAR_ENCODER_PATH, map_location=DEVICE))
    base_encoder.to(DEVICE)
    base_encoder.eval()

    sar_spatial_encoder = SAREcoderWithSkips(base_encoder).to(DEVICE)
    sar_spatial_encoder.eval()

    # 2. Load Proposed Spatial Decoder
    spatial_decoder = SpatiallyConditionedDecoder(embed_dim=64, out_channels=3).to(DEVICE)
    if os.path.exists(SPATIAL_DECODER_PATH):
        spatial_decoder.load_state_dict(torch.load(SPATIAL_DECODER_PATH, map_location=DEVICE))
        print(f"Loaded proposed spatial decoder from {SPATIAL_DECODER_PATH}")
    else:
        print(f"[warn] {SPATIAL_DECODER_PATH} not found. Running with initialized weights.")
    spatial_decoder.eval()

    # 3. Load Baseline Optical Image Decoder
    baseline_decoder = OpticalImageDecoder(latent_dim=64, out_channels=3).to(DEVICE)
    has_baseline = os.path.exists(BASELINE_DECODER_PATH)
    if has_baseline:
        try:
            baseline_decoder.load_state_dict(torch.load(BASELINE_DECODER_PATH, map_location=DEVICE))
            print(f"Loaded baseline image decoder from {BASELINE_DECODER_PATH}")
        except Exception as e:
            print(f"[warn] Failed to load {BASELINE_DECODER_PATH} ({e}). Initializing fallback.")
            has_baseline = False
    baseline_decoder.eval()

    # Evaluate across test batches
    base_ssims, base_psnrs, base_mses, base_maes = [], [], [], []
    prop_ssims, prop_psnrs, prop_mses, prop_maes = [], [], [], []

    sample_sar = None
    sample_base = None
    sample_prop = None
    sample_gt = None

    with torch.no_grad():
        for i, (sar_b, rgb_b) in enumerate(loader):
            sar_b, rgb_b = sar_b.to(DEVICE), rgb_b.to(DEVICE)

            # Extract representations
            h_sar, z_sar, bottleneck, skip3, skip2, skip1 = sar_spatial_encoder(sar_b)

            # Proposed Spatial Prediction (128x128)
            prop_pred = spatial_decoder(bottleneck, skip3, skip2, skip1, z_sar)

            # Baseline Prediction (interpolated to 128x128 if needed)
            base_pred = baseline_decoder(z_sar)
            if base_pred.shape[-2:] != rgb_b.shape[-2:]:
                base_pred = F.interpolate(base_pred, size=rgb_b.shape[-2:], mode="bilinear", align_corners=False)

            # Compute metrics for proposed
            p_ssim, p_psnr, p_mse, p_mae = compute_metrics(prop_pred, rgb_b)
            prop_ssims.append(p_ssim)
            prop_psnrs.append(p_psnr)
            prop_mses.append(p_mse)
            prop_maes.append(p_mae)

            # Compute metrics for baseline
            b_ssim, b_psnr, b_mse, b_mae = compute_metrics(base_pred, rgb_b)
            base_ssims.append(b_ssim)
            base_psnrs.append(b_psnr)
            base_mses.append(b_mse)
            base_maes.append(b_mae)

            if i == 0:
                sample_sar = sar_b[:4].cpu()
                sample_base = base_pred[:4].cpu()
                sample_prop = prop_pred[:4].cpu()
                sample_gt = rgb_b[:4].cpu()

    # Average metrics
    mean_b_ssim = np.nanmean(base_ssims)
    mean_b_psnr = np.nanmean(base_psnrs)
    mean_b_mse  = np.nanmean(base_mses)
    mean_b_mae  = np.nanmean(base_maes)

    mean_p_ssim = np.nanmean(prop_ssims)
    mean_p_psnr = np.nanmean(prop_psnrs)
    mean_p_mse  = np.nanmean(prop_mses)
    mean_p_mae  = np.nanmean(prop_maes)

    # Summary Benchmark Table
    results_df = pd.DataFrame([
        {
            "Method": "Baseline OpticalImageDecoder (Bottleneck Only)",
            "SSIM (↑)": round(mean_b_ssim, 4),
            "PSNR (dB) (↑)": round(mean_b_psnr, 2),
            "MSE (↓)": round(mean_b_mse, 5),
            "MAE (↓)": round(mean_b_mae, 5),
        },
        {
            "Method": "Proposed Spatially-Conditioned Decoder (SCD + FiLM)",
            "SSIM (↑)": round(mean_p_ssim, 4),
            "PSNR (dB) (↑)": round(mean_p_psnr, 2),
            "MSE (↓)": round(mean_p_mse, 5),
            "MAE (↓)": round(mean_p_mae, 5),
        },
    ])

    print("\n" + "=" * 75)
    print("   QUANTITATIVE TRANSLATION BENCHMARK (TEST SPLIT)")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)
    delta_ssim = (mean_p_ssim - mean_b_ssim) / max(mean_b_ssim, 1e-4) * 100
    delta_psnr = mean_p_psnr - mean_b_psnr
    print(f"Absolute SSIM Improvement : {mean_p_ssim - mean_b_ssim:+.4f} ({delta_ssim:+.1f}%)")
    print(f"Absolute PSNR Improvement : {delta_psnr:+.2f} dB")
    print("=" * 75)

    results_df.to_csv("spatial_decoder_benchmark_results.csv", index=False)
    print("Saved benchmark metrics to `spatial_decoder_benchmark_results.csv`")

    # Render Side-by-Side Visual Comparison (Publication Grade)
    if sample_sar is not None:
        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        titles = ["Input SAR Backscatter", "Baseline Decoder", "Proposed Spatial Decoder", "Sentinel-2 Ground Truth"]

        for row in range(min(4, len(sample_sar))):
            # SAR VV false color
            sar_vv = sample_sar[row, 0].numpy()
            axes[row, 0].imshow(sar_vv, cmap="gray")
            
            # Baseline optical
            b_img = np.clip(sample_base[row].permute(1, 2, 0).numpy(), 0, 1)
            axes[row, 1].imshow(b_img)

            # Proposed optical
            p_img = np.clip(sample_prop[row].permute(1, 2, 0).numpy(), 0, 1)
            axes[row, 2].imshow(p_img)

            # Ground truth
            gt_img = np.clip(sample_gt[row].permute(1, 2, 0).numpy(), 0, 1)
            axes[row, 3].imshow(gt_img)

            for col in range(4):
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(titles[col], fontsize=13, fontweight="bold", pad=8)

        plt.tight_layout()
        plot_path = "spatial_decoder_comparison_figure.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved publication comparison figure: {plot_path}")


if __name__ == "__main__":
    evaluate()