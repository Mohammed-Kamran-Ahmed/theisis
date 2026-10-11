"""
evaluate_generation.py (Final)
================================
Evaluates the trained Physical Feature Decoder and Optical Image Decoder
on the HELD-OUT TEST SPLIT only.

Update from the previous "Fixed" version: loads the normalization stats
that were computed on the TRAIN split and saved to image_norm_stats.json
by train_decoders.py, and reuses them unchanged here -- so test images are
normalized with the same scale the model was trained on, instead of each
split (or worse, each patch) inventing its own scale.
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import torch
from torch.utils.data import DataLoader

from train_decoders import ThesisRealDataset, NORM_STATS_PATH
from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder
from utils.metrics import compute_feature_metrics, compute_image_metrics


def evaluate_pipeline():
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    print(f"Using train-split normalization stats: {norm_stats}")

    test_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="test",
        norm_stats=norm_stats,  # reuse TRAIN stats -- never recompute on test
    )

    print(f"Evaluating on {len(test_dataset)} TEST-split patches "
          f"(never seen during decoder training).")

    loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    z_real, target_features_real, target_images_real = next(iter(loader))

    feat_model = PhysicalFeatureDecoder(latent_dim=z_real.shape[1], num_features=target_features_real.shape[1])
    img_model = OpticalImageDecoder(latent_dim=z_real.shape[1], out_channels=target_images_real.shape[1])

    feat_model.load_state_dict(torch.load("feature_decoder.pth"))
    img_model.load_state_dict(torch.load("image_decoder.pth"))

    feat_model.eval()
    img_model.eval()

    with torch.no_grad():
        pred_features = feat_model(z_real)
        pred_images = img_model(z_real)

    feat_res = compute_feature_metrics(pred_features, target_features_real)
    img_res = compute_image_metrics(pred_images, target_images_real)

    print("\n" + "=" * 60)
    print("   STAGE 8 & 9 GENERATION BENCHMARKS -- HELD-OUT TEST SET")
    print("=" * 60)
    print(f"Physical Feature Reconstruction MAE  : {feat_res['feat_mae']:.4f}")
    print(f"Physical Feature Reconstruction RMSE : {feat_res['feat_rmse']:.4f}")
    print("-" * 60)
    print(f"Mean Structural Similarity (SSIM)    : {img_res['mean_ssim']:.4f}")
    print(f"Mean Peak Signal-to-Noise Ratio      : {img_res['mean_psnr']:.2f} dB")
    print("=" * 60)
    print(f"\n(n={len(test_dataset)} test patches -- small sample, treat as indicative)\n")


if __name__ == "__main__":
    evaluate_pipeline()