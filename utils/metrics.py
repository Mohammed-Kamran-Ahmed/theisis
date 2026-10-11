"""
utils/metrics.py
=================
Reconstruction metrics for the Physical Feature Decoder and Optical Image
Decoder. Referenced by evaluate_generation.py — recreated here since it
wasn't part of what was shared, but the corrected evaluate_generation.py
depends on it having exactly this interface.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_feature_metrics(pred_features, target_features):
    """MAE / RMSE between predicted and ground-truth physical descriptor
    vectors. Inputs can be torch tensors or numpy arrays."""
    if isinstance(pred_features, torch.Tensor):
        pred_features = pred_features.detach().cpu().numpy()
    if isinstance(target_features, torch.Tensor):
        target_features = target_features.detach().cpu().numpy()

    mae = np.mean(np.abs(pred_features - target_features))
    rmse = np.sqrt(np.mean((pred_features - target_features) ** 2))
    return {"feat_mae": float(mae), "feat_rmse": float(rmse)}


def compute_image_metrics(pred_images, target_images):
    """Mean SSIM / PSNR across a batch of images. Expects tensors shaped
    [N, C, H, W] with values in [0, 1] (matches the Sigmoid output of
    OpticalImageDecoder and the min-max normalized ground-truth tensors)."""
    if isinstance(pred_images, torch.Tensor):
        pred_images = pred_images.detach().cpu().numpy()
    if isinstance(target_images, torch.Tensor):
        target_images = target_images.detach().cpu().numpy()

    n = pred_images.shape[0]
    ssim_scores = []
    psnr_scores = []

    for i in range(n):
        pred_img = np.transpose(pred_images[i], (1, 2, 0))   # C,H,W -> H,W,C
        target_img = np.transpose(target_images[i], (1, 2, 0))

        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)

        s = ssim(target_img, pred_img, channel_axis=-1, data_range=1.0)
        p = psnr(target_img, pred_img, data_range=1.0)

        ssim_scores.append(s)
        psnr_scores.append(p)

    return {
        "mean_ssim": float(np.mean(ssim_scores)),
        "mean_psnr": float(np.mean(psnr_scores)),
    }