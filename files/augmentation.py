"""
Paired SAR-optical augmentation to reduce the train/test generalization gap
(silhouette 0.855 train vs 0.153-0.289 test, Table 3).

Key design point: SAR and optical patches are co-registered, so GEOMETRIC
transforms (flip/rotate/crop) must be applied IDENTICALLY to both modalities
of a pair, or you destroy the pairing the whole contrastive objective depends
on. PHOTOMETRIC transforms (noise, brightness) should differ per modality,
since SAR speckle noise and optical radiometric noise are physically
different processes.

Drop this into your existing Dataset.__getitem__ (Stage B / CAHN training),
or wrap your existing dataset with `AugmentedPairDataset` below.

This does NOT replace collecting more data — augmentation reduces overfitting
on the *existing* training patches, it doesn't add new geographic diversity
or increase effective sample count for cluster-imbalance purposes (Cluster 3
is still n=5 real patches; augmenting them makes near-duplicates, which is
why the roadmap also asks you to merge/drop clusters — see recluster_k67.py).
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def paired_random_flip(sar: torch.Tensor, opt: torch.Tensor):
    """sar: (C_sar,H,W), opt: (C_opt,H,W) -- same flip applied to both."""
    if random.random() < 0.5:
        sar = torch.flip(sar, dims=[-1])  # horizontal
        opt = torch.flip(opt, dims=[-1])
    if random.random() < 0.5:
        sar = torch.flip(sar, dims=[-2])  # vertical
        opt = torch.flip(opt, dims=[-2])
    return sar, opt


def paired_random_rot90(sar: torch.Tensor, opt: torch.Tensor):
    """Rotate by 0/90/180/270 degrees -- exact pixel-grid rotation, no interpolation
    artifacts, safe for physical SAR statistics (VV/VH ratios etc. are rotation-invariant
    per-pixel so this doesn't distort the regression targets)."""
    k = random.randint(0, 3)
    if k > 0:
        sar = torch.rot90(sar, k, dims=[-2, -1])
        opt = torch.rot90(opt, k, dims=[-2, -1])
    return sar, opt


def sar_speckle_noise(sar: torch.Tensor, mean=1.0, std=0.15, p=0.5):
    """Multiplicative gamma-distributed speckle noise -- physically appropriate for SAR
    (SAR speckle is multiplicative, not additive Gaussian like optical sensor noise)."""
    if random.random() < p:
        # approximate gamma noise via log-normal for simplicity/differentiability
        noise = torch.exp(torch.randn_like(sar) * std) * mean
        sar = sar * noise
    return sar


def optical_radiometric_jitter(opt: torch.Tensor, brightness=0.1, contrast=0.1, p=0.5):
    """Mild brightness/contrast jitter for optical bands (simulates atmospheric/
    illumination variation across acquisitions)."""
    if random.random() < p:
        b = 1.0 + (random.random() * 2 - 1) * brightness
        c = 1.0 + (random.random() * 2 - 1) * contrast
        mean = opt.mean()
        opt = (opt - mean) * c + mean
        opt = opt * b
        opt = torch.clamp(opt, 0.0, 1.0)
    return opt


def cutout(img: torch.Tensor, n_holes=1, length=16, p=0.5):
    """Random square cutout. Apply to SAR and optical INDEPENDENTLY (not paired) --
    this forces the encoder to not rely on any single local patch region, which is
    the regularization effect you actually want against silhouette overfitting."""
    if random.random() >= p:
        return img
    c, h, w = img.shape
    mask = torch.ones(h, w, device=img.device)
    for _ in range(n_holes):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1, y2 = max(0, y - length // 2), min(h, y + length // 2)
        x1, x2 = max(0, x - length // 2), min(w, x + length // 2)
        mask[y1:y2, x1:x2] = 0.0
    return img * mask.unsqueeze(0)


def augment_pair(sar: torch.Tensor, opt: torch.Tensor, training: bool = True):
    """Full augmentation pipeline for one SAR-optical pair. Call this inside
    your Dataset.__getitem__ before returning the tensors, only when training=True."""
    if not training:
        return sar, opt
    sar, opt = paired_random_flip(sar, opt)
    sar, opt = paired_random_rot90(sar, opt)
    sar = sar_speckle_noise(sar)
    opt = optical_radiometric_jitter(opt)
    sar = cutout(sar, length=16, p=0.3)
    opt = cutout(opt, length=16, p=0.3)
    return sar, opt


class AugmentedPairDataset(Dataset):
    """Thin wrapper: pass your existing (unaugmented) dataset, get augmented pairs
    on the fly during training. Leave test/val dataset unwrapped (training=False
    behavior anywhere you evaluate silhouette/CCA/etc., since you want to measure
    generalization on clean, unaugmented patches).

    Example:
        base_train_ds = YourExistingPairedDataset(split="train")
        train_ds = AugmentedPairDataset(base_train_ds)
        # test/val stays as-is:
        test_ds = YourExistingPairedDataset(split="test")
    """

    def __init__(self, base_dataset, training: bool = True):
        self.base = base_dataset
        self.training = training

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # TODO: adapt this unpacking to your existing dataset's return signature.
        # Assumed here: (sar_img, opt_img, *rest) where *rest is e.g. physical
        # targets / cluster label / patch_id, passed through unchanged.
        sar_img, opt_img, *rest = item
        sar_img, opt_img = augment_pair(sar_img, opt_img, training=self.training)
        return (sar_img, opt_img, *rest)


if __name__ == "__main__":
    # smoke test
    sar = torch.randn(5, 128, 128)
    opt = torch.rand(3, 128, 128)
    sar_a, opt_a = augment_pair(sar, opt, training=True)
    print("SAR aug shape:", sar_a.shape, "Optical aug shape:", opt_a.shape)
    print("Augmentation pipeline OK.")
