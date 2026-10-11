"""
augmentations.py
==================
Fix #4 for the overfitting gap: stronger, synchronized SAR+optical
augmentation, applied identically to both modalities of a given patch
so spatial correspondence is preserved.

Your existing Dataset classes already do flips + 90-degree rotations
(see SpatialDecoderDataset.__getitem__ in train_spatial_decoder.py).
This module adds three more transforms that stack on top of those:

  1. Synchronized random-resized-crop (forces the encoder to learn from
     partial views of a patch instead of memorizing whole-patch identity
     -- this is the single most effective anti-memorization augmentation
     in contrastive learning generally, per SimCLR/SupCon ablations).
  2. Speckle noise injection for SAR (physically realistic -- real SAR
     imagery has multiplicative speckle noise from coherent backscatter;
     simulating more of it is a legitimate domain-appropriate
     augmentation, not just generic noise).
  3. Random erasing / cutout (applied to SAR channels only, forces the
     encoder not to rely on any single small region of the patch).

Usage: import `SyncedAugmentation` and call it once per __getitem__,
after your existing standardization step, before the flip/rotate block
you already have.
"""

import torch
import torch.nn.functional as F


class SyncedAugmentation:
    """
    Synchronized augmentation applied identically (same crop box, same
    flip/rotate decision) to a SAR tensor and its paired optical tensor,
    so pixel-level spatial correspondence between the two modalities is
    preserved. Speckle noise and erasing are applied to SAR only, since
    those are meant to simulate SAR-specific sensor characteristics /
    force robustness to partial occlusion -- applying them to the optical
    *target* would corrupt the ground truth you're trying to reconstruct.

    Args:
        crop_scale: (min, max) fraction of original area to keep in the
            random-resized-crop step. (0.7, 1.0) is a mild setting
            appropriate for small 64x64/128x128 patches -- more aggressive
            cropping (e.g. (0.5, 1.0)) risks cropping out the informative
            part of a small agricultural patch entirely.
        speckle_std: standard deviation of the multiplicative speckle
            noise applied to SAR channels. SAR speckle is usually modeled
            as I_noisy = I_clean * (1 + noise), noise ~ N(0, speckle_std).
        erase_prob: probability of applying random erasing per sample.
        erase_area: (min, max) fraction of area to erase.
    """

    def __init__(
        self,
        crop_scale=(0.7, 1.0),
        speckle_std: float = 0.08,
        erase_prob: float = 0.3,
        erase_area=(0.02, 0.15),
    ):
        self.crop_scale = crop_scale
        self.speckle_std = speckle_std
        self.erase_prob = erase_prob
        self.erase_area = erase_area

    def _random_resized_crop_box(self, H: int, W: int):
        area = H * W
        for _ in range(10):  # a few attempts to find a valid box
            target_area = area * (
                self.crop_scale[0]
                + torch.rand(1).item() * (self.crop_scale[1] - self.crop_scale[0])
            )
            aspect = 0.9 + torch.rand(1).item() * 0.2  # mild aspect jitter
            w = int(round((target_area * aspect) ** 0.5))
            h = int(round((target_area / aspect) ** 0.5))
            if w <= W and h <= H and w > 0 and h > 0:
                x0 = torch.randint(0, W - w + 1, (1,)).item()
                y0 = torch.randint(0, H - h + 1, (1,)).item()
                return x0, y0, w, h
        # fallback: full image
        return 0, 0, W, H

    def _apply_crop_resize(self, tensor: torch.Tensor, box, out_size):
        x0, y0, w, h = box
        cropped = tensor[:, y0:y0 + h, x0:x0 + w].unsqueeze(0)
        resized = F.interpolate(cropped, size=out_size, mode="bilinear", align_corners=False)
        return resized.squeeze(0)

    def _speckle_noise(self, sar_tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(sar_tensor) * self.speckle_std
        return sar_tensor * (1.0 + noise)

    def _random_erase(self, sar_tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.erase_prob:
            return sar_tensor
        C, H, W = sar_tensor.shape
        area = H * W
        erase_area = area * (
            self.erase_area[0] + torch.rand(1).item() * (self.erase_area[1] - self.erase_area[0])
        )
        side = int(round(erase_area ** 0.5))
        side = max(1, min(side, min(H, W)))
        y0 = torch.randint(0, H - side + 1, (1,)).item()
        x0 = torch.randint(0, W - side + 1, (1,)).item()
        # fill with per-channel mean (neutral value) rather than 0, since 0
        # may itself be a meaningful SAR value after standardization
        fill = sar_tensor.mean(dim=(1, 2), keepdim=True)
        sar_tensor = sar_tensor.clone()
        sar_tensor[:, y0:y0 + side, x0:x0 + side] = fill
        return sar_tensor

    def __call__(self, sar_t: torch.Tensor, rgb_t: torch.Tensor):
        """
        Args:
            sar_t: SAR tensor, shape [C_sar, H_sar, W_sar] (already
                standardized, e.g. output of your `_standardize` method).
            rgb_t: optical RGB tensor, shape [3, H_rgb, W_rgb], normalized
                to [0, 1] (same as your existing pipeline).

        Returns:
            (sar_t_aug, rgb_t_aug) -- same shapes as input.
        """
        _, H_sar, W_sar = sar_t.shape
        _, H_rgb, W_rgb = rgb_t.shape

        # 1. Synchronized random-resized-crop -- same *relative* box
        #    applied to both, since SAR and optical may be at different
        #    resolutions (your pipeline uses 64x64 SAR / 128x128 optical).
        x0f, y0f, wf, hf = self._random_resized_crop_box(H_sar, W_sar)
        # relative box, applied in each modality's own resolution
        rel_box = (x0f / W_sar, y0f / H_sar, wf / W_sar, hf / H_sar)

        sar_box = (
            int(rel_box[0] * W_sar), int(rel_box[1] * H_sar),
            max(1, int(rel_box[2] * W_sar)), max(1, int(rel_box[3] * H_sar)),
        )
        rgb_box = (
            int(rel_box[0] * W_rgb), int(rel_box[1] * H_rgb),
            max(1, int(rel_box[2] * W_rgb)), max(1, int(rel_box[3] * H_rgb)),
        )

        sar_t = self._apply_crop_resize(sar_t, sar_box, (H_sar, W_sar))
        rgb_t = self._apply_crop_resize(rgb_t, rgb_box, (H_rgb, W_rgb))

        # 2. SAR-only augmentations (do NOT apply to optical target)
        sar_t = self._speckle_noise(sar_t)
        sar_t = self._random_erase(sar_t)

        return sar_t, rgb_t


# ----------------------------------------------------------------------
# INTEGRATION NOTES for train_spatial_decoder.py / stage4_contrastive_learning.py
# ----------------------------------------------------------------------
# In SpatialDecoderDataset.__getitem__ (train_spatial_decoder.py), you
# currently have this augmentation block near the end:
#
#     if self.augment:
#         if torch.rand(1).item() > 0.5:
#             sar_t = torch.flip(sar_t, dims=[2])
#             rgb_t = torch.flip(rgb_t, dims=[2])
#         if torch.rand(1).item() > 0.5:
#             sar_t = torch.flip(sar_t, dims=[1])
#             rgb_t = torch.flip(rgb_t, dims=[1])
#         k = torch.randint(0, 4, (1,)).item()
#         if k > 0:
#             sar_t = torch.rot90(sar_t, k, dims=[1, 2])
#             rgb_t = torch.rot90(rgb_t, k, dims=[1, 2])
#
# Add the new augmentation BEFORE this block (crop first, then flip/rotate
# the cropped result -- order doesn't matter much here, but doing crop
# first means the flips/rotations operate on the already-cropped patch):
#
#     from augmentations import SyncedAugmentation
#     # instantiate once in __init__, not per __getitem__:
#     # self.synced_aug = SyncedAugmentation()
#
#     if self.augment:
#         sar_t, rgb_t = self.synced_aug(sar_t, rgb_t)
#         # ... existing flip/rotate block stays exactly as-is below this ...
#
# Same pattern applies to whatever Dataset class stage4_contrastive_learning.py
# uses for SupCon training -- apply it there too, since that's actually the
# stage most exposed to overfitting.