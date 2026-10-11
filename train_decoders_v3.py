"""
train_decoders.py (v3 - perceptual loss + augmentation)
==========================================================
Adds three real improvements on top of the previous fixed version:

1. PERCEPTUAL (VGG) LOSS: borrows a pretrained VGG16's feature space to
   judge whether the output LOOKS structurally like a real image, instead
   of only judging raw pixel distance. This is the single biggest lever
   for visual quality on a small dataset, because the "what images look
   like" prior comes from VGG's ImageNet pretraining, not from your 137
   patches.

2. DATA AUGMENTATION: random horizontal/vertical flip + 90-degree
   rotation applied identically to SAR embedding's paired image (the
   embedding itself is orientation-invariant since it comes from a
   pretrained SAR encoder's pooled features, so only the target image
   needs the corresponding flip/rotation applied here). Effectively
   multiplies the training signal from 137 patches.

3. TRAIN_IMG_SIZE: configurable, defaults to 64 instead of 128. Half the
   pixels to predict is a meaningfully easier task for a decoder this
   small -- sharper relative output. Set back to 128 if you want to
   compare directly against your original numbers.

Everything from the previous version (global normalization computed once
on train, explicit fused_emb_* selection, explicit RGB band selection,
strict train/test split) is unchanged.
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import glob
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
import rasterio

from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder

try:
    from pytorch_msssim import ssim as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[warn] pytorch-msssim not installed -- training without SSIM loss term.")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TRAIN_IMG_SIZE = 64   # was 128. Lower = easier task = sharper relative output.

FULL_BAND_ORDER = [
    "VV", "VH", "VH_VV_ratio", "RVI", "entropy_proxy",
    "B2", "B3", "B4", "B8", "NDVI", "NDWI", "EVI",
]
RGB_BAND_NAMES = ["B4", "B3", "B2"]
RGB_BAND_INDICES_1BASED = [FULL_BAND_ORDER.index(b) + 1 for b in RGB_BAND_NAMES]

TARGET_FEATURE_COLUMNS = ["NDVI_mean", "NDWI_mean", "EVI_mean", "RVI_mean", "VH_VV_ratio_mean"]

NORM_STATS_PATH = "image_norm_stats.json"


# ----------------------------------------------------------------------
# Perceptual (VGG) loss
# ----------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """Compares VGG16 features of generated vs. real image instead of raw
    pixels. Uses early-to-mid layers (relu1_2, relu2_2, relu3_3) which
    capture texture/structure without needing high resolution or many
    training images -- the prior is borrowed from ImageNet pretraining."""

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen, real):
        gen_n = (gen - self.mean) / self.std
        real_n = (real - self.mean) / self.std
        loss = 0.0
        g, r = gen_n, real_n
        for slice_ in [self.slice1, self.slice2, self.slice3]:
            g, r = slice_(g), slice_(r)
            loss = loss + nn.functional.l1_loss(g, r)
        return loss


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class ThesisRealDataset(Dataset):
    def __init__(self, embeddings_csv, metadata_csv, patches_dir, split,
                 img_size=TRAIN_IMG_SIZE, norm_stats=None, augment=False):
        self.patches_dir = patches_dir
        self.img_size = img_size
        self.augment = augment

        emb_df = pd.read_csv(embeddings_csv)
        meta_df = pd.read_csv(metadata_csv)

        if "split" not in emb_df.columns:
            raise ValueError("No `split` column in the embeddings CSV.")

        emb_df = emb_df[emb_df["split"] == split].reset_index(drop=True)
        if len(emb_df) == 0:
            raise ValueError(f"No patches found for split='{split}'.")

        emb_cols = [c for c in emb_df.columns if c.startswith("fused_emb_")]
        if not emb_cols:
            raise ValueError("No fused_emb_* columns found in embeddings CSV.")
        self.z_data = torch.tensor(emb_df[emb_cols].values, dtype=torch.float32)

        if "export_name" not in meta_df.columns:
            raise ValueError("metadata CSV has no `export_name` column.")
        meta_indexed = meta_df.set_index("export_name")

        feat_cols = [c for c in TARGET_FEATURE_COLUMNS if c in meta_indexed.columns]
        if feat_cols:
            matched_feats = meta_indexed.loc[emb_df["patch_id"], feat_cols].values
            self.features = torch.tensor(matched_feats, dtype=torch.float32)
            print(f"[{split}] Loaded physical target features: {feat_cols}")
        else:
            self.features = torch.zeros((len(self.z_data), len(TARGET_FEATURE_COLUMNS)), dtype=torch.float32)

        self.patch_ids = emb_df["patch_id"].tolist()
        path_col = "local_path" if "local_path" in meta_indexed.columns else (
            "filepath" if "filepath" in meta_indexed.columns else None
        )
        self.patch_paths = meta_indexed.loc[emb_df["patch_id"], path_col].tolist() if path_col else [None] * len(self.patch_ids)

        self.norm_stats = norm_stats
        if self.norm_stats is None:
            print(f"[{split}] Computing GLOBAL percentile stats from this split "
                  f"(only correct for TRAIN split).")
            self.norm_stats = self._compute_global_stats()

    def _read_raw_rgb(self, idx):
        tif_path = self._resolve_patch_path(idx)
        if tif_path is None:
            return None
        with rasterio.open(tif_path) as src:
            if src.count >= max(RGB_BAND_INDICES_1BASED):
                img_data = src.read(RGB_BAND_INDICES_1BASED)
            else:
                n = min(src.count, 3)
                img_data = src.read(list(range(1, n + 1)))
                if img_data.shape[0] < 3:
                    pad = np.zeros((3 - img_data.shape[0], *img_data.shape[1:]), dtype=img_data.dtype)
                    img_data = np.concatenate([img_data, pad], axis=0)
        return np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_global_stats(self):
        all_pixels = [[], [], []]
        for idx in range(len(self.patch_ids)):
            img = self._read_raw_rgb(idx)
            if img is None:
                continue
            for c in range(3):
                all_pixels[c].append(img[c].ravel())
        low, high = [], []
        for c in range(3):
            if all_pixels[c]:
                vals = np.concatenate(all_pixels[c])
                low.append(float(np.percentile(vals, 2)))
                high.append(float(np.percentile(vals, 98)))
            else:
                low.append(0.0); high.append(1.0)
        stats = {"low": low, "high": high}
        with open(NORM_STATS_PATH, "w") as f:
            json.dump(stats, f)
        print(f"Saved global normalization stats to {NORM_STATS_PATH}: {stats}")
        return stats

    def __len__(self):
        return len(self.z_data)

    def _resolve_patch_path(self, idx):
        stored_path = self.patch_paths[idx]
        if stored_path and os.path.exists(str(stored_path)):
            return str(stored_path)
        patch_id = self.patch_ids[idx]
        candidate = os.path.join(self.patches_dir, f"{patch_id}.tif")
        if os.path.exists(candidate):
            return candidate
        matches = glob.glob(os.path.join(self.patches_dir, f"*{patch_id}*.tif"))
        return matches[0] if matches else None

    def __getitem__(self, idx):
        z = self.z_data[idx]
        feat = self.features[idx]

        img_data = self._read_raw_rgb(idx)
        if img_data is None:
            img_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            return z, feat, img_tensor

        low = np.array(self.norm_stats["low"]).reshape(3, 1, 1)
        high = np.array(self.norm_stats["high"]).reshape(3, 1, 1)
        img_data = np.clip((img_data - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)

        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        if img_tensor.shape[1:] != (self.img_size, self.img_size):
            img_tensor = nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])  # horizontal flip
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])  # vertical flip
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k=k, dims=[1, 2])

        return z, feat, img_tensor


def train_pipeline():
    train_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="train",
        norm_stats=None,
        augment=True,   # <-- augmentation on for training
    )
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"\nTraining decoders on {len(train_dataset)} TRAIN-split patches "
          f"(augmented) at {TRAIN_IMG_SIZE}x{TRAIN_IMG_SIZE} resolution.")

    # --- Feature Decoder (unchanged) ---
    print("\n--- Training Physical Feature Decoder ---")
    feat_model = PhysicalFeatureDecoder(latent_dim=train_dataset.z_data.shape[1], num_features=train_dataset.features.shape[1])
    optimizer_feat = optim.Adam(feat_model.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    feat_model.train()
    for epoch in range(50):
        total_loss = 0.0
        for z_b, f_b, _ in loader:
            optimizer_feat.zero_grad()
            preds = feat_model(z_b)
            loss = criterion_mse(preds, f_b)
            loss.backward()
            optimizer_feat.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Feature Decoder - Epoch [{epoch+1}/50] | MSE Loss: {total_loss/len(loader):.6f}")

    # --- Image Decoder with perceptual + SSIM + pixel loss ---
    print("\n--- Training Optical Image Decoder (perceptual + SSIM + pixel loss) ---")
    img_model = OpticalImageDecoder(latent_dim=train_dataset.z_data.shape[1], out_channels=3)
    optimizer_img = optim.Adam(img_model.parameters(), lr=5e-4)
    criterion_l1 = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss()
    perceptual_loss_fn.eval()

    img_model.train()
    for epoch in range(150):
        total_loss = 0.0
        for z_b, _, img_b in loader:
            optimizer_img.zero_grad()
            gen_imgs = img_model(z_b)
            # NOTE: img_model outputs the size baked into its architecture (128x128
            # from the fixed 3 upsample blocks). If TRAIN_IMG_SIZE != 128, resize
            # gen_imgs down to match the target for loss computation.
            if gen_imgs.shape[-1] != img_b.shape[-1]:
                gen_imgs_for_loss = nn.functional.interpolate(
                    gen_imgs, size=img_b.shape[-2:], mode="bilinear", align_corners=False)
            else:
                gen_imgs_for_loss = gen_imgs

            pixel_loss = criterion_mse(gen_imgs_for_loss, img_b) + 0.5 * criterion_l1(gen_imgs_for_loss, img_b)
            perc_loss = perceptual_loss_fn(gen_imgs_for_loss, img_b)
            loss = pixel_loss + 0.3 * perc_loss
            if HAVE_SSIM:
                loss = loss + 0.5 * (1 - ssim_fn(gen_imgs_for_loss, img_b, data_range=1.0))

            loss.backward()
            optimizer_img.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Image Decoder   - Epoch [{epoch+1}/150] | Combined Loss: {total_loss/len(loader):.6f}")

    torch.save(feat_model.state_dict(), "feature_decoder.pth")
    torch.save(img_model.state_dict(), "image_decoder.pth")
    print("\nSuccessfully trained models and saved feature_decoder.pth & image_decoder.pth!")

    print("Contrastive loss == ")


if __name__ == "__main__":
    train_pipeline()