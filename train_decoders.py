"""
train_decoders.py (Final)
==========================
Trains the Physical Feature Decoder and Optical Image Decoder.

Changes from the previous "Fixed" version:

4. GLOBAL NORMALIZATION FIX: images were previously normalized per-patch
   ((img - img.min()) / (img.max() - img.min()) computed independently for
   EACH patch). That means patch A's "1.0" and patch B's "1.0" represent
   different real reflectance values -- there's no consistent target scale
   for the decoder to learn a single global z -> image mapping onto. Now
   normalization stats (2nd/98th percentile clip, per-band) are computed
   ONCE across the TRAIN split only, saved, and applied identically to
   both train and test images.

5. SSIM LOSS ADDED: pure MSE+L1 pixel losses push the decoder toward
   blurry, averaged outputs. Added an SSIM term (via pytorch-msssim) to
   push toward better structural/perceptual quality.
   Install once: pip install pytorch-msssim

Everything else (strict train-only fitting, explicit fused_emb_* column
selection, explicit true RGB band selection) is unchanged from the
previous fixed version.
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio

from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder

try:
    from pytorch_msssim import ssim as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[warn] pytorch-msssim not installed (pip install pytorch-msssim) "
          "-- training without the SSIM loss term.")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

FULL_BAND_ORDER = [
    "VV", "VH", "VH_VV_ratio", "RVI", "entropy_proxy",
    "B2", "B3", "B4", "B8", "NDVI", "NDWI", "EVI",
]
RGB_BAND_NAMES = ["B4", "B3", "B2"]
RGB_BAND_INDICES_1BASED = [FULL_BAND_ORDER.index(b) + 1 for b in RGB_BAND_NAMES]

TARGET_FEATURE_COLUMNS = ["NDVI_mean", "NDWI_mean", "EVI_mean", "RVI_mean", "VH_VV_ratio_mean"]

NORM_STATS_PATH = "image_norm_stats.json"


class ThesisRealDataset(Dataset):
    """Loads (embedding, physical target features, RGB optical image) triples
    for a given split, matched by patch_id via export_name.

    norm_stats: dict with 'low' and 'high' (each a 3-element list, one per
    RGB channel) used for global percentile-clip normalization. Must be
    computed on the TRAIN split and reused unchanged for TEST -- passing
    None causes this dataset to COMPUTE stats from its own patches (only
    correct to do this for the train split)."""

    def __init__(self, embeddings_csv, metadata_csv, patches_dir, split, img_size=128, norm_stats=None):
        self.patches_dir = patches_dir
        self.img_size = img_size

        emb_df = pd.read_csv(embeddings_csv)
        meta_df = pd.read_csv(metadata_csv)

        if "split" not in emb_df.columns:
            raise ValueError(
                "No `split` column in the embeddings CSV. Run utils/data_split.py "
                "and regenerate Stage 5 output before training decoders."
            )

        emb_df = emb_df[emb_df["split"] == split].reset_index(drop=True)
        if len(emb_df) == 0:
            raise ValueError(f"No patches found for split='{split}'.")

        emb_cols = [c for c in emb_df.columns if c.startswith("fused_emb_")]
        if not emb_cols:
            raise ValueError("No fused_emb_* columns found in embeddings CSV.")
        self.z_data = torch.tensor(emb_df[emb_cols].values, dtype=torch.float32)

        if "export_name" not in meta_df.columns:
            raise ValueError("metadata CSV has no `export_name` column -- cannot match patches reliably.")
        meta_indexed = meta_df.set_index("export_name")

        feat_cols = [c for c in TARGET_FEATURE_COLUMNS if c in meta_indexed.columns]
        if feat_cols:
            matched_feats = meta_indexed.loc[emb_df["patch_id"], feat_cols].values
            self.features = torch.tensor(matched_feats, dtype=torch.float32)
            print(f"[{split}] Loaded physical target features: {feat_cols}")
        else:
            print(f"[{split}] Warning: target feature columns not found -- using zero tensor.")
            self.features = torch.zeros((len(self.z_data), len(TARGET_FEATURE_COLUMNS)), dtype=torch.float32)

        self.patch_ids = emb_df["patch_id"].tolist()
        path_col = "local_path" if "local_path" in meta_indexed.columns else (
            "filepath" if "filepath" in meta_indexed.columns else None
        )
        if path_col:
            self.patch_paths = meta_indexed.loc[emb_df["patch_id"], path_col].tolist()
        else:
            self.patch_paths = [None] * len(self.patch_ids)

        self.norm_stats = norm_stats
        if self.norm_stats is None:
            print(f"[{split}] No norm_stats provided -- computing GLOBAL percentile "
                  f"stats from this split (only correct for the TRAIN split).")
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
        """2nd/98th percentile per RGB channel, computed ONCE across all
        patches in this split. Fixes the old per-patch min-max normalization
        bug where every patch had its own arbitrary [0,1] scale."""
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
                low.append(0.0)
                high.append(1.0)
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

        # GLOBAL normalization using train-split-derived stats (fixed bug).
        low = np.array(self.norm_stats["low"]).reshape(3, 1, 1)
        high = np.array(self.norm_stats["high"]).reshape(3, 1, 1)
        img_data = np.clip((img_data - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)

        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        if img_tensor.shape[1:] != (self.img_size, self.img_size):
            img_tensor = nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        return z, feat, img_tensor


def train_pipeline():
    train_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="train",
        norm_stats=None,  # computed fresh from train split, saved to disk
    )
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"\nTraining decoders on {len(train_dataset)} TRAIN-split patches only "
          f"(test-split patches are held out for evaluate_generation.py).")

    # --- Train Feature Decoder ---
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

    # --- Train Optical Image Decoder ---
    print("\n--- Training Optical Image Decoder ---")
    img_model = OpticalImageDecoder(latent_dim=train_dataset.z_data.shape[1], out_channels=3)
    optimizer_img = optim.Adam(img_model.parameters(), lr=5e-4)
    criterion_l1 = nn.L1Loss()

    img_model.train()
    for epoch in range(100):
        total_loss = 0.0
        for z_b, _, img_b in loader:
            optimizer_img.zero_grad()
            gen_imgs = img_model(z_b)
            loss = criterion_mse(gen_imgs, img_b) + 0.5 * criterion_l1(gen_imgs, img_b)
            if HAVE_SSIM:
                loss = loss + 0.5 * (1 - ssim_fn(gen_imgs, img_b, data_range=1.0))
            loss.backward()
            optimizer_img.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Image Decoder   - Epoch [{epoch+1}/100] | Combined Loss: {total_loss/len(loader):.6f}")

    torch.save(feat_model.state_dict(), "feature_decoder.pth")
    torch.save(img_model.state_dict(), "image_decoder.pth")
    print("\nSuccessfully trained models and saved feature_decoder.pth & image_decoder.pth!")


if __name__ == "__main__":
    train_pipeline()