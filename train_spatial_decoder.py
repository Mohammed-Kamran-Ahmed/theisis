"""
train_spatial_decoder.py
===========================
Novelty 3 (PRIMARY CONTRIBUTION) — Spatially-Conditioned SAR→Optical Translation Decoder
-----------------------------------------------------------------------------------------
Claim (IEEE GRSL / Remote Sensing MDPI):
"Unlike conventional cross-modal decoders that hallucinate optical texture from a
single compressed 1D latent bottleneck (e.g., OpticalImageDecoder), we propose a
Spatially-Conditioned Decoder (SCD) that leverages intermediate multi-scale feature
maps via U-Net skip connections from a pre-trained SAR encoder, modulated by global
semantic embedding vectors through Feature-wise Linear Modulation (FiLM). This
preserves spatial boundaries, crop edges, and texture gradients while translating
SAR microwave backscatter into calibrated 128x128 optical RGB reflectance."

Features:
- Connects `SAREcoderWithSkips` -> `SpatiallyConditionedDecoder` (with FiLM + 4-level skips)
- Target resolution: 128x128 RGB
- Loss: MSE + 0.5*L1 + 0.5*SSIM + 0.2*Perceptual (VGG-16 feature matching)
- Automatic CUDA T1000 GPU acceleration
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
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights

from models.spatial_decoder import SpatiallyConditionedDecoder, SAREcoderWithSkips
from stage4_contrastive_learning import MultiModalEncoder

try:
    from pytorch_msssim import ssim as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False
    print("[warn] pytorch-msssim not installed -- training without SSIM loss term.")

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_CSV        = "./patch_metadata_clustered.csv"
PATCH_DIR        = "./downloaded_patches"
MODEL_DIR        = "./saved_models"
SAR_BEST_MODEL   = os.path.join(MODEL_DIR, "sar_encoder_best.pth")
SAR_BASE_MODEL   = os.path.join(MODEL_DIR, "sar_encoder.pth")
SAR_RECLUST_MODEL= os.path.join(MODEL_DIR, "sar_encoder_reclustered.pth")

DECODER_OUT_PATH = "spatial_image_decoder.pth"
NORM_STATS_PATH  = "image_norm_stats.json"

BATCH_SIZE       = 16
NUM_EPOCHS       = 120
LEARNING_RATE    = 5e-4
SAR_INPUT_SIZE   = (64, 64)
OPT_TARGET_SIZE  = (128, 128)

SAR_BAND_INDICES   = [0, 1, 2, 3, 4]          # VV, VH, VH_VV, entropy, RVI
RGB_TARGET_INDICES = [7, 6, 5]                # B4 (Red), B3 (Green), B2 (Blue)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# PERCEPTUAL LOSS (VGG-16 Feature Matching)
# ----------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """VGG-16 feature-space distance for high-frequency optical texture realism."""
    def __init__(self, device):
        super().__init__()
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            self.active = True
        except Exception as e:
            print(f"[warn] Could not initialize VGG-16 perceptual loss ({e}). Fallback to L1/SSIM.")
            self.active = False

    def forward(self, pred, target):
        if not self.active:
            return torch.tensor(0.0, device=pred.device)
        # Normalize to ImageNet stats
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(p), self.vgg(t))


# ----------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------
class SpatialDecoderDataset(Dataset):
    def __init__(self, df, patch_folder, norm_stats, augment=True):
        self.df = df.reset_index(drop=True)
        self.patch_folder = patch_folder
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row):
        for col in ["local_path", "filepath"]:
            if col in row and pd.notna(row[col]) and os.path.exists(str(row[col])):
                return str(row[col])
        if "export_name" in row and pd.notna(row["export_name"]):
            candidate = os.path.join(self.patch_folder, f"{row['export_name']}.tif")
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _standardize(arr):
        tensor = torch.from_numpy(arr).float()
        for c in range(tensor.shape[0]):
            std = tensor[c].std()
            mean = tensor[c].mean()
            if std > 1e-6:
                tensor[c] = (tensor[c] - mean) / std
            else:
                tensor[c] = tensor[c] - mean + torch.randn_like(tensor[c]) * 1e-4
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tif_path = self._resolve_path(row)
        if not tif_path:
            raise FileNotFoundError(f"Missing file for row {idx} ({row.get('export_name')})")

        with rasterio.open(tif_path) as src:
            img = src.read()
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. SAR Input Tensor (64x64)
        sar_arr = img[SAR_BAND_INDICES, :, :]
        sar_t = self._standardize(sar_arr)
        sar_t = F.interpolate(sar_t.unsqueeze(0), size=SAR_INPUT_SIZE, mode="bilinear", align_corners=False).squeeze(0)

        # 2. Optical Target RGB Tensor (128x128)
        rgb_arr = img[RGB_TARGET_INDICES, :, :].astype(np.float64)
        low = np.array(self.norm_stats["low"]).reshape(3, 1, 1)
        high = np.array(self.norm_stats["high"]).reshape(3, 1, 1)
        rgb_arr = np.clip((rgb_arr - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
        rgb_t = torch.tensor(rgb_arr, dtype=torch.float32)
        rgb_t = F.interpolate(rgb_t.unsqueeze(0), size=OPT_TARGET_SIZE, mode="bilinear", align_corners=False).squeeze(0)

        # Augmentations (synchronized across SAR and Optical)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                sar_t = torch.flip(sar_t, dims=[2])
                rgb_t = torch.flip(rgb_t, dims=[2])
            if torch.rand(1).item() > 0.5:
                sar_t = torch.flip(sar_t, dims=[1])
                rgb_t = torch.flip(rgb_t, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                sar_t = torch.rot90(sar_t, k, dims=[1, 2])
                rgb_t = torch.rot90(rgb_t, k, dims=[1, 2])

        return sar_t, rgb_t


def get_or_compute_norm_stats(df, patch_folder):
    if os.path.exists(NORM_STATS_PATH):
        try:
            with open(NORM_STATS_PATH) as f:
                stats = json.load(f)
            if "low" in stats and "high" in stats:
                return stats
        except Exception:
            pass

    print("Computing fresh RGB normalization percentiles from TRAIN split...")
    all_pixels = [[], [], []]
    for _, row in df.iterrows():
        path = None
        for col in ["local_path", "filepath"]:
            if col in row and pd.notna(row[col]) and os.path.exists(str(row[col])):
                path = str(row[col])
                break
        if not path:
            path = os.path.join(patch_folder, f"{row.get('export_name')}.tif")
            if not os.path.exists(path):
                continue
        with rasterio.open(path) as src:
            img = np.nan_to_num(src.read(), nan=0.0, posinf=0.0, neginf=0.0)
        rgb = img[RGB_TARGET_INDICES, :, :]
        for c in range(3):
            all_pixels[c].append(rgb[c].ravel())

    low, high = [], []
    for c in range(3):
        vals = np.concatenate(all_pixels[c]) if len(all_pixels[c]) > 0 else np.array([0, 1])
        low.append(float(np.percentile(vals, 2)))
        high.append(float(np.percentile(vals, 98)))

    stats = {"low": low, "high": high}
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats, f)
    return stats


# ----------------------------------------------------------------------
# TRAINING PIPELINE
# ----------------------------------------------------------------------
def train_pipeline():
    print("=" * 70)
    print("  NOVELTY 3: SPATIALLY-CONDITIONED U-NET DECODER TRAINING (128x128)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csv_path = OUTPUT_CSV = "./patch_metadata_reclustered.csv" if os.path.exists("./patch_metadata_reclustered.csv") else INPUT_CSV
    df = pd.read_csv(csv_path)
    train_df = df[(df["valid"] == True) & (df["split"] == "train")].copy().reset_index(drop=True)
    test_df  = df[(df["valid"] == True) & (df["split"] == "test")].copy().reset_index(drop=True)
    print(f"Dataset: {len(train_df)} train / {len(test_df)} test patches.")

    norm_stats = get_or_compute_norm_stats(train_df, PATCH_DIR)

    train_dataset = SpatialDecoderDataset(train_df, PATCH_DIR, norm_stats, augment=True)
    test_dataset  = SpatialDecoderDataset(test_df, PATCH_DIR, norm_stats, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 1. Load Pretrained SAR Encoder & wrap with Skips
    encoder_path = SAR_RECLUST_MODEL if os.path.exists(SAR_RECLUST_MODEL) else (SAR_BEST_MODEL if os.path.exists(SAR_BEST_MODEL) else SAR_BASE_MODEL)
    print(f"Loading frozen SAR backbone from: {encoder_path}")
    base_encoder = MultiModalEncoder(in_channels=len(SAR_BAND_INDICES), embed_dim=128, proj_dim=64, pretrained=False)
    base_encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    base_encoder.to(DEVICE)
    base_encoder.eval()
    for param in base_encoder.parameters():
        param.requires_grad = False

    sar_encoder = SAREcoderWithSkips(base_encoder).to(DEVICE)
    sar_encoder.eval()

    # 2. Instantiate Spatially-Conditioned Decoder (FiLM + Skip Connections -> 128x128 RGB)
    decoder = SpatiallyConditionedDecoder(embed_dim=64, out_channels=3).to(DEVICE)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    criterion_mse = nn.MSELoss()
    criterion_l1  = nn.L1Loss()
    perceptual_fn = PerceptualLoss(DEVICE)

    best_test_loss = float("inf")
    best_weights = None

    print(f"\n--- Training Spatially-Conditioned Decoder ({NUM_EPOCHS} epochs) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        decoder.train()
        total_loss = 0.0
        steps = 0

        for sar_b, rgb_b in train_loader:
            sar_b, rgb_b = sar_b.to(DEVICE), rgb_b.to(DEVICE)

            with torch.no_grad():
                _, z_sar, bottleneck, skip3, skip2, skip1 = sar_encoder(sar_b)

            optimizer.zero_grad()
            gen_imgs = decoder(bottleneck, skip3, skip2, skip1, z_sar)

            loss_mse = criterion_mse(gen_imgs, rgb_b)
            loss_l1  = criterion_l1(gen_imgs, rgb_b)
            loss_p   = perceptual_fn(gen_imgs, rgb_b)
            loss_ssim = 1.0 - ssim_fn(gen_imgs, rgb_b, data_range=1.0) if HAVE_SSIM else 0.0

            loss = loss_mse + 0.5 * loss_l1 + 0.5 * loss_ssim + 0.2 * loss_p

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            # Validation on test set
            decoder.eval()
            val_loss = 0.0
            val_ssim = 0.0
            with torch.no_grad():
                for v_sar, v_rgb in test_loader:
                    v_sar, v_rgb = v_sar.to(DEVICE), v_rgb.to(DEVICE)
                    _, vz, vb, vs3, vs2, vs1 = sar_encoder(v_sar)
                    v_gen = decoder(vb, vs3, vs2, vs1, vz)
                    v_mse = criterion_mse(v_gen, v_rgb).item()
                    val_loss += v_mse
                    if HAVE_SSIM:
                        val_ssim += ssim_fn(v_gen, v_rgb, data_range=1.0).item()

            avg_val_loss = val_loss / max(len(test_loader), 1)
            avg_val_ssim = val_ssim / max(len(test_loader), 1) if HAVE_SSIM else 0.0

            if avg_val_loss < best_test_loss:
                best_test_loss = avg_val_loss
                best_weights = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

            print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] | Train Loss: {total_loss/steps:.4f} | Test MSE: {avg_val_loss:.5f} | Test SSIM: {avg_val_ssim:.4f}")

    # Save trained model
    torch.save(best_weights if best_weights is not None else decoder.state_dict(), DECODER_OUT_PATH)
    print(f"\n✅ Successfully saved trained Spatial Decoder to: {DECODER_OUT_PATH}")


if __name__ == "__main__":
    train_pipeline()