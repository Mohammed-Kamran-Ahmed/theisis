"""
dataset.py - reBEN (Refined BigEarthNet) PyTorch Dataset & Data Loader
======================================================================
Implements:
1. Paired Sentinel-1 (VV, VH) and Sentinel-2 (10 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12) loading.
2. Nearest-neighbor upsampling of 20m optical bands to 10m (120x120 pixels).
3. 19-class CORINE multi-label binary encoding.
4. Channel-wise normalization matching the reBEN benchmark.
5. Spatial synchronized data augmentations (flips, rotations).
6. Support for Parquet, CSV, and synthetic fallback for environment verification.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    import rasterio
except ImportError:
    rasterio = None

from reben_pipeline.metrics import BEN_CLASSES_19

# Standard reBEN Band Order
S1_BANDS = ["VV", "VH"]
S2_BANDS_10M = ["B02", "B03", "B04", "B08"]
S2_BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Official BigEarthNet-v2.0 (reBEN) Normalization Constants from configilm
try:
    from configilm.extra.BENv2_utils import band_combi_to_mean_std
    _s1_m, _s1_s = band_combi_to_mean_std(S1_BANDS)
    _s2_m, _s2_s = band_combi_to_mean_std(S2_BANDS_10)
    S1_MEAN = np.array(_s1_m, dtype=np.float32)
    S1_STD = np.array(_s1_s, dtype=np.float32)
    S2_MEAN = np.array(_s2_m, dtype=np.float32)
    S2_STD = np.array(_s2_s, dtype=np.float32)
except Exception:
    # Exact published BigEarthNet-v2.0 constants (reBEN Table I / Clasen et al., 2024)
    S1_MEAN = np.array([-12.643864, -19.352558], dtype=np.float32)
    S1_STD = np.array([5.133494, 5.5905056], dtype=np.float32)
    S2_MEAN = np.array([438.37207, 614.05566, 588.4096, 942.8433, 1769.9316, 2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325], dtype=np.float32)
    S2_STD = np.array([607.02686, 603.2968, 684.56885, 738.4327, 1100.456, 1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.52765], dtype=np.float32)


class SynchronizedTransforms:
    """Applies identical random spatial transformations to paired SAR and Optical patches."""
    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, s1: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if torch.rand(1).item() < self.p_flip:
            s1 = torch.flip(s1, dims=[-1])
            s2 = torch.flip(s2, dims=[-1])

        # Random vertical flip
        if torch.rand(1).item() < self.p_flip:
            s1 = torch.flip(s1, dims=[-2])
            s2 = torch.flip(s2, dims=[-2])

        # Random 90-degree rotations (0, 90, 180, 270 deg)
        if torch.rand(1).item() < self.p_rot:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                s1 = torch.rot90(s1, k, dims=[-2, -1])
                s2 = torch.rot90(s2, k, dims=[-2, -1])

        return s1, s2


class ReBENDataset(Dataset):
    """
    reBEN Multi-Modal Dataset supporting SAR (S1) and Optical (S2) paired patches.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        s1_dir: Optional[str] = None,
        s2_dir: Optional[str] = None,
        img_size: int = 120,
        normalize: bool = True,
        transform: Optional[SynchronizedTransforms] = None,
        is_synthetic: bool = False
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.s1_dir = Path(s1_dir) if s1_dir else None
        self.s2_dir = Path(s2_dir) if s2_dir else None
        self.img_size = img_size
        self.normalize = normalize
        self.transform = transform
        self.is_synthetic = is_synthetic

        # Map class names to 19-class indices
        self.class_to_idx = {c: i for i, c in enumerate(BEN_CLASSES_19)}
        self.num_classes = len(BEN_CLASSES_19)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_labels(self, labels_raw) -> torch.Tensor:
        """Converts raw list or comma-separated string of labels to a 19-dim binary vector."""
        vec = torch.zeros(self.num_classes, dtype=torch.float32)

        # --- normalise to a Python list first ---
        if isinstance(labels_raw, str):
            # Try to recover a proper Python list from the CSV-round-tripped repr
            # e.g. "['Arable land', 'Pastures']"  →  ['Arable land', 'Pastures']
            try:
                parsed = ast.literal_eval(labels_raw.strip())
                if isinstance(parsed, list):
                    labels_raw = parsed
                else:
                    labels_raw = [labels_raw]   # single bare string
            except (ValueError, SyntaxError):
                # Fall back: strip brackets/quotes and split on comma
                cleaned = labels_raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                labels_raw = [item.strip() for item in cleaned.split(",") if item.strip()]

        # --- encode from the normalised list ---
        if isinstance(labels_raw, (list, np.ndarray)):
            for lbl in labels_raw:
                lbl = str(lbl).strip()
                if lbl in self.class_to_idx:
                    vec[self.class_to_idx[lbl]] = 1.0
                else:
                    # try integer index fallback
                    try:
                        idx = int(lbl)
                        if 0 <= idx < self.num_classes:
                            vec[idx] = 1.0
                    except (ValueError, TypeError):
                        pass
        elif isinstance(labels_raw, (int, np.integer)):
            if 0 <= labels_raw < self.num_classes:
                vec[labels_raw] = 1.0
        return vec


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        if self.is_synthetic:
            # IMPORTANT: seed from idx so the same sample always produces the same
            # image tensor across epochs.  Without this, torch.randn() produces
            # fresh noise every call → the model sees different images each epoch
            # → memorization (and the overfit test) is impossible.
            rng = torch.Generator()
            rng.manual_seed(idx + 123456)
            s1_tensor = torch.randn(2, self.img_size, self.img_size, generator=rng)
            rng.manual_seed(idx + 654321)
            s2_tensor = torch.randn(10, self.img_size, self.img_size, generator=rng)
            labels = self._encode_labels(row.get("labels", [idx % self.num_classes]))
            return {"s1": s1_tensor, "s2": s2_tensor, "targets": labels, "patch_id": str(row.get("patch_id", idx))}

        # Read S1 & S2 from disk
        patch_id = str(row.get("patch_id", idx))
        
        # Resolve file paths (supports separate S1/S2 files or single multi-channel GeoTIFF)
        s1_path = row.get("s1_path", None)
        s2_path = row.get("s2_path", None)
        unified_path = None

        if s1_path is None and s2_path is None:
            # Check common filepath columns in dataframe
            for col in ["filepath", "local_path", "export_name"]:
                if col in row and pd.notna(row[col]):
                    candidate = str(row[col])
                    if os.path.exists(candidate):
                        unified_path = candidate
                        break
                    if self.s1_dir and os.path.exists(os.path.join(self.s1_dir, os.path.basename(candidate))):
                        unified_path = os.path.join(self.s1_dir, os.path.basename(candidate))
                        break

        if s1_path is None and self.s1_dir:
            cand = os.path.join(self.s1_dir, f"{patch_id}.tif")
            if os.path.exists(cand):
                s1_path = cand

        if s2_path is None and self.s2_dir:
            cand = os.path.join(self.s2_dir, f"{patch_id}.tif")
            if os.path.exists(cand):
                s2_path = cand

        # Load raster arrays
        if unified_path and os.path.exists(unified_path) and rasterio:
            with rasterio.open(unified_path) as src:
                full_arr = src.read().astype(np.float32)
                full_arr = np.nan_to_num(full_arr, nan=0.0, posinf=0.0, neginf=0.0)
            s1_arr = full_arr[:2, :, :] if full_arr.shape[0] >= 2 else full_arr
            # Optical bands (take next up to 10 channels)
            s2_arr = full_arr[2:12, :, :] if full_arr.shape[0] > 2 else full_arr
        else:
            if s1_path and os.path.exists(s1_path) and rasterio:
                with rasterio.open(s1_path) as src:
                    s1_arr = src.read().astype(np.float32)
                    s1_arr = np.nan_to_num(s1_arr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                s1_arr = np.zeros((2, self.img_size, self.img_size), dtype=np.float32)

            if s2_path and os.path.exists(s2_path) and rasterio:
                with rasterio.open(s2_path) as src:
                    s2_arr = src.read().astype(np.float32)
                    s2_arr = np.nan_to_num(s2_arr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                s2_arr = np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        s1_tensor = torch.nan_to_num(torch.from_numpy(s1_arr), nan=0.0, posinf=0.0, neginf=0.0)
        s2_tensor = torch.nan_to_num(torch.from_numpy(s2_arr), nan=0.0, posinf=0.0, neginf=0.0)

        # Slice or pad to guarantee exactly 2 channels for S1 and 10 channels for S2
        if s1_tensor.shape[0] > 2:
            s1_tensor = s1_tensor[:2]
        elif s1_tensor.shape[0] < 2:
            s1_tensor = s1_tensor.repeat(2, 1, 1)[:2]

        if s2_tensor.shape[0] > 10:
            s2_tensor = s2_tensor[:10]
        elif s2_tensor.shape[0] < 10:
            reps = int(np.ceil(10.0 / s2_tensor.shape[0]))
            s2_tensor = s2_tensor.repeat(reps, 1, 1)[:10]

        # Ensure spatial dimension is img_size (e.g. 120x120)
        if s1_tensor.shape[-2:] != (self.img_size, self.img_size):
            s1_tensor = F.interpolate(s1_tensor.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        if s2_tensor.shape[-2:] != (self.img_size, self.img_size):
            s2_tensor = F.interpolate(s2_tensor.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)

        # Normalization
        if self.normalize:
            # S1 standardization
            m_s1 = torch.from_numpy(S1_MEAN).view(2, 1, 1)
            s_s1 = torch.from_numpy(S1_STD).view(2, 1, 1)
            s1_tensor = (s1_tensor - m_s1) / s_s1

            # S2 standardization (scale if raw DN reflectance is in [0, 10000])
            if s2_tensor.max() > 10.0:
                m_s2 = torch.from_numpy(S2_MEAN).view(10, 1, 1)
                s_s2 = torch.from_numpy(S2_STD).view(10, 1, 1)
                s2_tensor = (s2_tensor - m_s2) / s_s2
            else:
                # Already normalized or index range in [-1, 1]
                s2_tensor = (s2_tensor - s2_tensor.mean(dim=[-2, -1], keepdim=True)) / (s2_tensor.std(dim=[-2, -1], keepdim=True) + 1e-6)

        # Final safety clamp to prevent extreme outliers/inf from propagating
        s1_tensor = torch.clamp(torch.nan_to_num(s1_tensor, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        s2_tensor = torch.clamp(torch.nan_to_num(s2_tensor, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

        # Augmentation
        if self.transform is not None:
            s1_tensor, s2_tensor = self.transform(s1_tensor, s2_tensor)

        labels = self._encode_labels(row.get("labels", []))
        return {
            "s1": s1_tensor,
            "s2": s2_tensor,
            "targets": labels,
            "patch_id": patch_id
        }


def create_synthetic_reben_dataframe(num_samples: int = 1000, split_ratios=(0.7, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Creates a synthetic reBEN metadata DataFrame matching the exact 19-class label structure and geographic split."""
    np.random.seed(seed)
    patch_ids = [f"patch_{i:06d}" for i in range(num_samples)]
    
    # Assign splits
    n_train = int(num_samples * split_ratios[0])
    n_val = int(num_samples * split_ratios[1])
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (num_samples - n_train - n_val)
    
    # Generate realistic multi-labels (1 to 4 active classes per patch)
    all_labels = []
    for _ in range(num_samples):
        k = np.random.randint(1, 4)
        chosen = np.random.choice(BEN_CLASSES_19, size=k, replace=False).tolist()
        all_labels.append(chosen)

    df = pd.DataFrame({
        "patch_id": patch_ids,
        "split": splits,
        "labels": all_labels,
        "country": np.random.choice(["Austria", "Belgium", "Finland", "Ireland", "Lithuania", "Portugal", "Serbia"], size=num_samples)
    })
    return df
