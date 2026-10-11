"""
DFC2020 Loader + Feature Derivation -- v1
================================================================================
Loads GFM-Bench/DFC2020 (Hugging Face) and derives EXACTLY the same SAR/optical
feature set your GEE collector (v2) produces, so DFC2020 patches can sit
alongside your 172 GEE-collected patches in the same pipeline without forking
any downstream code (Stage B/CAHN/IERC, SCD, recluster_k67, compute_ari).

Why derive features instead of just using raw bands: your paper's SAR feature
space is 5-dim (VV, VH, VH/VV ratio, entropy_proxy, RVI) and your optical
targets are (NDVI_mean, NDWI_mean, EVI_mean, RVI_mean, VH_VV_ratio_mean) plus
3-band RGB reconstruction target. DFC2020 gives you raw VV/VH (2ch) and 13-band
Sentinel-2 -- this script turns that into the same 5+3(+3 physical) shape your
encoder/decoders already expect, using the IDENTICAL formulas from your GEE
script's add_sar_descriptors() / add_optical_descriptors(), so a patch from
DFC2020 and a patch from GEE are numerically comparable, not just shape-compatible.

Key difference from your GEE patches: DFC2020 additionally gives you a REAL
per-pixel land-cover label (8 classes, Robinson et al. 2021) -- this script
reduces that to a patch-level majority label, which is what you plug into
compute_ari.py as ground truth instead of your hand-crafted GMM labels.

Output: mirrors your GEE script's contract --
  - downloaded_patches/DFC2020_p<idx>.npy   (9, 128, 128) float32: [SAR x5, RGB x3, true_label x1]
  - appended rows in patch_metadata.csv with region="DFC2020_<split>", plus a
    `true_label` column your original GEE rows don't have (left blank for them).

Run:
    python dfc2020_loader.py --split train --limit 2000 --out_dir . 
    # start small (--limit) to sanity check before pulling the full 41,537/4,615/8,874 split sizes
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# DFC2020 (Robinson et al. 2021) 8-class scheme, per the dataset card / dfc2020_baseline
# --------------------------------------------------------------------------
DFC2020_CLASSES = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
}
IGNORE_LABEL = 255  # common convention for "no data" pixels; adjust if GFM-Bench uses a different sentinel


# --------------------------------------------------------------------------
# SAR feature derivation -- IDENTICAL formulas to your GEE add_sar_descriptors()
# --------------------------------------------------------------------------
def derive_sar_features(vv_db: np.ndarray, vh_db: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """vv_db, vh_db: (H,W) arrays in dB (Sentinel-1 GRD convention, matches your GEE VV/VH bands).
    Returns (5,H,W): [VV_db, VH_db, VH_VV_ratio, entropy_proxy, RVI] -- same 5 channels/order
    as your paper's hand-crafted SAR feature space (Section 3).
    """
    vv_lin = np.power(10.0, vv_db / 10.0)
    vh_lin = np.power(10.0, vh_db / 10.0)

    vh_vv_ratio = vh_lin / (vv_lin + eps)
    rvi = (4.0 * vh_lin) / (vv_lin + vh_lin + eps)

    total_power = vv_lin + vh_lin + eps
    p1 = vv_lin / total_power
    p2 = vh_lin / total_power
    # same polarization-power entropy proxy as your GEE script (NOT a GLCM texture
    # measure -- kept identical on purpose so DFC2020 and GEE patches are comparable)
    entropy_proxy = -(p1 * np.log2(p1 + eps) + p2 * np.log2(p2 + eps))

    return np.stack([vv_db, vh_db, vh_vv_ratio, entropy_proxy, rvi], axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Optical feature derivation -- IDENTICAL formulas to your GEE add_optical_descriptors()
# --------------------------------------------------------------------------
def derive_optical_features(bands: dict) -> dict:
    """bands: dict of band_name -> (H,W) reflectance arrays, Sentinel-2 L2A convention
    (raw DN, NOT yet /10000). Returns dict with rgb (3,H,W) in [0,1] and the three
    physical indices (H,W) each, matching your GEE add_optical_descriptors() exactly.
    """
    b2, b3, b4, b8 = bands["B02"], bands["B03"], bands["B04"], bands["B08"]
    eps = 1e-6

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)

    nir = b8 / 10000.0
    red = b4 / 10000.0
    blue = b2 / 10000.0
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps))

    # RGB reconstruction target, normalized like your paper's 2nd/98th percentile
    # approach -- percentile bounds computed at dataset-build time, see normalize_rgb()
    rgb = np.stack([b4, b3, b2], axis=0).astype(np.float32)  # R,G,B order matching your GEE B4,B3,B2

    return {"rgb": rgb, "NDVI": ndvi.astype(np.float32), "NDWI": ndwi.astype(np.float32), "EVI": evi.astype(np.float32)}


def normalize_rgb(rgb: np.ndarray, low: np.ndarray = None, high: np.ndarray = None):
    """Per-channel 2nd/98th percentile normalization, matching paper Section 3
    ("Optical reflectance is normalized per channel using the 2nd/98th percentile
    of the training split only ... this same normalization is re-applied at test time").
    Pass low/high computed on your DFC2020 TRAIN split to normalize val/test consistently.
    """
    if low is None or high is None:
        low = np.percentile(rgb, 2, axis=(1, 2), keepdims=True)
        high = np.percentile(rgb, 98, axis=(1, 2), keepdims=True)
    rgb_norm = (rgb - low) / (high - low + 1e-6)
    return np.clip(rgb_norm, 0.0, 1.0).astype(np.float32), low, high


# --------------------------------------------------------------------------
# Patch-level label reduction (per-pixel mask -> single majority label)
# --------------------------------------------------------------------------
def patch_majority_label(label_mask: np.ndarray) -> int:
    """Your pipeline clusters at the PATCH level (GMM on patch-level SAR stats),
    so reduce DFC2020's per-pixel label mask to one label per patch via majority
    vote, ignoring IGNORE_LABEL pixels. This is what feeds compute_ari.py as the
    real-label array to compare against your GMM cluster assignments."""
    valid = label_mask[label_mask != IGNORE_LABEL]
    if valid.size == 0:
        return -1  # no valid pixels in this patch; caller should drop it
    vals, counts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(counts)])


# --------------------------------------------------------------------------
# Resize 96x96 -> 128x128 (GFM-Bench's DFC2020 repack is 96x96; your pipeline is 128x128)
# --------------------------------------------------------------------------
def resize_chw(arr: np.ndarray, size: int = 128) -> np.ndarray:
    """arr: (C,H,W). Bilinear resize per channel, nearest-neighbor for label masks
    (pass is_label=True in the caller) to avoid inventing fractional class ids."""
    from skimage.transform import resize
    c = arr.shape[0]
    out = np.zeros((c, size, size), dtype=arr.dtype)
    for i in range(c):
        out[i] = resize(arr[i], (size, size), order=1, preserve_range=True, anti_aliasing=True)
    return out


def resize_label(label_mask: np.ndarray, size: int = 128) -> np.ndarray:
    from skimage.transform import resize
    return resize(label_mask, (size, size), order=0, preserve_range=True, anti_aliasing=False).astype(label_mask.dtype)


# --------------------------------------------------------------------------
# Main processing loop -- mirrors your GEE script's resumability + metadata append pattern
# --------------------------------------------------------------------------
def process_split(split: str, out_dir: str, limit: int = None, resize_to: int = 128):
    from datasets import load_dataset

    print(f"[DFC2020] loading split='{split}' from GFM-Bench/DFC2020 ...")
    ds = load_dataset(
    "GFM-Bench/DFC2020",
    split=split,
    download_mode="force_redownload",
    trust_remote_code=True
)
    n_total = len(ds) if limit is None else min(limit, len(ds))
    print(f"[DFC2020] processing {n_total} of {len(ds)} samples")

    output_dir = os.path.join(out_dir, "downloaded_patches")
    metadata_csv = os.path.join(out_dir, "patch_metadata.csv")
    os.makedirs(output_dir, exist_ok=True)

    # compute RGB normalization stats on TRAIN split only, cache for val/test reuse
    norm_stats_path = os.path.join(out_dir, "dfc2020_rgb_norm_stats.npz")

    metadata_rows = []
    skipped_no_valid_label = 0
    skipped_already_done = 0

    rgb_low, rgb_high = None, None
    if split != "train" and os.path.exists(norm_stats_path):
        stats = np.load(norm_stats_path)
        rgb_low, rgb_high = stats["low"], stats["high"]
        print(f"[DFC2020] loaded RGB normalization stats from {norm_stats_path} (train-split percentiles)")
    elif split != "train":
        print("[DFC2020] WARNING: no cached train-split RGB norm stats found -- "
              "run --split train first, or this split will compute its own percentiles "
              "(violates the 'train-only normalization, reapplied at test time' rule "
              "in your paper's Section 3 -- test-set leakage risk).")

    all_train_rgb_for_stats = [] if split == "train" else None

    for idx in range(n_total):
        export_name = f"DFC2020_{split}_p{idx}"
        local_path = os.path.join(output_dir, f"{export_name}.npy")

        if os.path.exists(local_path):
            skipped_already_done += 1
            continue

        sample = ds[idx]
        radar = np.array(sample["radar"], dtype=np.float32)      # expected (2,H,W): VV,VH
        optical = np.array(sample["optical"], dtype=np.float32)  # expected (13,H,W)
        label = np.array(sample["label"], dtype=np.int64)        # (H,W)

        if radar.ndim != 3 or radar.shape[0] != 2:
            raise ValueError(f"Unexpected radar shape {radar.shape} at idx {idx} -- "
                              f"check GFM-Bench/DFC2020's current feature layout (schema may "
                              f"have changed since this script was written).")
        if optical.shape[0] != 13:
            raise ValueError(f"Unexpected optical band count {optical.shape} at idx {idx} -- "
                              f"expected 13 Sentinel-2 bands per the dataset card.")

        majority_label = patch_majority_label(label)
        if majority_label == -1:
            skipped_no_valid_label += 1
            continue

        sar_feats = derive_sar_features(radar[0], radar[1])

        band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                      "B08", "B8A", "B09", "B10", "B11", "B12"]
        bands = {name: optical[i] for i, name in enumerate(band_names)}
        opt_feats = derive_optical_features(bands)

        if split == "train":
            all_train_rgb_for_stats.append(opt_feats["rgb"])

        rgb_norm, low_used, high_used = normalize_rgb(opt_feats["rgb"], rgb_low, rgb_high)

        stacked = np.concatenate([
            sar_feats,                                  # 5 x H x W
            rgb_norm,                                    # 3 x H x W
            label[None, :, :].astype(np.float32),        # 1 x H x W (pixel-level, kept for optional segmentation use)
        ], axis=0)

        if resize_to is not None:
            geo_part = resize_chw(stacked[:8], size=resize_to)
            label_part = resize_label(stacked[8], size=resize_to)[None, :, :]
            stacked = np.concatenate([geo_part, label_part], axis=0)

        np.save(local_path, stacked)

        metadata_rows.append({
            "region": f"DFC2020_{split}",
            "patch_idx": idx,
            "s1_date": "",  # not provided per-patch by GFM-Bench repack
            "s2_date": "",
            "date_gap_days": "",
            "export_name": export_name,
            "scale_m": 10,
            "local_path": local_path,
            "file_size_bytes": os.path.getsize(local_path),
            "true_label": majority_label,
            "true_label_name": DFC2020_CLASSES.get(majority_label, "unknown"),
            "NDVI_mean": float(np.nanmean(opt_feats["NDVI"])),
            "NDWI_mean": float(np.nanmean(opt_feats["NDWI"])),
            "EVI_mean": float(np.nanmean(opt_feats["EVI"])),
            "RVI_mean": float(np.nanmean(sar_feats[4])),
            "VH_VV_ratio_mean": float(np.nanmean(sar_feats[2])),
        })

        if (idx + 1) % 200 == 0:
            print(f"  [{split}] {idx + 1}/{n_total} processed "
                  f"({skipped_no_valid_label} skipped: no valid label pixels)")

    if split == "train" and all_train_rgb_for_stats:
        stacked_rgb = np.stack(all_train_rgb_for_stats, axis=0)  # (N,3,H,W)
        low = np.percentile(stacked_rgb, 2, axis=(0, 2, 3), keepdims=False).reshape(3, 1, 1)
        high = np.percentile(stacked_rgb, 98, axis=(0, 2, 3), keepdims=False).reshape(3, 1, 1)
        np.savez(norm_stats_path, low=low, high=high)
        print(f"[DFC2020] saved train-split RGB normalization stats to {norm_stats_path} "
              f"(re-run process_split for val/test AFTER this, so they reuse these stats)")

    new_meta_df = pd.DataFrame(metadata_rows)
    if os.path.exists(metadata_csv) and len(new_meta_df) > 0:
        old_meta_df = pd.read_csv(metadata_csv)
        combined = pd.concat([old_meta_df, new_meta_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["export_name"], keep="first")
    elif len(new_meta_df) > 0:
        combined = new_meta_df
    else:
        combined = pd.read_csv(metadata_csv) if os.path.exists(metadata_csv) else pd.DataFrame()

    if len(combined) > 0:
        combined.to_csv(metadata_csv, index=False)

    print(f"\n[DFC2020] split='{split}' done: {len(metadata_rows)} new patches written, "
          f"{skipped_already_done} already-done skipped, "
          f"{skipped_no_valid_label} skipped (no valid label pixels)")
    print(f"[DFC2020] metadata -> {metadata_csv}")
    return metadata_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=None, help="cap samples processed (start small to sanity check)")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--resize_to", type=int, default=128)
    args = p.parse_args()

    t0 = time.time()
    process_split(args.split, args.out_dir, limit=args.limit, resize_to=args.resize_to)
    print(f"[DFC2020] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
