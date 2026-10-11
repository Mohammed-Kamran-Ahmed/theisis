"""
Phase 2 — Patch Validation & Descriptor Summary
=================================================

Purpose
-------
Once GeoTIFFs from Phase 1 are downloaded from Google Drive into a local
folder, this script:
  1. Loads every patch and checks it has all 12 expected bands.
  2. Sanity-checks value ranges (e.g. NDVI must be in [-1, 1]).
  3. Detects dead/masked optical blackouts cleanly (low reflectance / flatband).
  4. Flags patches with excessive NaNs or missing bands.
  5. Computes per-patch descriptor summary stats (mean/std/min/max) and
     merges them into patch_metadata.csv.
  6. Saves visual sanity-check figures without Matplotlib NaN-cast warnings.

Setup
-----
pip install rasterio numpy pandas matplotlib

Usage
-----
python validate_patches.py
"""

import os
import sys
import glob
import warnings

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# Suppress annoying Matplotlib runtime warnings for NaN casting
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

LOCAL_PATCH_DIR = "./downloaded_patches"       # <-- where you synced the GeoTIFFs
METADATA_CSV = "./patch_metadata.csv"          # <-- from Phase 1
OUTPUT_METADATA_CSV = "./patch_metadata_validated.csv"
SAMPLE_PLOT_DIR = "./sanity_check_plots"
N_SAMPLE_PLOTS = 6                             # how many patches to visually spot-check

EXPECTED_BANDS = [
    "VV", "VH", "VH_VV_ratio", "RVI", "entropy_proxy",
    "B2", "B3", "B4", "B8", "NDVI", "NDWI", "EVI",
]

OPTICAL_BANDS = ["B2", "B3", "B4", "B8"]

# Expected sane value ranges per band — outside these = flag for review
RANGE_CHECKS = {
    "NDVI": (-1.0, 1.0),
    "NDWI": (-1.0, 1.0),
    "EVI": (-2.0, 2.0),          # EVI can exceed [-1,1] slightly in practice
    "entropy_proxy": (0.0, 1.0),
    "VH_VV_ratio": (0.0, 5.0),   # generous upper bound; true outliers still flagged
}


# ----------------------------------------------------------------------
# BAND MAPPING HELPER
# ----------------------------------------------------------------------

def get_band_mapping(src):
    """
    Returns a dictionary mapping band names to 1-based rasterio band indices.
    Falls back to deterministic position index if GDAL descriptions are missing/None.
    """
    descriptions = list(src.descriptions)
    has_descriptions = any(d is not None for d in descriptions)
    
    band_index = {}
    if has_descriptions:
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                band_index[desc] = idx
    else:
        # Fallback to positional mapping matching export order
        for idx, name in enumerate(EXPECTED_BANDS, start=1):
            if idx <= src.count:
                band_index[name] = idx
                
    return band_index


# ----------------------------------------------------------------------
# CORE VALIDATION
# ----------------------------------------------------------------------

def validate_patch(filepath):
    """Opens one patch GeoTIFF, checks bands/ranges/zeros, returns a summary dict."""
    result = {
        "filepath": filepath,
        "valid": True,
        "issues": [],
    }

    try:
        with rasterio.open(filepath) as src:
            band_index = get_band_mapping(src)
            data = src.read()  # shape: (bands, H, W)
            total_bands = src.count
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"failed to open: {e}")
        return result

    # 1. Band count & presence check
    missing = [b for b in EXPECTED_BANDS if b not in band_index]
    if missing or total_bands < len(EXPECTED_BANDS):
        result["valid"] = False
        result["issues"].append(f"missing bands or count mismatch (found {total_bands}/{len(EXPECTED_BANDS)}): {missing}")

    # 2. NaN / Inf check across all bands
    nan_frac = np.isnan(data).mean()
    inf_frac = np.isinf(data).mean()
    result["nan_fraction"] = float(nan_frac)
    result["inf_fraction"] = float(inf_frac)
    if nan_frac > 0.5:
        result["valid"] = False
        result["issues"].append(f"excessive NaNs: {nan_frac:.2%}")

    # 3. Robust check for optical blackout (e.g. cloud-masked or missing scene)
    zero_optical = []
    for ob in OPTICAL_BANDS:
        if ob in band_index:
            idx = band_index[ob] - 1
            arr = data[idx]
            valid_arr = arr[np.isfinite(arr)]
            
            # Flag if band is completely dead/flat or maximum reflectance is essentially 0
            if valid_arr.size == 0 or np.nanmax(valid_arr) <= 0 or np.nanstd(valid_arr) == 0:
                zero_optical.append(ob)
    
    if zero_optical:
        result["valid"] = False
        result["issues"].append(f"optical blackout / dead bands: {zero_optical}")

    # 4. Per-band stats + range checks
    for band in EXPECTED_BANDS:
        if band not in band_index:
            continue
            
        # rasterio uses 1-based indexing, array is 0-indexed
        idx = band_index[band] - 1
        arr = data[idx].astype("float64")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            result["issues"].append(f"{band}: all-invalid pixels")
            continue

        result[f"{band}_mean"] = float(np.mean(arr))
        result[f"{band}_std"] = float(np.std(arr))
        result[f"{band}_min"] = float(np.min(arr))
        result[f"{band}_max"] = float(np.max(arr))

        if band in RANGE_CHECKS:
            lo, hi = RANGE_CHECKS[band]
            out_of_range_frac = ((arr < lo) | (arr > hi)).mean()
            if out_of_range_frac > 0.05:  # more than 5% of pixels out of range
                result["issues"].append(
                    f"{band}: {out_of_range_frac:.1%} of pixels outside expected [{lo},{hi}]"
                )

    return result


def plot_sample(filepath, out_dir):
    """Saves a quick RGB / NDVI / VV panel for visual co-registration check."""
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(filepath) as src:
        band_index = get_band_mapping(src)
        data = src.read()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # RGB (B4=Red, B3=Green, B2=Blue), simple percentile stretch
    try:
        b4_idx = band_index["B4"] - 1
        b3_idx = band_index["B3"] - 1
        b2_idx = band_index["B2"] - 1
        
        rgb = np.stack(
            [data[b4_idx], data[b3_idx], data[b2_idx]],
            axis=-1,
        ).astype("float64")
        
        # Safe cleaning for NaN/Inf before plotting
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb = np.clip(rgb / 3000.0, 0, 1)  # rough reflectance scaling for S2 SR
        axes[0].imshow(rgb)
        axes[0].set_title("Optical RGB")
    except KeyError:
        axes[0].set_title("Optical RGB (bands missing)")
    axes[0].axis("off")

    if "NDVI" in band_index:
        ndvi_idx = band_index["NDVI"] - 1
        ndvi_data = np.nan_to_num(data[ndvi_idx], nan=0.0, posinf=0.0, neginf=0.0)
        im1 = axes[1].imshow(ndvi_data, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1].set_title("NDVI")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].axis("off")

    if "VV" in band_index:
        vv_idx = band_index["VV"] - 1
        vv_data = np.nan_to_num(data[vv_idx], nan=-30.0, posinf=0.0, neginf=-30.0)
        im2 = axes[2].imshow(vv_data, cmap="gray")
        axes[2].set_title("SAR VV (dB)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, os.path.basename(filepath).replace(".tif", ".png"))
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    patch_files = sorted(glob.glob(os.path.join(LOCAL_PATCH_DIR, "*.tif")))
    if not patch_files:
        print(f"No .tif files found in {LOCAL_PATCH_DIR} — check the path/sync.")
        return

    print(f"Found {len(patch_files)} patch files. Validating...")

    results = [validate_patch(fp) for fp in patch_files]
    results_df = pd.DataFrame(results)

    n_valid = results_df["valid"].sum()
    n_flagged = len(results_df) - n_valid
    print(f"\n{n_valid}/{len(results_df)} patches passed validation cleanly.")
    if n_flagged:
        print(f"⚠️  {n_flagged} patches flagged — see 'issues' column in output CSV.")

    # Merge with Phase 1 metadata if it exists
    if os.path.exists(METADATA_CSV):
        meta_df = pd.read_csv(METADATA_CSV)
        meta_df["export_name"] = meta_df["export_name"].astype(str)
        results_df["export_name"] = results_df["filepath"].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
        merged = meta_df.merge(results_df, on="export_name", how="left")
        merged.to_csv(OUTPUT_METADATA_CSV, index=False)
        print(f"Merged validated descriptor stats written to {OUTPUT_METADATA_CSV}")
    else:
        results_df.to_csv(OUTPUT_METADATA_CSV, index=False)
        print(f"(No Phase 1 metadata found at {METADATA_CSV} — wrote validation-only CSV)")

    # Sample sanity-check plots (prefer valid patches for clean display)
    sample_files = [fp for fp, r in zip(patch_files, results) if r["valid"]][:N_SAMPLE_PLOTS]
    if not sample_files:
        # Fallback to any files if no valid ones exist
        sample_files = patch_files[:N_SAMPLE_PLOTS]

    for fp in sample_files:
        out_path = plot_sample(fp, SAMPLE_PLOT_DIR)
        print(f"Saved sanity-check plot: {out_path}")


if __name__ == "__main__":
    main()