"""
spatial_resplit.py
====================
Your current train/test split lets the same geographic region appear
in both partitions -- confirmed by spatial_leakage_check.py. This
script fixes that by:

  1. Reading each patch's real geographic centroid from its GeoTIFF
     (via rasterio), since the metadata CSV doesn't store coordinates.
  2. Dividing each region into a spatial grid, and assigning whole
     grid CELLS (not individual patches) to train or test -- so
     train and test patches are never from the same neighborhood.
  3. Overwriting the 'split' column in patch_metadata_clustered.csv
     (after backing up the old, leaky split) so the rest of your
     pipeline (Stage 4 onward) picks it up automatically.

Run once, before re-running the training pipeline:

    python spatial_resplit.py
    python stage4_contrastive_learning.py
    python stage4b_recluster_and_finetune.py
    python cca_analysis.py --mode full_comparison

IMPORTANT CAVEAT: this only fixes the split used by Stages 4/4b/CCA.
Your Stage 3 GMM baseline + the optical reflectance normalization
(2nd/98th percentile) were computed on the OLD train split. Re-run
Stage 3's normalization/GMM fit on the NEW train partition too, or
that baseline will still be quietly leaking test-region statistics.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

METADATA_CSV = "./patch_metadata_clustered.csv"
BACKUP_CSV = "./patch_metadata_clustered_ORIGINAL_LEAKY_SPLIT_BACKUP.csv"
PATCH_DIR = "./downloaded_patches"
GRID_CELLS_PER_AXIS = 6      # divides each region into a 6x6 spatial grid
TEST_FRACTION = 0.20
SEED = 42


def get_patch_centroid(path):
    with rasterio.open(path) as src:
        bounds = src.bounds
        if src.crs is not None and src.crs.to_epsg() != 4326:
            bounds = transform_bounds(src.crs, "EPSG:4326", *bounds)
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        return cx, cy


def resolve_path(row):
    for col in ["filepath", "local_path", "export_name"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col])
            if os.path.exists(val):
                return val
            rel = os.path.join(PATCH_DIR, os.path.basename(val))
            if os.path.exists(rel):
                return rel
    return None


def main():
    df = pd.read_csv(METADATA_CSV)
    df.to_csv(BACKUP_CSV, index=False)
    print(f"Backed up original (leaky) split to {BACKUP_CSV}\n")

    print("Reading patch centroids from raster files (this reads every .tif once)...")
    lons, lats = [], []
    n_failed = 0
    for _, row in df.iterrows():
        path = resolve_path(row)
        if path is None:
            lons.append(np.nan)
            lats.append(np.nan)
            n_failed += 1
            continue
        try:
            cx, cy = get_patch_centroid(path)
        except Exception as e:
            lons.append(np.nan)
            lats.append(np.nan)
            n_failed += 1
            continue
        lons.append(cx)
        lats.append(cy)

    df["centroid_lon"] = lons
    df["centroid_lat"] = lats
    if n_failed:
        print(f"⚠️ Could not read coordinates for {n_failed}/{len(df)} patches "
              f"(kept their old split assignment as a fallback).\n")

    rng = np.random.RandomState(SEED)
    new_split = pd.Series(index=df.index, dtype=object)

    for region, g in df.groupby("region"):
        g = g.dropna(subset=["centroid_lon", "centroid_lat"])
        if len(g) == 0:
            continue

        lon_bins = np.linspace(g["centroid_lon"].min(), g["centroid_lon"].max(), GRID_CELLS_PER_AXIS + 1)
        lat_bins = np.linspace(g["centroid_lat"].min(), g["centroid_lat"].max(), GRID_CELLS_PER_AXIS + 1)
        cell_lon = np.clip(np.digitize(g["centroid_lon"], lon_bins) - 1, 0, GRID_CELLS_PER_AXIS - 1)
        cell_lat = np.clip(np.digitize(g["centroid_lat"], lat_bins) - 1, 0, GRID_CELLS_PER_AXIS - 1)
        cell_id = cell_lon * GRID_CELLS_PER_AXIS + cell_lat

        unique_cells = np.unique(cell_id)
        rng.shuffle(unique_cells)
        n_test_cells = max(1, int(round(len(unique_cells) * TEST_FRACTION)))
        test_cells = set(unique_cells[:n_test_cells])

        region_split = pd.Series(
            ["test" if c in test_cells else "train" for c in cell_id],
            index=g.index,
        )
        new_split.loc[g.index] = region_split

        n_train = int((region_split == "train").sum())
        n_test = int((region_split == "test").sum())
        print(f"{region}: {len(unique_cells)} occupied grid cells, "
              f"{n_test_cells} assigned to TEST -> {n_train} train / {n_test} test patches")

    df["split"] = new_split.fillna(df["split"])
    df.to_csv(METADATA_CSV, index=False)

    print(f"\nSaved spatially-blocked split to {METADATA_CSV}")
    print("\nNew patches per region (train / test):")
    print(df.groupby(["region", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()