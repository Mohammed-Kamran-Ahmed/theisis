"""
SAR-Optical Paired Agricultural Dataset Collector (Google Earth Engine) -- v2
================================================================================
Same pipeline as before. Changes for expanding the dataset:

1. MORE REGIONS: 3 new AOIs added (Bihar Ganga plain, Tamil Nadu Cauvery
   delta, Gujarat Saurashtra) for additional geographic diversity --
   directly addresses "Limited geographic diversity" in your limitations.
   ⚠ These 3 bounding boxes are my best-effort approximate agricultural-belt
   guesses, NOT verified against actual field boundaries the way your
   original 4 AOIs presumably were. Sanity-check them in Google Earth /
   QGIS before a full run -- a bad bbox wastes GEE quota on ocean/urban/
   forest pixels instead of cropland.

2. MULTI-YEAR DATE RANGE: 2021-2024 instead of just 2023 -- directly
   addresses "Limited temporal diversity" (seasonal/crop-cycle coverage).

3. MORE PATCHES PER REGION: 6 -> 14.

4. RESUMABLE: skips any (region, patch_idx, date) combination whose output
   file already exists on disk. A multi-year, multi-region run is a LONG
   GEE job -- if it dies partway (quota, network, your laptop sleeping),
   rerunning the script won't re-download what you already have or
   duplicate rows in the metadata CSV.

5. RATE-LIMIT RESILIENCE: small delay + retry-with-backoff around each
   export, since GEE will throttle/reject requests if you hammer it,
   especially across this many region x date x patch combinations.

6. APPENDS to existing patch_metadata.csv instead of overwriting it, so
   your original 172 patches' metadata is preserved and the new patches
   are added alongside them.

Everything else (band math, speckle filter, entropy proxy caveat, direct
synchronous download) is UNCHANGED from your original script.

Setup: same as before (pip install earthengine-api geemap geopandas pandas,
earthengine authenticate, edit CONFIG, run).
"""

import ee
import geemap
import pandas as pd
import datetime as dt
import time
import os

# ----------------------------------------------------------------------
# CONFIG -- expanded for a bigger dataset
# ----------------------------------------------------------------------

AOIS = {
    # --- original 4 ---
    "punjab_wheat_rice":      [74.8, 30.2, 75.3, 30.6],   # Indo-Gangetic plain, dense canopy
    "krishna_godavari_rice":  [81.0, 16.4, 81.5, 16.8],   # AP delta, flooded paddy
    "vidarbha_cotton":        [77.8, 20.5, 78.3, 20.9],   # Maharashtra, sparse/semi-arid
    "kerala_backwaters_rice": [76.2, 9.3, 76.5, 9.6],     # tropical wet, high cloud cover

    # --- new: verify these bboxes before a full run, see module docstring ---
    "bihar_ganga_plain":      [85.6, 25.5, 86.1, 25.9],   # Bihar, dense multi-crop Ganga plain
    "tn_cauvery_delta":       [79.3, 10.6, 79.8, 11.0],   # Tamil Nadu, irrigated paddy delta
    "gujarat_saurashtra":     [70.8, 21.5, 71.3, 21.9],   # Gujarat, semi-arid groundnut/cotton
}

# Multi-year instead of single-year -- more seasonal/crop-cycle coverage.
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

MAX_CLOUD_PCT = 20
S1_ORBIT = "DESCENDING"
MATCH_WINDOW_DAYS = 5
PATCH_SIZE_M = 640
N_PATCHES_PER_REGION = 14      # was 6
EXPORT_SCALE_M = 10
GEE_PROJECT = "btp-project-493111"

METADATA_CSV = os.path.join(os.getcwd(), "patch_metadata.csv")
OUTPUT_DIR = os.path.join(os.getcwd(), "downloaded_patches")

EXPORT_RETRY_ATTEMPTS = 3
EXPORT_RETRY_BACKOFF_S = 15     # doubles each retry
DELAY_BETWEEN_EXPORTS_S = 2     # small pause to avoid hammering GEE

# ----------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------

def init_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)


def generate_patch_centers(bbox, n_patches):
    import math
    minlon, minlat, maxlon, maxlat = bbox
    n_side = max(1, int(math.sqrt(n_patches)))
    lons = [minlon + (maxlon - minlon) * (i + 1) / (n_side + 1) for i in range(n_side)]
    lats = [minlat + (maxlat - minlat) * (i + 1) / (n_side + 1) for i in range(n_side)]
    centers = [(lon, lat) for lon in lons for lat in lats]
    return centers[:n_patches]


def make_patch_geom(center_lon, center_lat, patch_size_m):
    point = ee.Geometry.Point([center_lon, center_lat])
    return point.buffer(patch_size_m / 2).bounds()


# ----------------------------------------------------------------------
# SAR (Sentinel-1) PROCESSING -- unchanged
# ----------------------------------------------------------------------

def get_s1_collection(aoi_geom, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", S1_ORBIT))
        .select(["VV", "VH"])
    )


def speckle_filter(img, radius=1):
    return img.focal_median(radius, "circle", "pixels")


def add_sar_descriptors(img):
    vv_db = img.select("VV")
    vh_db = img.select("VH")

    vv_lin = ee.Image(10).pow(vv_db.divide(10)).rename("VV_lin")
    vh_lin = ee.Image(10).pow(vh_db.divide(10)).rename("VH_lin")

    vh_vv_ratio = vh_lin.divide(vv_lin).rename("VH_VV_ratio")
    rvi = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename("RVI")

    total_power = vv_lin.add(vh_lin)
    p1 = vv_lin.divide(total_power)
    p2 = vh_lin.divide(total_power)
    entropy_proxy = (
        p1.multiply(p1.log().divide(ee.Image(2).log())).multiply(-1)
        .add(p2.multiply(p2.log().divide(ee.Image(2).log())).multiply(-1))
    ).rename("entropy_proxy")

    return img.addBands([vh_vv_ratio, rvi, entropy_proxy])


# ----------------------------------------------------------------------
# OPTICAL (Sentinel-2) PROCESSING -- unchanged
# ----------------------------------------------------------------------

def mask_s2_clouds(img):
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)


def get_s2_collection(aoi_geom, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
    )


def add_optical_descriptors(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))",
        {
            "NIR": img.select("B8").divide(10000),
            "RED": img.select("B4").divide(10000),
            "BLUE": img.select("B2").divide(10000),
        },
    ).rename("EVI")
    return img.addBands([ndvi, ndwi, evi])


# ----------------------------------------------------------------------
# PAIRING LOGIC -- unchanged
# ----------------------------------------------------------------------

def find_paired_dates(s1_col, s2_col):
    s1_dates = s1_col.aggregate_array("system:time_start").getInfo()
    s2_dates = s2_col.aggregate_array("system:time_start").getInfo()

    s1_dates = [dt.datetime.fromtimestamp(d / 1000, dt.timezone.utc) for d in s1_dates]
    s2_dates = [dt.datetime.fromtimestamp(d / 1000, dt.timezone.utc) for d in s2_dates]

    pairs = []
    for d1 in s1_dates:
        closest_s2 = min(s2_dates, key=lambda d2: abs((d2 - d1).days), default=None)
        if closest_s2 and abs((closest_s2 - d1).days) <= MATCH_WINDOW_DAYS:
            pairs.append((d1, closest_s2))
    return pairs


# ----------------------------------------------------------------------
# EXPORT -- with resumability + retry
# ----------------------------------------------------------------------

def already_downloaded(local_path):
    return os.path.exists(local_path) and os.path.getsize(local_path) > 0


def export_patch(region_name, patch_geom, patch_idx, s1_img, s2_img, s1_date, s2_date,
                  metadata_rows, output_dir):
    export_name = f"{region_name}_p{patch_idx}_{s1_date.strftime('%Y%m%d')}"
    local_path = os.path.join(output_dir, f"{export_name}.tif")

    if already_downloaded(local_path):
        print(f"  skip (already downloaded): {local_path}")
        metadata_rows.append({
            "region": region_name, "patch_idx": patch_idx,
            "s1_date": s1_date.isoformat(), "s2_date": s2_date.isoformat(),
            "date_gap_days": abs((s2_date - s1_date).days),
            "export_name": export_name, "scale_m": EXPORT_SCALE_M,
            "local_path": local_path, "file_size_bytes": os.path.getsize(local_path),
        })
        return local_path

    s1_processed = add_sar_descriptors(speckle_filter(s1_img))
    s2_processed = add_optical_descriptors(s2_img)

    stacked = s1_processed.resample("bilinear").reproject(crs="EPSG:4326", scale=EXPORT_SCALE_M)
    stacked = stacked.addBands(
        s2_processed.select(
            ["B2", "B3", "B4", "B8", "NDVI", "NDWI", "EVI"]
        ).resample("bilinear").reproject(crs="EPSG:4326", scale=EXPORT_SCALE_M)
    )
    stacked = stacked.toFloat().clip(patch_geom)

    last_err = None
    for attempt in range(1, EXPORT_RETRY_ATTEMPTS + 1):
        try:
            geemap.ee_export_image(
                stacked, filename=local_path, scale=EXPORT_SCALE_M,
                region=patch_geom, file_per_band=False,
            )
            if not already_downloaded(local_path):
                raise RuntimeError(f"export did not produce a file at {local_path} (silent geemap failure)")
            break
        except Exception as e:
            last_err = e
            wait = EXPORT_RETRY_BACKOFF_S * (2 ** (attempt - 1))
            print(f"  attempt {attempt}/{EXPORT_RETRY_ATTEMPTS} failed ({e}); retrying in {wait}s ...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"export failed after {EXPORT_RETRY_ATTEMPTS} attempts: {last_err}")

    metadata_rows.append({
        "region": region_name, "patch_idx": patch_idx,
        "s1_date": s1_date.isoformat(), "s2_date": s2_date.isoformat(),
        "date_gap_days": abs((s2_date - s1_date).days),
        "export_name": export_name, "scale_m": EXPORT_SCALE_M,
        "local_path": local_path, "file_size_bytes": os.path.getsize(local_path),
    })
    time.sleep(DELAY_BETWEEN_EXPORTS_S)
    return local_path


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    init_ee()
    metadata_rows = []
    downloaded = []
    failed = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_regions = len(AOIS)
    for region_num, (region_name, bbox) in enumerate(AOIS.items(), start=1):
        print(f"\n{'='*70}\n[{region_num}/{total_regions}] Starting region: {region_name}\n{'='*70}")
        try:
            aoi_geom = ee.Geometry.Rectangle(bbox)
            s1_col = get_s1_collection(aoi_geom, START_DATE, END_DATE)
            s2_col = get_s2_collection(aoi_geom, START_DATE, END_DATE)

            pairs = find_paired_dates(s1_col, s2_col)
            print(f"[{region_name}] found {len(pairs)} SAR-optical date pairs")

            patch_centers = generate_patch_centers(bbox, N_PATCHES_PER_REGION)
            print(f"[{region_name}] sampling {len(patch_centers)} patches per date pair "
                  f"({len(pairs) * len(patch_centers)} exports planned for this region)")

            for pair_num, (s1_date, s2_date) in enumerate(pairs, start=1):
                print(f"  [{region_name}] date pair {pair_num}/{len(pairs)}: "
                      f"S1={s1_date.date()} S2={s2_date.date()}")
                s1_img = (
                    get_s1_collection(aoi_geom, s1_date - dt.timedelta(hours=12), s1_date + dt.timedelta(hours=12))
                    .first()
                )
                s2_img = (
                    get_s2_collection(aoi_geom, s2_date - dt.timedelta(hours=12), s2_date + dt.timedelta(hours=12))
                    .first()
                )

                for patch_idx, (lon, lat) in enumerate(patch_centers):
                    patch_geom = make_patch_geom(lon, lat, PATCH_SIZE_M)
                    try:
                        local_path = export_patch(
                            region_name, patch_geom, patch_idx, s1_img, s2_img,
                            s1_date, s2_date, metadata_rows, OUTPUT_DIR,
                        )
                        downloaded.append(local_path)
                        print(f"    downloaded: {local_path}")
                    except Exception as e:
                        failed.append((region_name, patch_idx, str(s1_date.date()), str(e)))
                        print(f"    FAILED region={region_name} patch={patch_idx} date={s1_date.date()}: {e}")

        except Exception as region_err:
            # A region-level failure (bad AOI query, GEE quota/network hiccup,
            # empty collection, etc.) no longer kills the whole multi-region
            # run -- log it and move on to the next region.
            print(f"\n!! REGION-LEVEL FAILURE for {region_name}: {region_err}")
            print(f"!! Skipping to next region. Fix this region's AOI/dates and rerun later "
                  f"-- the resumable download logic will skip patches already fetched.\n")
            failed.append((region_name, "ALL", "region-level", str(region_err)))
            continue

    new_meta_df = pd.DataFrame(metadata_rows)

    if os.path.exists(METADATA_CSV):
        old_meta_df = pd.read_csv(METADATA_CSV)
        combined = pd.concat([old_meta_df, new_meta_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["export_name"], keep="first")
        print(f"\nMerged with existing metadata: {len(old_meta_df)} old + "
              f"{len(new_meta_df)} new -> {len(combined)} total (after de-duplication)")
    else:
        combined = new_meta_df

    combined.to_csv(METADATA_CSV, index=False)
    print(f"\nDownloaded {len(downloaded)} patches this run to {OUTPUT_DIR}")
    print(f"{len(failed)} patches failed (see printed FAILED lines above)")
    print(f"Metadata written to {METADATA_CSV} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()