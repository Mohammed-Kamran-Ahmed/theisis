"""
spatial_leakage_check_v2.py
=============================
The first leakage checker only compared region NAMES between train/test,
which is the wrong granularity now that spatial_resplit.py deliberately
keeps each region in both splits -- separated by grid cell, not by
region. This script checks what actually matters: the real geographic
distance between the nearest train patch and the nearest test patch,
using the centroid_lon/centroid_lat columns spatial_resplit.py added.

For each region, reports:
  - the minimum distance (in meters) between any train patch and any
    test patch
  - how that compares to your patch size (10m pixels, patches are
    likely 64-256px => ~640m-2560m across)

A minimum distance close to 0 means a train and test patch are
adjacent/overlapping despite being in different grid cells -- that IS
leakage, just at the grid-cell-boundary level rather than the region
level. A minimum distance comfortably larger than one patch width is
real separation.

Run with no arguments:

    python spatial_leakage_check_v2.py
"""
import numpy as np
import pandas as pd

CSV_PATH = "./patch_metadata_clustered.csv"

# Rough conversion: 1 degree latitude ~= 111,000 m. Longitude scales by
# cos(latitude); we approximate per-region using each region's mean
# latitude, which is accurate enough for a same-region distance check.
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} patches from {CSV_PATH}\n")

    if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
        print("❌ No centroid_lon/centroid_lat columns found. Run "
              "spatial_resplit.py first -- it computes and saves these.")
        return

    df = df.dropna(subset=["centroid_lon", "centroid_lat"])

    print(f"{'Region':<28}{'Min train-test dist (m)':>26}{'Verdict':>14}")
    print("-" * 68)

    any_leak = False
    for region, g in df.groupby("region"):
        train = g[g["split"] == "train"]
        test = g[g["split"] == "test"]
        if len(train) == 0 or len(test) == 0:
            print(f"{region:<28}{'(no train or no test patches)':>26}")
            continue

        min_dist = np.inf
        # Small N per region (dozens to low hundreds) -- brute force is fine.
        for _, trow in test.iterrows():
            dists = haversine_m(
                trow["centroid_lat"], trow["centroid_lon"],
                train["centroid_lat"].values, train["centroid_lon"].values,
            )
            min_dist = min(min_dist, dists.min())

        # Sentinel-1/2 patches here are 10m/pixel; without a stored patch
        # size, treat anything under 500m as suspicious adjacency.
        verdict = "🔴 TOO CLOSE" if min_dist < 500 else "✅ separated"
        if min_dist < 500:
            any_leak = True
        print(f"{region:<28}{min_dist:>26.1f}{verdict:>14}")

    print("\n" + ("🔴 Some regions have train/test patches within 500m of "
                   "each other -- treat those regions' held-out metrics "
                   "with caution."
                   if any_leak else
                   "✅ All regions show real geographic separation "
                   "(>500m) between every train and test patch."))


if __name__ == "__main__":
    main()