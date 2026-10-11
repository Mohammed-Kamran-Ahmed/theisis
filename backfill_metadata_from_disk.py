"""
backfill_metadata_from_disk.py
=================================
Fixes the root cause of "why is it still only 172 patches": your fetch
script only writes patch_metadata.csv ONCE, at the very end of the entire
multi-region run. If that run was interrupted anywhere (crash, Ctrl+C,
GEE quota error, killed terminal) before reaching that final write, every
.tif file downloaded up to that point is a real, valid file on disk --
but its metadata row only ever existed in memory and was never saved.
Downstream scripts (validate_patches.py, train_gmm_clustering.py) only
know about rows in the CSV, so they silently ignore hundreds of real,
already-downloaded patches.

This script does NOT re-download anything. It scans every .tif already
in downloaded_patches/, and for any file missing from patch_metadata.csv,
reconstructs its metadata row directly from the filename (which encodes
region, patch index, and S1 date by construction:
    {region}_p{patch_idx}_{YYYYMMDD}.tif
), then merges it into patch_metadata.csv.

Reconstructed rows get s1_date == s2_date (the exact optical acquisition
date isn't recoverable from the filename alone -- date_gap_days is left
as unknown/0 and a `reconstructed=True` flag column is added so you can
see exactly which rows came from this recovery path vs. the original
export_patch() call, if that distinction ever matters for a methods note).

Usage:
    python backfill_metadata_from_disk.py
"""

import os
import re
import glob
import pandas as pd
import datetime as dt

PATCH_DIR = "./downloaded_patches"
METADATA_CSV = "./patch_metadata.csv"
EXPORT_SCALE_M = 10  # must match your fetch script's EXPORT_SCALE_M

FILENAME_PATTERN = re.compile(r"^(?P<region>.+)_p(?P<patch_idx>\d+)_(?P<date>\d{8})$")


def parse_filename(stem):
    m = FILENAME_PATTERN.match(stem)
    if not m:
        return None
    region = m.group("region")
    patch_idx = int(m.group("patch_idx"))
    date_str = m.group("date")
    try:
        date = dt.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return region, patch_idx, date


def main():
    all_tifs = sorted(glob.glob(os.path.join(PATCH_DIR, "*.tif")))
    print(f"Found {len(all_tifs)} .tif files on disk in {PATCH_DIR}")

    if os.path.exists(METADATA_CSV):
        existing_df = pd.read_csv(METADATA_CSV)
        existing_names = set(existing_df["export_name"].astype(str))
        print(f"Existing patch_metadata.csv has {len(existing_df)} rows")
    else:
        existing_df = pd.DataFrame()
        existing_names = set()
        print("No existing patch_metadata.csv found -- will create one from scratch.")

    new_rows = []
    unparseable = []

    for tif_path in all_tifs:
        export_name = os.path.splitext(os.path.basename(tif_path))[0]
        if export_name in existing_names:
            continue  # already has a metadata row, nothing to backfill

        parsed = parse_filename(export_name)
        if parsed is None:
            unparseable.append(export_name)
            continue

        region, patch_idx, s1_date = parsed
        new_rows.append({
            "region": region,
            "patch_idx": patch_idx,
            "s1_date": s1_date.isoformat(),
            "s2_date": s1_date.isoformat(),   # not recoverable from filename alone -- see docstring
            "date_gap_days": 0,                # unknown; filename doesn't encode the real gap
            "export_name": export_name,
            "scale_m": EXPORT_SCALE_M,
            "local_path": tif_path,
            "file_size_bytes": os.path.getsize(tif_path),
            "reconstructed": True,
        })

    print(f"\n{len(new_rows)} orphaned files found on disk with NO metadata row -- backfilling now.")
    if unparseable:
        print(f"{len(unparseable)} files did not match the expected naming pattern "
              f"({{region}}_p{{idx}}_{{YYYYMMDD}}.tif) and were skipped:")
        for name in unparseable[:10]:
            print(f"   {name}")
        if len(unparseable) > 10:
            print(f"   ... and {len(unparseable) - 10} more")

    if not new_rows:
        print("\nNothing to backfill -- patch_metadata.csv already covers every file on disk.")
        return

    new_df = pd.DataFrame(new_rows)

    if not existing_df.empty and "reconstructed" not in existing_df.columns:
        existing_df["reconstructed"] = False

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["export_name"], keep="first")

    combined.to_csv(METADATA_CSV, index=False)
    print(f"\nWrote {len(combined)} total rows to {METADATA_CSV} "
          f"({len(existing_df)} original + {len(new_rows)} recovered from disk).")
    print("\nNext step: rerun validate_patches.py -> train_gmm_clustering.py -> "
          "utils/data_split.py -> ... with this corrected metadata file.")


if __name__ == "__main__":
    main()