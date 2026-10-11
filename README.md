# SAR-Optical Paired Dataset Collector

Fetches co-registered Sentinel-1 (SAR) + Sentinel-2 (optical) patches over
diverse agricultural AOIs, with descriptor bands (VH/VV, RVI, entropy proxy,
NDVI, NDWI, EVI) computed and stacked into a single export, plus a metadata
CSV indexing every patch.

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate
```

You'll also need a Google Cloud project with the Earth Engine API enabled —
set its ID as `GEE_PROJECT` in `fetch_sar_optical_data.py`. Free to create at
https://console.cloud.google.com if you don't have one.

## Configure

Open `fetch_sar_optical_data.py` and edit the `CONFIG` section at the top:

- `AOIS` — dict of region name -> bounding box `[min_lon, min_lat, max_lon, max_lat]`.
  The three example regions are placeholders for crop-type diversity (wheat /
  rice / mixed) — replace with your actual thesis AOIs.
- `START_DATE` / `END_DATE` — date range to search for imagery.
- `MAX_CLOUD_PCT` — optical cloud filter.
- `MATCH_WINDOW_DAYS` — how close SAR and optical acquisition dates must be
  to count as a "pair" (5 days is a reasonable start for cropland).

## Run

```bash
python fetch_sar_optical_data.py
```

This submits export tasks to Google Earth Engine, which writes multi-band
GeoTIFFs to a folder in your Google Drive (`DRIVE_FOLDER`). Exports run
server-side and can take minutes to hours depending on AOI size and count —
monitor progress at https://code.earthengine.google.com/tasks.

A `patch_metadata.csv` is written locally, indexing every exported patch
(region, matched dates, date gap, export name) — this is your dataset index
for the downstream GMM/regression/embedding pipeline.

## Output bands per patch

| Band | Source | Meaning |
|---|---|---|
| VV, VH | Sentinel-1 | Calibrated backscatter (dB) |
| VH_VV_ratio | derived | Cross/co-pol ratio |
| RVI | derived | Radar Vegetation Index |
| entropy_proxy | derived | Dual-pol power-ratio entropy proxy (see caveat in script docstring — NOT full Cloude-Pottier entropy) |
| B2, B3, B4, B8 | Sentinel-2 | Blue, Green, Red, NIR reflectance |
| NDVI, NDWI, EVI | derived | Standard vegetation/water indices |

## Next steps in the pipeline

1. Download the exported GeoTIFFs from Drive.
2. Load into your GMM clustering step using the `entropy_proxy`, `VH_VV_ratio`,
   `RVI` (SAR side) — this replaces/extends your original descriptor set.
3. Use `patch_metadata.csv` to track which patches belong to which region/date,
   so your later per-cluster, per-region comparisons stay traceable.

## Known limitations to state explicitly in your methodology chapter

- `entropy_proxy` is a simplified dual-pol power-ratio entropy, not the
  rigorous multi-look polarimetric entropy from SLC covariance-matrix
  eigendecomposition. If reviewers push back on this, the honest answer is:
  "Sentinel-1 GRD is dual-pol intensity-only; full entropy/alpha decomposition
  requires SLC data and is left as future work" — this is a legitimate,
  common scoping decision, not a flaw to hide.
- The speckle filter is a simple focal-median filter, not the more rigorous
  Refined Lee filter used in SNAP. Fine for a thesis-scale study; mention it
  as a simplification if asked.
