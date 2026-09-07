#!/usr/bin/env python3
"""
03c_nirv_assemble.py
====================
Read 10 yearly NIRv GeoTIFFs exported from GEE, stack into a daily Zarr.
 
Each file: NIRv_{year}.tif — one band per 8-day composite (~46 bands/year).
Band descriptions carry YYYYMMDD date strings (set by GEE .rename()).
 
Fix applied vs. previous version
---------------------------------
Grid dimensions (H, W) and coordinate arrays (lats, lons) are now derived
from the actual TIF affine transform, not from config BBOX arithmetic.
The config calculation can disagree with the exported pixel count because
GEE rounds the region/crsTransform intersection to whole pixels.
Reading from the TIF is always the ground truth.
"""
 
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import zarr
from pathlib import Path
from config import PROC_ROOT, MODIS_DIR
 
NIRV_DIR   = Path(MODIS_DIR) / "/mnt/scratch/olusegu3/drought/conus_drought/data/raw/modis"
ZARR_8D    = Path(PROC_ROOT) / "nirv_8day_conus.zarr"
ZARR_DAILY = Path(PROC_ROOT) / "nirv_daily_conus.zarr"
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Derive grid from the first TIF  (ground truth from export)
# ─────────────────────────────────────────────────────────────────────────────
yearly_tifs = sorted(NIRV_DIR.glob("NIRv_????.tif"))
assert len(yearly_tifs) > 0, (
    "No NIRv_????.tif found in {}\n"
    "Run: rclone copy 'gdrive:drought_forecast_nirv/' data/raw/modis/nirv/".format(NIRV_DIR)
)
 
with rasterio.open(yearly_tifs[0]) as src:
    H          = src.height
    W          = src.width
    tf         = src.transform    # affine: maps pixel (col, row) to (lon, lat)
    actual_crs = src.crs
 
# Pixel-centre coordinates from the affine transform:
#   lon_centre(col) = tf.c + tf.a * (col + 0.5)
#   lat_centre(row) = tf.f + tf.e * (row + 0.5)
# tf.a = +RES (pixel width in degrees, going east)
# tf.e = -RES (pixel height in degrees, going south -> negative)
lons = np.array([tf.c + tf.a * (j + 0.5) for j in range(W)], dtype=np.float32)
lats = np.array([tf.f + tf.e * (i + 0.5) for i in range(H)], dtype=np.float32)
 
print("Grid from TIF  : {} lat x {} lon".format(H, W))
print("CRS            : {}".format(actual_crs))
print("Lon range      : {:.3f} to {:.3f}".format(float(lons[0]),  float(lons[-1])))
print("Lat range      : {:.3f} to {:.3f}".format(float(lats[-1]), float(lats[0])))
print("Pixel spacing  : lon {:.4f} deg  lat {:.4f} deg".format(
    float(lons[1] - lons[0]), float(lats[0] - lats[1])))
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Read all yearly TIFs — parse dates from band descriptions
# ─────────────────────────────────────────────────────────────────────────────
print("\nFound {} yearly files: {} ... {}".format(
    len(yearly_tifs), yearly_tifs[0].name, yearly_tifs[-1].name))
 
all_dates  = []
all_arrays = []
 
for tif_path in yearly_tifs:
    with rasterio.open(tif_path) as src:
        # Sanity-check: all yearly files must share the same grid
        if src.height != H or src.width != W:
            raise ValueError(
                "Grid mismatch in {}: expected {}x{}, got {}x{}".format(
                    tif_path.name, H, W, src.height, src.width)
            )
        descriptions = src.descriptions          # ("20150407", "20150423", ...)
        data = src.read().astype(np.float32)     # (n_bands, H, W)
        nodata = src.nodata
 
    if nodata is not None:
        data[data == nodata] = np.nan
 
    # Physical NIRv range guard (-0.2, 1.1)
    data[data < -0.2] = np.nan
    data[data >  1.1] = np.nan
 
    n_loaded, n_skip = 0, 0
    for band_idx, desc in enumerate(descriptions):
        if not desc:
            print("  WARNING: band {} in {} has no description -- skipping".format(
                band_idx + 1, tif_path.name))
            n_skip += 1
            continue
        try:
            dt = pd.Timestamp(str(desc))
        except Exception as exc:
            print("  WARNING: cannot parse '{}' -- skipping ({})".format(desc, exc))
            n_skip += 1
            continue
        all_dates.append(dt)
        all_arrays.append(data[band_idx])
        n_loaded += 1
 
    valid_frac = float(np.isfinite(data).mean())
    print("  {}: {} bands  valid={:.3f}  skip={}".format(
        tif_path.name, n_loaded, valid_frac, n_skip))
 
# Sort chronologically
order      = np.argsort(all_dates)
all_dates  = [all_dates[i]  for i in order]
all_arrays = [all_arrays[i] for i in order]
 
nirv_arr    = np.stack(all_arrays, axis=0)          # (T, H, W)
time_coords = np.array(all_dates, dtype="datetime64[ns]")
 
print("\n8-day stack shape  : {}".format(nirv_arr.shape))
print("Date range         : {} -- {}".format(all_dates[0].date(), all_dates[-1].date()))
print("Valid fraction     : {:.3f}".format(float(np.isfinite(nirv_arr).mean())))
print("Mean NIRv          : {:.3f}".format(float(np.nanmean(nirv_arr))))
print("  (note: mean includes full BBOX; dryland mask applied in Step 8)")
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Write 8-day Zarr
# ─────────────────────────────────────────────────────────────────────────────
compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
 
ds_8d = xr.Dataset(
    {
        "NIRv": (
            ["time", "lat", "lon"],
            nirv_arr,
            {
                "long_name":  "Near-infrared reflectance of vegetation (NIR x NDVI)",
                "units":      "dimensionless",
                "grid_H":     H,
                "grid_W":     W,
                "source":     "MODIS MOD09A1 v061 via Google Earth Engine",
                "references": (
                    "Badgley et al. 2017 Sci. Adv. 3:e1602244; "
                    "Badgley et al. 2019 GCB 25:3502-3513"
                ),
                "qa_note": (
                    "StateQA bits 0-1 cloud, bit 2 shadow, bits 3-5 land/water. "
                    "Bit 12 (adjacent-to-cloud) relaxed for 8-day composites."
                ),
            },
        )
    },
    coords={"time": time_coords, "lat": lats, "lon": lons},
)
ds_8d = ds_8d.chunk({"time": 1, "lat": 256, "lon": 256})
 
print("\nWriting 8-day Zarr -> {}".format(ZARR_8D))
ds_8d.to_zarr(
    ZARR_8D, mode="w", consolidated=True,
    encoding={"NIRv": {"dtype": "float32", "compressor": compressor}},
)
est_gb = nirv_arr.nbytes / 1e9 / 3.0
print("  Done. Estimated on-disk: ~{:.1f} GB after zstd".format(est_gb))
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Interpolate 8-day to daily
# ─────────────────────────────────────────────────────────────────────────────
print("\nInterpolating 8-day -> daily ...")
ds_in   = xr.open_zarr(ZARR_8D, consolidated=True)
daily_t = pd.date_range(
    str(all_dates[0].date()),
    str(all_dates[-1].date()),
    freq="1D",
)
ds_daily = ds_in.interp(
    time   = daily_t.values,
    method = "linear",
    kwargs = {"fill_value": "extrapolate"},
)
ds_daily = ds_daily.chunk({"time": 10, "lat": 256, "lon": 256})
 
print("Writing daily Zarr -> {}".format(ZARR_DAILY))
ds_daily.to_zarr(
    ZARR_DAILY, mode="w", consolidated=True,
    encoding={"NIRv": {"dtype": "float32", "compressor": compressor}},
)
print("  Done. {} daily time steps".format(len(daily_t)))
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Final QC
# ─────────────────────────────────────────────────────────────────────────────
ds_check  = xr.open_zarr(ZARR_DAILY, consolidated=True)
mid_t     = ds_check.time.values[len(ds_check.time) // 2]
sample    = ds_check["NIRv"].sel(time=mid_t).values
valid_pct = float(np.isfinite(sample).mean()) * 100
 
print("\nQC snapshot ({})".format(str(mid_t)[:10]))
print("  Shape        : {}".format(ds_check["NIRv"].shape))
print("  Valid pixels : {:.1f}%".format(valid_pct))
print("  Mean NIRv    : {:.3f}".format(float(np.nanmean(sample))))
print("  Min / Max    : {:.3f} / {:.3f}".format(
    float(np.nanmin(sample)), float(np.nanmax(sample))))
 
if valid_pct < 60:
    print("  WARNING: Valid % below 60 -- check StateQA masking.")
else:
    print("  OK -- Proceed to Step 5 (grid harmonisation).")
 
# Print coordinate summary for downstream alignment check
print("\nCoordinate summary (for SMAP alignment verification in Step 5):")
print("  lats[0]  = {:.4f}  (top row centre)".format(float(lats[0])))
print("  lats[-1] = {:.4f}  (bottom row centre)".format(float(lats[-1])))
print("  lons[0]  = {:.4f}  (left col centre)".format(float(lons[0])))
print("  lons[-1] = {:.4f}  (right col centre)".format(float(lons[-1])))
