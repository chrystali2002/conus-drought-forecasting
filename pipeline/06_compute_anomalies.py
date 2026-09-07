#!/usr/bin/env python3
"""
06_compute_anomalies.py
=======================
Compute standardised anomalies for RZSM and NIRv relative to a
±15-day rolling DOY climatology. This removes the annual cycle while
preserving inter-annual drought signals.

Method
------
For each day-of-year (DOY) d and each pixel (i, j):
  1. Collect all time steps t where |DOY(t) - d| <= 15 days
     (circular: year-boundary wrapped).
  2. Compute climatological mean mu(i,j) and std sigma(i,j)
     across those ~310 context samples (31 days x ~10 years).
  3. Anomaly = (observed - mu) / (sigma + epsilon)
     where epsilon = 1e-6 prevents division by zero in bare pixels.

This is a pixel-wise z-score normalisation. Identical to the approach
used in the Africa source study (Adeyeri et al.) for both RZSM and NIRv.

Input
-----
  data/processed/master_dataset.zarr
    rzsm : (3552, 321, 656)  root-zone soil moisture m3/m3
    NIRv : (3552, 321, 656)  near-infrared reflectance of vegetation

Output
------
  data/processed/anomalies.zarr
    rzsm_anom : (3552, 321, 656)  RZSM z-score anomaly
    nirv_anom : (3552, 321, 656)  NIRv z-score anomaly

Memory note
-----------
Each variable is loaded fully as float32:
  3552 x 321 x 656 x 4 bytes = ~3.0 GB per variable.
Peak usage ~6 GB (both in RAM simultaneously during QC).
On nodes with <8 GB RAM, process one variable at a time (set
PROCESS_ONE_AT_A_TIME = True below).
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from config import PROC_ROOT

WINDOW_DAYS = 15          # ±15 days around each DOY (31-day window)
EPSILON     = 1e-6        # prevent /0 in pixels with near-zero variance
MIN_SAMPLES = 10          # minimum context samples to compute anomaly

# ═══════════════════════════════════════════════════════════════════════════
# 1. Open master dataset
# ═══════════════════════════════════════════════════════════════════════════
master_zarr = Path(PROC_ROOT) / "master_dataset.zarr"
assert master_zarr.exists(), "master_dataset.zarr not found. Run Step 5 first."

ds = xr.open_zarr(master_zarr, consolidated=True)
times = pd.DatetimeIndex(ds.time.values)
doys  = np.array(times.day_of_year, dtype=np.int16)   # (T,)

T = len(times)
H = len(ds.lat)
W = len(ds.lon)

print("Master dataset  : {} time x {} lat x {} lon".format(T, H, W))
print("Period          : {} to {}".format(
    str(times[0].date()), str(times[-1].date())))
print("DOY window      : ±{} days ({} days total context per year)".format(
    WINDOW_DAYS, 2*WINDOW_DAYS+1))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Core anomaly function
# ═══════════════════════════════════════════════════════════════════════════
def compute_anomaly(values: np.ndarray,
                    doys:   np.ndarray,
                    window: int = WINDOW_DAYS,
                    eps:    float = EPSILON,
                    min_n:  int   = MIN_SAMPLES,
                    label:  str   = "") -> np.ndarray:
    """
    Compute pixel-wise z-score anomaly relative to a rolling DOY climatology.

    Parameters
    ----------
    values : (T, H, W) float32   -- raw observations
    doys   : (T,) int16          -- day-of-year for each time step
    window : int                 -- half-window in days (±window)
    eps    : float               -- variance floor to prevent /0
    min_n  : int                 -- minimum samples required for climatology

    Returns
    -------
    anom   : (T, H, W) float32   -- standardised anomaly (z-score)
    """
    anom = np.full_like(values, np.nan, dtype=np.float32)

    for doy in tqdm(range(1, 367), desc="{} DOY loop".format(label),
                    ncols=70, leave=True):
        # Circular distance to handle year-wrap (DOY 1 is close to DOY 365)
        diff     = np.abs(doys.astype(np.int32) - doy)
        diff_circ = np.minimum(diff, 366 - diff)
        ctx_mask  = diff_circ <= window    # bool (T,)

        if ctx_mask.sum() < min_n:
            continue

        ctx       = values[ctx_mask]                        # (N, H, W)
        clim_mean = np.nanmean(ctx, axis=0)                 # (H, W)
        clim_std  = np.nanstd(ctx,  axis=0) + eps           # (H, W)

        tgt_mask  = (doys == doy)                           # bool (T,)
        anom[tgt_mask] = (values[tgt_mask] - clim_mean) / clim_std

    return anom


# ═══════════════════════════════════════════════════════════════════════════
# 3. Compute anomalies
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading RZSM ...")
rzsm_raw = ds["rzsm"].values.astype(np.float32)     # (T, H, W) ~3 GB
print("  shape={} NaN={:.1f}%".format(
    rzsm_raw.shape, np.isnan(rzsm_raw).mean()*100))

print("Computing RZSM anomalies ...")
rzsm_anom = compute_anomaly(rzsm_raw, doys, label="RZSM")
del rzsm_raw   # free ~3 GB before loading NIRv

print("\nLoading NIRv ...")
nirv_raw = ds["NIRv"].values.astype(np.float32)
print("  shape={} NaN={:.1f}%".format(
    nirv_raw.shape, np.isnan(nirv_raw).mean()*100))

print("Computing NIRv anomalies ...")
nirv_anom = compute_anomaly(nirv_raw, doys, label="NIRv")
del nirv_raw

# ═══════════════════════════════════════════════════════════════════════════
# 4. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════
wlc_mask = ds["water_limited_crop"].values.astype(bool)   # (H, W)
print("\nAnomaly diagnostics (water-limited cropland pixels only):")
for name, anom in [("RZSM", rzsm_anom), ("NIRv", nirv_anom)]:
    # Mid-period slice over the domain mask
    mid = len(times) // 2
    sample = anom[mid][wlc_mask]
    sample_finite = sample[np.isfinite(sample)]
    print("  {:5s} ({}): valid={:.1f}%  mean={:.3f}  std={:.3f}  "
          "range=[{:.2f},{:.2f}]".format(
        name, str(times[mid].date()),
        len(sample_finite)/len(sample)*100 if len(sample)>0 else 0,
        float(np.nanmean(sample)),
        float(np.nanstd(sample)),
        float(np.nanmin(sample)), float(np.nanmax(sample))))
print("  Expected: mean ~0, std ~1 for standardised anomalies")

# Summer drought example: 2012 mega-drought
drought_date = pd.Timestamp("2012-07-15")
if drought_date in times:
    t_idx = times.get_loc(drought_date)
    drought_rzsm = rzsm_anom[t_idx][wlc_mask]
    drought_nirv = nirv_anom[t_idx][wlc_mask]
    neg_rzsm_pct = float((drought_rzsm < -1.0).mean()) * 100
    neg_nirv_pct = float((drought_nirv < -1.0).mean()) * 100
    print("\n2012 mega-drought signal (2012-07-15):")
    print("  RZSM < -1σ in {:.1f}% of dryland cropland pixels".format(neg_rzsm_pct))
    print("  NIRv < -1σ in {:.1f}% of dryland cropland pixels".format(neg_nirv_pct))
    print("  (Expect significantly elevated negative anomaly fractions)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save anomaly Zarr
# ═══════════════════════════════════════════════════════════════════════════
anom_zarr   = Path(PROC_ROOT) / "anomalies.zarr"
compressor  = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

time_coords = times.values.astype("datetime64[ns]")
coord_kw    = {"time": time_coords,
               "lat":  ds.lat.values,
               "lon":  ds.lon.values}

ds_anom = xr.Dataset(
    {
        "rzsm_anom": (["time","lat","lon"], rzsm_anom,
                      {"long_name": "RZSM standardised anomaly (z-score)",
                       "units": "sigma",
                       "climatology": "±{}-day DOY rolling window".format(WINDOW_DAYS),
                       "source": "SMAP L4 SPL4SMGP v007"}),
        "nirv_anom": (["time","lat","lon"], nirv_anom,
                      {"long_name": "NIRv standardised anomaly (z-score)",
                       "units": "sigma",
                       "climatology": "±{}-day DOY rolling window".format(WINDOW_DAYS),
                       "source": "MODIS MOD09A1 v061"}),
    },
    coords=coord_kw,
    attrs={
        "title":       "CONUS Drought Forecast — RZSM and NIRv Anomalies",
        "window_days": WINDOW_DAYS,
        "epsilon":     EPSILON,
    }
).chunk({"time": 10, "lat": H, "lon": W})

print("\nWriting anomalies.zarr ...")
ds_anom.to_zarr(
    anom_zarr, mode="w", consolidated=True,
    encoding={
        "rzsm_anom": {"dtype": "float32", "compressor": compressor},
        "nirv_anom": {"dtype": "float32", "compressor": compressor},
    }
)
size_mb = sum(f.stat().st_size for f in anom_zarr.rglob("*") if f.is_file()) / 1e6
print("Saved: {}  ({:.0f} MB)".format(anom_zarr, size_mb))

# ═══════════════════════════════════════════════════════════════════════════
# 6. QC plot: anomaly maps for a dry and wet year
# ═══════════════════════════════════════════════════════════════════════════
ds_check = xr.open_zarr(anom_zarr, consolidated=True)
ref_lats  = ds.lat.values
ref_lons  = ds.lon.values
extent    = [float(ref_lons[0]), float(ref_lons[-1]),
             float(ref_lats[-1]), float(ref_lats[0])]

# Pick two diagnostic dates: 2012 drought + 2019 wet year
plot_dates = []
for d in ["2012-07-15", "2019-07-15", "2017-07-15"]:
    ts = pd.Timestamp(d)
    if ts in times:
        plot_dates.append(ts)
plot_dates = plot_dates[:2]   # at most 2 panels per row

fig, axes = plt.subplots(2, len(plot_dates), figsize=(8*len(plot_dates), 10))
if len(plot_dates) == 1:
    axes = axes.reshape(2, 1)

for col, dt in enumerate(plot_dates):
    r_slice = ds_check["rzsm_anom"].sel(time=dt, method="nearest").values
    n_slice = ds_check["nirv_anom"].sel(time=dt, method="nearest").values

    # Mask non-dryland pixels for cleaner visualisation
    r_slice[~wlc_mask] = np.nan
    n_slice[~wlc_mask] = np.nan

    for row, (data, title, cmap) in enumerate([
        (r_slice, "RZSM anomaly", "RdBu"),
        (n_slice, "NIRv anomaly", "RdYlGn"),
    ]):
        ax  = axes[row, col]
        im  = ax.imshow(data, extent=extent, origin="upper",
                        cmap=cmap, vmin=-3, vmax=3, aspect="auto")
        plt.colorbar(im, ax=ax, label="z-score (σ)", shrink=0.8)
        ax.set_title("{}\n{}".format(title, dt.date()), fontsize=10)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

plt.suptitle("Anomaly QC — water-limited cropland pixels only", fontsize=12)
plt.tight_layout()
qc_path = Path(PROC_ROOT) / "qc_anomalies.png"
plt.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.close()
print("QC plot: {}".format(qc_path))

print("\n── Step 6 complete ──")
print("Outputs:")
print("  anomalies.zarr       rzsm_anom, nirv_anom  (T={}, H={}, W={})".format(T,H,W))
print("  qc_anomalies.png")
print("\nExpected QC results:")
print("  mean anomaly ~0.0 (zero-centred by construction)")
print("  std  anomaly ~1.0 (unit variance by construction)")
print("  2012-07-15: strongly negative RZSM and NIRv anomalies")
print("  2019-07-15: mixed / positive anomalies (wetter year)")
print("\nNext: Step 7 (irrigation deconvolution)")
