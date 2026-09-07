#!/usr/bin/env python3
"""
05_harmonize_grids.py
=====================
Assemble all preprocessed layers onto the common 321x656 SMAP/NIRv
0.09-degree grid and write a single master Zarr dataset.

Layer inventory and expected sources
--------------------------------------
Layer             Format     Source step   Grid at entry
-----------       ---------  -----------   ---------------
NIRv (daily)      Zarr       Step 3c       321x656  (reference)
SMAP RZSM (daily) NetCDF     Step 2        ~varies  (regrid here)
Irr. fraction     TIF + npy  Step 4b       321x656  (already aligned)
Aridity index     npy        Step 4c       321x656  (already aligned)
CDL crop mask     npy        Step 4a       321x656  (already aligned)

Design principles (consistent with all previous steps)
--------------------------------------------------------
* Reference grid (H, W, lats, lons, transform) always derived from the
  NIRv Zarr, never from config BBOX arithmetic.
* SMAP is the ONLY layer that may need spatial regridding (it comes from
  a separate EASE-2 download pipeline with its own coordinate system).
* All static layers (Steps 4a-4c) are already at 321x656 -- loaded
  directly without interpolation.
* Output: master_dataset.zarr  (Zarr format, chunked time x lat x lon).

SMAP file lookup order
-----------------------
  1. data/processed/smap_daily_conus.nc   (Step 2 NetCDF output)
  2. data/processed/smap_daily_conus.zarr (Zarr alternative)
  If neither exists: script prints instructions and exits cleanly.

Outputs
-------
  data/processed/master_dataset.zarr
  data/processed/qc_master_dataset.png
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import zarr
from pathlib import Path
from config import PROC_ROOT

out_dir = Path(PROC_ROOT)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Reference grid from NIRv Zarr
# ═══════════════════════════════════════════════════════════════════════════
nirv_zarr = out_dir / "nirv_daily_conus.zarr"
assert nirv_zarr.exists(), "NIRv Zarr not found. Run fix_config_and_zarr.sh."

ds_nirv = xr.open_zarr(nirv_zarr, consolidated=True)
ref_lats = ds_nirv.lat.values     # (321,) descending
ref_lons = ds_nirv.lon.values     # (656,) ascending
H, W     = len(ref_lats), len(ref_lons)
T_total  = len(ds_nirv.time)

print("Reference grid  : {} lat x {} lon  (from NIRv Zarr)".format(H, W))
print("NIRv time axis  : {} to {}  ({} steps)".format(
    str(ds_nirv.time.values[0])[:10],
    str(ds_nirv.time.values[-1])[:10], T_total))

# ═══════════════════════════════════════════════════════════════════════════
# 2. SMAP RZSM
# ═══════════════════════════════════════════════════════════════════════════
smap_candidates = [
    out_dir / "smap_daily_2230utc_conus.nc",   # actual filename from Step 2
    out_dir / "smap_daily_conus.nc",            # fallback generic name
    out_dir / "smap_daily_conus.zarr",
]
smap_path = next((p for p in smap_candidates if p.exists()), None)

if smap_path is None:
    print("\n" + "="*60)
    print("SMAP file not found. Expected one of:")
    for p in smap_candidates:
        print("  {}".format(p))
    print("\nSMAP must be processed before Step 5.")
    print("Run Steps 1-2:")
    print("  python 01_download_smap.py   # downloads HDF5 from NASA Earthdata")
    print("  python 02_read_smap.py       # stacks HDF5 -> smap_daily_conus.nc")
    print("="*60)
    raise SystemExit(1)

print("\nSMAP source     : {}".format(smap_path))
if smap_path.suffix == ".zarr":
    ds_smap_raw = xr.open_zarr(smap_path, consolidated=True)
else:
    ds_smap_raw = xr.open_dataset(smap_path)

smap_var = "rzsm" if "rzsm" in ds_smap_raw else list(ds_smap_raw.data_vars)[0]
print("SMAP variable   : {}".format(smap_var))
print("SMAP grid       : {} lat x {} lon".format(
    len(ds_smap_raw.lat), len(ds_smap_raw.lon)))
print("SMAP time axis  : {} to {}  ({} steps)".format(
    str(ds_smap_raw.time.values[0])[:10],
    str(ds_smap_raw.time.values[-1])[:10],
    len(ds_smap_raw.time)))

# ── Align SMAP to common daily time axis ──────────────────────────────────
common_start = max(
    pd.Timestamp(str(ds_nirv.time.values[0])[:10]),
    pd.Timestamp(str(ds_smap_raw.time.values[0])[:10]),
    pd.Timestamp("2015-04-01")
)
common_end = min(
    pd.Timestamp(str(ds_nirv.time.values[-1])[:10]),
    pd.Timestamp(str(ds_smap_raw.time.values[-1])[:10]),
    pd.Timestamp("2024-12-31")
)
common_times = pd.date_range(common_start, common_end, freq="1D")
print("\nCommon period   : {} to {}  ({} days)".format(
    common_start.date(), common_end.date(), len(common_times)))

# Reindex SMAP to daily (nearest-day tolerance=1D handles any time offsets)
smap_daily = (ds_smap_raw[smap_var]
              .reindex(time=common_times, method="nearest",
                       tolerance=pd.Timedelta("1D"))
              .fillna(np.nan))

# ── Regrid SMAP to 321x656 reference grid if needed ───────────────────────
smap_h = len(smap_daily.lat)
smap_w = len(smap_daily.lon)

if smap_h == H and smap_w == W:
    smap_regrid = smap_daily
    print("SMAP grid matches reference -- no spatial regridding needed.")
else:
    print("Regridding SMAP {} x {} -> {} x {} (bilinear) ...".format(
        smap_h, smap_w, H, W))
    smap_regrid = smap_daily.interp(
        lat=ref_lats, lon=ref_lons,
        method="linear",
        kwargs={"fill_value": np.nan}
    )
    print("Regridding complete.")

# Reindex NIRv to same common times
nirv_daily = (ds_nirv["NIRv"]
              .reindex(time=common_times, method="nearest",
                       tolerance=pd.Timedelta("1D"))
              .fillna(np.nan))

print("\nSMAP NaN fraction (sample day 0): {:.3f}".format(
    float(np.isnan(smap_regrid.values[0]).mean())))
print("NIRv NaN fraction (sample day 0): {:.3f}".format(
    float(np.isnan(nirv_daily.values[0]).mean())))

# ═══════════════════════════════════════════════════════════════════════════
# 3. Static layers (already at 321x656 from Steps 4a-4c)
# ═══════════════════════════════════════════════════════════════════════════
def load_npy_check(fname, H, W):
    """Load .npy and verify shape matches reference grid."""
    path = out_dir / fname
    assert path.exists(), "Missing: {}".format(path)
    arr = np.load(path)
    if arr.shape != (H, W):
        raise ValueError("{}: shape {} != reference ({}, {})".format(
            fname, arr.shape, H, W))
    return arr

irr_frac     = load_npy_check("irrigation_fraction_smap_grid.npy", H, W)
aridity_idx  = load_npy_check("aridity_index.npy",   H, W)
aridity_cls  = load_npy_check("aridity_class.npy",   H, W)
dryland_mask = load_npy_check("dryland_mask.npy",    H, W)
crop_mask    = load_npy_check("cdl_crop_mask.npy",   H, W)
crop_type    = load_npy_check("cdl_crop_type.npy",   H, W)

print("\nStatic layers loaded (all {}x{}):".format(H, W))
for name, arr in [("irr_frac", irr_frac), ("aridity_idx", aridity_idx),
                   ("dryland_mask", dryland_mask), ("crop_mask", crop_mask)]:
    print("  {:15s}: range [{:.3f}, {:.3f}]  NaN={}".format(
        name,
        float(np.nanmin(arr)), float(np.nanmax(arr)),
        int(np.isnan(arr).sum())))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Combined dryland cropland mask
#    = dryland (AI < 0.65)  AND  any target crop (CDL label 1-8)
# ═══════════════════════════════════════════════════════════════════════════
water_limited_crop = dryland_mask & crop_mask
print("\nWater-limited cropland pixels: {:,} ({:.1f}% of grid)".format(
    int(water_limited_crop.sum()),
    water_limited_crop.sum() / (H*W) * 100))
print("  (This is the primary model domain for RZSM skill analysis)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Assemble master dataset
# ═══════════════════════════════════════════════════════════════════════════
time_coords = common_times.values.astype("datetime64[ns]")
coord_kw    = {"time": time_coords, "lat": ref_lats, "lon": ref_lons}

ds_master = xr.Dataset(
    {
        # Dynamic layers (time, lat, lon)
        "rzsm": (["time","lat","lon"], smap_regrid.values.astype(np.float32),
                 {"long_name": "Root-zone soil moisture (0-100 cm)",
                  "units": "m3/m3",
                  "source": "SMAP L4 SPL4SMGP v007"}),
        "NIRv": (["time","lat","lon"], nirv_daily.values.astype(np.float32),
                 {"long_name": "Near-infrared reflectance of vegetation",
                  "units": "dimensionless",
                  "source": "MODIS MOD09A1 v061 via GEE"}),
        # Static layers (lat, lon)
        "irr_frac":          (["lat","lon"], irr_frac.astype(np.float32),
                              {"long_name": "Irrigation fraction (LANID 2020)",
                               "units": "fraction 0-1"}),
        "aridity_index":     (["lat","lon"], aridity_idx.astype(np.float32),
                              {"long_name": "Aridity Index P/PET",
                               "source": "CGIAR-CSI v3.1"}),
        "aridity_class":     (["lat","lon"], aridity_cls.astype(np.int8),
                              {"flag_values": [-1,0,1,2,3,4],
                               "flag_meanings": "nodata hyper_arid arid semi_arid "
                                                "dry_subhumid humid"}),
        "dryland_mask":      (["lat","lon"], dryland_mask.astype(np.int8),
                              {"long_name": "Dryland mask (AI 0.05-0.65)"}),
        "crop_mask":         (["lat","lon"], crop_mask.astype(np.int8),
                              {"long_name": "CDL 2020 crop mask (1=target crop)"}),
        "crop_type":         (["lat","lon"], crop_type.astype(np.uint8),
                              {"long_name": "CDL 2020 crop type (1-8=crops 0=non-crop)"}),
        "water_limited_crop":(["lat","lon"], water_limited_crop.astype(np.int8),
                              {"long_name": "Water-limited cropland (primary model domain)",
                               "note": "dryland_mask AND crop_mask"}),
    },
    coords=coord_kw,
    attrs={
        "title":       "CONUS Sub-seasonal Drought Forecast — Master Dataset",
        "description": "SMAP RZSM + MODIS NIRv + static masks on 0.09-deg SMAP grid",
        "conventions": "CF-1.8",
        "time_period": "{} to {}".format(
            str(common_start.date()), str(common_end.date())),
        "grid":        "321 lat x 656 lon  0.09-deg EPSG:4326",
    }
)

# Chunk for efficient time-axis reads (ConvLSTM context windows)
ds_master = ds_master.chunk({"time": 10, "lat": 321, "lon": 656})

# ═══════════════════════════════════════════════════════════════════════════
# 6. Save as Zarr
# ═══════════════════════════════════════════════════════════════════════════
master_zarr = out_dir / "master_dataset.zarr"
compressor  = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

encoding = {}
for var in ["rzsm", "NIRv", "irr_frac", "aridity_index"]:
    encoding[var] = {"dtype": "float32", "compressor": compressor}
for var in ["aridity_class", "dryland_mask", "crop_mask",
            "crop_type", "water_limited_crop"]:
    encoding[var] = {"dtype": "int8", "compressor": compressor}

print("\nWriting master_dataset.zarr ...")
ds_master.to_zarr(master_zarr, mode="w", consolidated=True, encoding=encoding)

size_mb = sum(f.stat().st_size for f in master_zarr.rglob("*") if f.is_file()) / 1e6
print("Saved: {}  ({:.0f} MB)".format(master_zarr, size_mb))

# ═══════════════════════════════════════════════════════════════════════════
# 7. QC
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds_check = xr.open_zarr(master_zarr, consolidated=True)
mid_t    = ds_check.time.values[len(ds_check.time) // 2]
print("\nQC snapshot ({})".format(str(mid_t)[:10]))

for var in ["rzsm", "NIRv"]:
    sample = ds_check[var].sel(time=mid_t).values
    valid  = np.isfinite(sample).mean() * 100
    print("  {:6s}: valid={:.1f}%  mean={:.4f}  range=[{:.3f}, {:.3f}]".format(
        var, valid,
        float(np.nanmean(sample)),
        float(np.nanmin(sample)), float(np.nanmax(sample))))

# Plot mid-period snapshot of each dynamic variable
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

extent = [float(ref_lons[0]), float(ref_lons[-1]),
          float(ref_lats[-1]), float(ref_lats[0])]

rzsm_s = ds_check["rzsm"].sel(time=mid_t).values
nirv_s = ds_check["NIRv"].sel(time=mid_t).values

axes[0,0].imshow(rzsm_s, extent=extent, origin="upper",
                 cmap="Blues_r", vmin=0.05, vmax=0.45, aspect="auto")
axes[0,0].set_title("RZSM ({})".format(str(mid_t)[:10]))

axes[0,1].imshow(nirv_s, extent=extent, origin="upper",
                 cmap="YlGn", vmin=-0.1, vmax=0.4, aspect="auto")
axes[0,1].set_title("NIRv ({})".format(str(mid_t)[:10]))

axes[1,0].imshow(water_limited_crop, extent=extent, origin="upper",
                 cmap="Greens", vmin=0, vmax=1, aspect="auto")
axes[1,0].set_title("Water-limited cropland mask")

import matplotlib.colors as mcolors
_arid_colors = ["#d0d0d0","#a63c06","#e07b27","#f5c842","#8cc97e","#2a7ab8"]
_arid_labels = ["Ocean/NoData","Hyper-arid","Arid","Semi-arid *","Dry sub-humid","Humid"]
_cmap_a = mcolors.ListedColormap(_arid_colors)
_norm_a = mcolors.BoundaryNorm([-1.5,-0.5,0.5,1.5,2.5,3.5,4.5], _cmap_a.N)
im11 = axes[1,1].imshow(aridity_cls, extent=extent, origin="upper",
                        cmap=_cmap_a, norm=_norm_a, aspect="auto")
cbar11 = plt.colorbar(im11, ax=axes[1,1], ticks=[-1,0,1,2,3,4], shrink=0.8)
cbar11.set_ticklabels(_arid_labels, fontsize=7)
axes[1,1].set_title("Aridity class  (* = primary model domain)")

for ax in axes.flat:
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.suptitle("Master dataset QC — {}".format(str(mid_t)[:10]), fontsize=12)
plt.tight_layout()
plt.savefig(out_dir / "qc_master_dataset.png", dpi=150, bbox_inches="tight")
plt.close()
print("QC plot: {}".format(out_dir / "qc_master_dataset.png"))

print("\n── Step 5 complete ──")
print("Master dataset variables:")
for v in ds_check.data_vars:
    print("  {:22s}: {}".format(v, ds_check[v].dims))
print("\nNext: Step 6 (anomaly computation)")
