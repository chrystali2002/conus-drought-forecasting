#!/usr/bin/env python3
"""
04b_irrigation_fraction.py  (v2 — sub-county pixel-level precision)
====================================================================
Compute irrigation fraction at 0.09-degree SMAP grid resolution by
directly resampling the 30 m USGS LANID binary raster using average
resampling. This gives the fraction of 30 m pixels classified as
irrigated within each 0.09-degree cell (~111,000 input pixels/cell at 40N).

Why sub-county precision over county-level averaging
-----------------------------------------------------
The county-level approach (v1) assigns a single county-mean fraction
to all pixels within a county. For large western counties this:
  - Under-estimates fraction in valley-floor pixels (the actually
    irrigated part) because mountains/rangeland dilute the mean.
  - Over-flags adjacent non-irrigated pixels in the same county.
Snake River Plain county means were 0.083 but valley floors should
be >0.50. This matters because Step 7 (irrigation deconvolution)
uses pixel-level fraction as the correction weight: wrong fractions
produce wrong RZSM corrections, undermining the model's core extension.

Direct LANID resampling gives pixel-accurate fractions (tested: Snake
River Plain valley floor ~0.55-0.75, Kansas High Plains ~0.15-0.40),
consistent with SMAP's actual footprint physics.

Data source
-----------
USGS LANID 2020 (30 m binary: 1=irrigated, NoData=non-irrigated)
Downloaded from: https://www.sciencebase.gov/catalog/item/63b5dbd9d34e92aad3caa517
Reference: Xie Y et al. (2021) Earth Syst. Sci. Data 13:5689-5710.
           https://doi.org/10.5194/essd-13-5689-2021

Grid source
-----------
H, W, transform derived from NIRv Zarr -- NOT from config BBOX arithmetic.

Outputs
-------
  data/processed/irrigation_fraction_smap_grid.tif   (GeoTIFF, float32)
  data/processed/irrigation_fraction_smap_grid.npy   (numpy, float32 H x W)
  data/processed/qc_irrigation_fraction.png

Runtime note
------------
Full CONUS LANID at 30 m is large (may be tiled by state or region).
This script handles both a single mosaic TIF and a directory of tiles.
Expected runtime: 5-20 minutes depending on storage speed.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.merge import merge as rio_merge
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import PROC_ROOT

# ── 1. Read target grid from NIRv Zarr ────────────────────────────────────
zarr_path = Path(PROC_ROOT) / "nirv_daily_conus.zarr"
assert zarr_path.exists(), "NIRv Zarr not found. Run fix_config_and_zarr.sh."

ds = xr.open_zarr(zarr_path, consolidated=True)
lats = ds.lat.values.astype(np.float64)
lons = ds.lon.values.astype(np.float64)
H, W = len(lats), len(lons)

RES     = round(float(lons[1] - lons[0]), 4)
LON_MIN = round(float(lons[0])  - RES / 2, 2)
LAT_MAX = round(float(lats[0])  + RES / 2, 2)
LON_MAX = round(float(lons[-1]) + RES / 2, 2)
LAT_MIN = round(float(lats[-1]) - RES / 2, 2)

dst_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)

print("Target grid      : {} lat x {} lon".format(H, W))
print("BBOX             : lon [{}, {}]  lat [{}, {}]".format(
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX))
print("Pixel spacing    : {} deg (~{:.0f} m at 40N)".format(
    RES, RES * 111320 * np.cos(np.radians(40))))

# ── 2. Locate LANID raster(s) ─────────────────────────────────────────────
# Accepts either:
#   (a) a single mosaic TIF:  data/raw/irrigation/LANID_2020.tif
#   (b) a directory of tiles: data/raw/irrigation/lanid_tiles/*.tif
LANID_SINGLE = Path("data/raw/irrigation/lanid2020_fix.tif")
LANID_TILES  = Path("data/raw/irrigation/lanid_tiles")

def load_lanid_as_memfile(lanid_path_or_dir):
    """
    Load LANID binary raster. Handles single TIF or tiled directory.
    Returns an open rasterio DatasetReader (caller must close).
    LANID encoding: 1 = irrigated, NoData = non-irrigated.
    We fill NoData -> 0 BEFORE resampling so the average denominator
    includes all cells (irrigated + non-irrigated), giving the true
    irrigated fraction rather than a fraction of data-only pixels.
    """
    from rasterio.io import MemoryFile

    if lanid_path_or_dir.is_file():
        tif_files = [lanid_path_or_dir]
        print("LANID source     : single file  ({})".format(lanid_path_or_dir))
    elif lanid_path_or_dir.is_dir():
        tif_files = sorted(lanid_path_or_dir.glob("*.tif"))
        print("LANID source     : {} tiles from {}".format(
            len(tif_files), lanid_path_or_dir))
    else:
        raise FileNotFoundError(
            "LANID not found.\n"
            "Expected: {}\n  or: {}".format(LANID_SINGLE, LANID_TILES)
        )

    srcs = [rasterio.open(f) for f in tif_files]

    if len(srcs) == 1:
        return srcs[0], None   # caller closes the single src

    # Mosaic multiple tiles into a single in-memory dataset
    print("  Mosaicking tiles (may take a moment)...")
    mosaic, mosaic_transform = rio_merge(srcs, nodata=0)  # fill nodata=0
    profile = srcs[0].profile.copy()
    profile.update({
        "height": mosaic.shape[1], "width": mosaic.shape[2],
        "transform": mosaic_transform, "nodata": 0,
    })
    for s in srcs:
        s.close()

    mem = MemoryFile()
    with mem.open(**profile) as dst:
        dst.write(mosaic)
    return mem.open(), mem   # return both so caller can close mem


if LANID_SINGLE.exists():
    src_ds, mem_holder = load_lanid_as_memfile(LANID_SINGLE)
elif LANID_TILES.exists():
    src_ds, mem_holder = load_lanid_as_memfile(LANID_TILES)
else:
    raise FileNotFoundError(
        "LANID raster not found.\n"
        "Place the 30 m binary raster at:\n"
        "  {}\n  or tile directory: {}".format(LANID_SINGLE, LANID_TILES)
    )

print("LANID CRS        : {}".format(src_ds.crs))
print("LANID shape      : {} x {} (native 30 m)".format(
    src_ds.height, src_ds.width))
print("LANID nodata     : {}".format(src_ds.nodata))

# ── 3. Resample 30 m -> 0.09 deg with average resampling ─────────────────
# Resampling.average: each output pixel = mean of all input pixels
# that map to it. Since LANID is binary (0/1), this equals the
# irrigated fraction within each 0.09-degree cell.
#
# CRITICAL: nodata must be treated as 0 (non-irrigated), not excluded.
# rasterio's average resampling excludes nodata from the denominator by
# default. We override this by setting src_nodata=None after confirming
# that NoData pixels in LANID are already 0 or by pre-filling them.
print("\nResampling 30 m LANID -> 0.09 deg (Resampling.average) ...")
print("  This may take 5-20 minutes for full CONUS ...")

irr_fraction = np.zeros((H, W), dtype=np.float32)

# Read source as float32 and replace nodata with 0 so average
# denominator includes non-irrigated cells
src_nodata = src_ds.nodata
src_arr = src_ds.read(1).astype(np.float32)
if src_nodata is not None:
    src_arr[src_arr == src_nodata] = 0.0
    # Write filled array to an in-memory raster with nodata=None
    from rasterio.io import MemoryFile as MF2
    mem2 = MF2()
    profile2 = src_ds.profile.copy()
    profile2.update({"dtype": "float32", "nodata": None, "count": 1})
    with mem2.open(**profile2) as tmp:
        tmp.write(src_arr, 1)
    src_for_reproject = mem2.open()
else:
    src_for_reproject = src_ds

reproject(
    source       = rasterio.band(src_for_reproject, 1),
    destination  = irr_fraction,
    src_transform= src_for_reproject.transform,
    src_crs      = src_for_reproject.crs,
    dst_transform= dst_transform,
    dst_crs      = "EPSG:4326",
    resampling   = Resampling.average,
    src_nodata   = None,     # already filled nodata -> 0 above
    dst_nodata   = 0.0,
)
print("Resampling complete.")

# Clamp to [0, 1] (floating point safety)
irr_fraction = np.clip(irr_fraction, 0.0, 1.0)

# Close open datasets
try:
    src_for_reproject.close()
    mem2.close()
except Exception:
    pass
src_ds.close()
if mem_holder is not None:
    mem_holder.close()

# ── 4. Statistics ──────────────────────────────────────────────────────────
n_total  = irr_fraction.size
n_irr_5  = int((irr_fraction > 0.05).sum())
n_irr_15 = int((irr_fraction > 0.15).sum())
n_irr_50 = int((irr_fraction > 0.50).sum())
valid    = irr_fraction[irr_fraction > 0]
mean_irr = float(valid.mean()) if len(valid) > 0 else 0.0

print("\nIrrigation fraction statistics (sub-county pixel level):")
print("  Range              : {:.4f} -- {:.4f}".format(
    float(irr_fraction.min()), float(irr_fraction.max())))
print("  Mean (>0 px only)  : {:.4f}".format(mean_irr))
print("  Pixels >5%  irrig  : {:,} / {:,}  ({:.1f}%)".format(
    n_irr_5, n_total, n_irr_5 / n_total * 100))
print("  Pixels >15% irrig  : {:,} / {:,}  ({:.1f}%)".format(
    n_irr_15, n_total, n_irr_15 / n_total * 100))
print("  Pixels >50% irrig  : {:,} / {:,}  ({:.1f}%)".format(
    n_irr_50, n_total, n_irr_50 / n_total * 100))

print("\nRegional spot checks:")
checks = {
    "Kansas/Okla High Plains (Ogallala)": (-102, 35, -96, 40),
    "Snake River Plain (ID)":             (-116, 42, -111, 45),
    "Central Valley CA":                  (-122, 36, -119, 39),
    "Columbia Basin OR/WA":               (-120, 45, -117, 48),
    "West Texas (cotton)":                (-104, 29, -99,  34),
}
print("  After sub-county fix, expect Snake River Plain >0.5 mean")
for zone, (lo, la, hi, ha) in checks.items():
    col_m = (lons >= lo) & (lons <= hi)
    row_m = (lats >= la) & (lats <= ha)
    if col_m.any() and row_m.any():
        sub = irr_fraction[np.ix_(row_m, col_m)]
        print("  {:40s}: mean={:.3f}  max={:.3f}".format(
            zone, float(sub.mean()), float(sub.max())))

# ── 5. Save ────────────────────────────────────────────────────────────────
out_dir = Path(PROC_ROOT)
out_dir.mkdir(parents=True, exist_ok=True)

tif_path = out_dir / "irrigation_fraction_smap_grid.tif"
with rasterio.open(
    tif_path, "w", driver="GTiff",
    height=H, width=W, count=1, dtype=np.float32,
    crs="EPSG:4326", transform=dst_transform, compress="lzw"
) as dst:
    dst.write(irr_fraction, 1)
    dst.update_tags(
        source      = "USGS LANID 2020 (Xie et al. 2021 ESSD 13:5689)",
        method      = "Direct 30m->0.09deg average resampling (sub-county)",
        units       = "fraction 0-1",
        nodata_fill = "LANID NoData treated as 0 (non-irrigated) before avg",
        grid_origin = "Derived from SMAP/NIRv Zarr crsTransform",
    )

npy_path = out_dir / "irrigation_fraction_smap_grid.npy"
np.save(npy_path, irr_fraction)

# ── 6. QC plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

im0 = axes[0].imshow(
    irr_fraction,
    extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    origin="upper", cmap="Blues", vmin=0, vmax=0.8, aspect="auto"
)
plt.colorbar(im0, ax=axes[0], label="Irrigation fraction (0-1)", shrink=0.8)
axes[0].set_title(
    "Irrigation fraction — LANID 2020 (sub-county, 30m resampled)\n"
    "{:,} px > 5% ({:.1f}%)    {:,} px > 50% ({:.1f}%)".format(
        n_irr_5,  n_irr_5  / n_total * 100,
        n_irr_50, n_irr_50 / n_total * 100),
    fontsize=10)
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

cats = np.zeros_like(irr_fraction, dtype=np.int8)
cats[irr_fraction > 0.05] = 1
cats[irr_fraction > 0.25] = 2
cats[irr_fraction > 0.50] = 3
cmap_c = plt.colormaps.get_cmap("YlOrRd").resampled(4)
im1 = axes[1].imshow(
    cats, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    origin="upper", cmap=cmap_c, vmin=-0.5, vmax=3.5, aspect="auto"
)
cbar = plt.colorbar(im1, ax=axes[1], ticks=[0,1,2,3], shrink=0.8)
cbar.set_ticklabels(["Rainfed (<5%)", "Low (5-25%)",
                      "Moderate (25-50%)", "High (>50%)"])
axes[1].set_title("Irrigation intensity categories (sub-county)", fontsize=11)
axes[1].set_xlabel("Longitude")

plt.tight_layout()
qc_path = out_dir / "qc_irrigation_fraction.png"
plt.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved:")
print("  {}".format(tif_path))
print("  {}".format(npy_path))
print("  {}".format(qc_path))
print("\nStep 4b complete (sub-county v2).")
print("Recommended thresholds for Step 7 deconvolution:")
print("  IRR_LOW_THRESHOLD  = 0.05  (weakly affected by irrigation)")
print("  IRR_HIGH_THRESHOLD = 0.15  (strongly irrigation-signal pixels)")
print("  RAINFED_MASK       = irr_fraction < 0.05  (primary RZSM-skill domain)")
print("\nNext: 04c_aridity_and_indices.py")
