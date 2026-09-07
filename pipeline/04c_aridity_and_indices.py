#!/usr/bin/env python3
"""
04c_aridity_and_indices.py
==========================
A. Reproject CGIAR-CSI Aridity Index to the 321x656 SMAP/NIRv grid.
B. Download daily ENSO (Nino3.4), PDO, and AMO teleconnection indices.

Bug fixed vs previous version
-------------------------------
  src_nodata fallback of -9999.0 caused:
    ValueError: src_nodata must be in valid range for source dtype
  because CGIAR AI v3.1 is stored as uint16 (AI x10000, range 0-65535)
  and -9999 is outside uint16 range.
  Fix: pass src_nodata=None when src.nodata is None. Ocean/background
  pixels (AI==0 after scaling) are removed by the ai_smap<=0.001 mask.

CGIAR file details
------------------
  File    : data/raw/aridity/ai_v31_yr.tif
  CRS     : EPSG:4326 (already geographic -- no CRS transform needed)
  Shape   : 18000 x 43200 (30 arc-sec global)
  Dtype   : uint16  (AI stored as int(AI * 10000))
  Nodata  : None    (no nodata tag; ocean pixels have value 0)

Aridity classification (UNEP/IPCC)
------------------------------------
  Hyper-arid  : AI < 0.05
  Arid        : 0.05-0.20
  Semi-arid   : 0.20-0.50  <- primary RZSM-skill domain
  Dry sub-hum : 0.50-0.65
  Humid       : >= 0.65    <- excluded

Teleconnection sources
-----------------------
  Nino3.4 : NOAA CPC sstoi.indices
  AMO     : NOAA PSL amon.us.long.data
  PDO     : NCEI ERSST v5 pdo.dat
  AMO 2024 NaN filled by ffill().bfill() (data ends 2023-12).

Outputs (all at 321x656 SMAP grid)
------------------------------------
  data/processed/aridity_index.npy
  data/processed/aridity_class.npy
  data/processed/dryland_mask.npy
  data/processed/semi_arid_mask.npy
  data/processed/aridity_domain_masks.nc
  data/processed/teleconnection_indices_daily.csv
  data/processed/qc_aridity.png
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from config import PROC_ROOT, DRYLAND_AI_MIN, DRYLAND_AI_MAX

# ═══════════════════════════════════════════════════════════════════════════
# A. ARIDITY INDEX
# ═══════════════════════════════════════════════════════════════════════════

# ── A1. Target grid from NIRv Zarr ─────────────────────────────────────────
zarr_path = Path(PROC_ROOT) / "nirv_daily_conus.zarr"
assert zarr_path.exists(), "NIRv Zarr not found. Run fix_config_and_zarr.sh."

ds_nirv = xr.open_zarr(zarr_path, consolidated=True)
lats = ds_nirv.lat.values.astype(np.float64)
lons = ds_nirv.lon.values.astype(np.float64)
H, W = len(lats), len(lons)

RES     = round(float(lons[1] - lons[0]), 4)
LON_MIN = round(float(lons[0])  - RES / 2, 2)
LAT_MAX = round(float(lats[0])  + RES / 2, 2)
LON_MAX = round(float(lons[-1]) + RES / 2, 2)
LAT_MIN = round(float(lats[-1]) - RES / 2, 2)
dst_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)

print("Target grid : {} lat x {} lon".format(H, W))
print("BBOX        : lon [{}, {}]  lat [{}, {}]".format(
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX))

# ── A2. Open CGIAR raster and check dtype ──────────────────────────────────
ai_tif = Path("data/raw/aridity/ai_v31_yr.tif")
assert ai_tif.exists(), "CGIAR AI raster not found: {}".format(ai_tif)

with rasterio.open(ai_tif) as src:
    print("CGIAR CRS   : {}".format(src.crs))
    print("CGIAR shape : {} x {}".format(src.height, src.width))
    print("CGIAR dtype : {}".format(src.dtypes[0]))
    print("CGIAR nodata: {}  (None = no nodata tag; ocean pixels = 0)".format(
        src.nodata))

    # FIX: pass src_nodata=None when the file has no nodata tag.
    # src_nodata=-9999 would raise ValueError for uint16 (range 0-65535).
    # Ocean/background pixels have AI=0 after /10000 scaling and are
    # removed by the ai_smap<=0.001 mask applied after reprojection.
    src_nodata_val = src.nodata   # None for CGIAR AI v3.1

    print("\nReprojecting CGIAR AI -> {}-deg SMAP grid ...".format(RES))
    ai_raw = np.zeros((H, W), dtype=np.float32)
    reproject(
        source        = rasterio.band(src, 1),
        destination   = ai_raw,
        src_transform = src.transform,
        src_crs       = src.crs,
        dst_transform = dst_transform,
        dst_crs       = "EPSG:4326",
        resampling    = Resampling.average,
        src_nodata    = src_nodata_val,  # None: all pixels contribute
        dst_nodata    = 0.0,
    )

# ── A3. Scale and mask ─────────────────────────────────────────────────────
# CGIAR stores AI * 10000 as uint16; rasterio read/reproject on raw values
ai_smap = ai_raw / 10000.0

# Remove ocean/background: CGIAR ocean pixels have raw value 0 -> AI=0.
# Threshold 0.001 (well below hyper-arid 0.05) removes ocean without
# affecting any valid dryland pixels.
ai_smap[ai_smap <= 0.001] = np.nan

print("AI range (0.09-deg): {:.4f} -- {:.4f}".format(
    float(np.nanmin(ai_smap)), float(np.nanmax(ai_smap))))
valid_px = int(np.isfinite(ai_smap).sum())
print("Valid land pixels  : {:,} / {:,}  ({:.1f}%)".format(
    valid_px, H*W, valid_px / (H*W) * 100))

# ── A4. Classify into UNEP/IPCC zones ─────────────────────────────────────
CLASS_MAP = {
    "hyper_arid":   (0,  0.00, 0.05),
    "arid":         (1,  0.05, 0.20),
    "semi_arid":    (2,  0.20, 0.50),
    "dry_subhumid": (3,  0.50, 0.65),
    "humid":        (4,  0.65, 9.99),
}
aridity_class = np.full((H, W), -1, dtype=np.int8)
pixel_counts  = {}
for cname, (label, lo, hi) in CLASS_MAP.items():
    m = (ai_smap >= lo) & (ai_smap < hi)
    aridity_class[m] = label
    pixel_counts[cname] = int(m.sum())

dryland_mask   = (ai_smap >= DRYLAND_AI_MIN) & (ai_smap < DRYLAND_AI_MAX)
semi_arid_mask = (ai_smap >= 0.20)           & (ai_smap < 0.50)

print("\nUNEP/IPCC pixel distribution on 0.09-deg SMAP grid:")
for cname, count in pixel_counts.items():
    _, lo, hi = CLASS_MAP[cname]
    pct    = count / valid_px * 100
    marker = "  <- primary model domain" if cname == "semi_arid" else ""
    print("  {:15s} AI {:.2f}-{:.2f}: {:6,d} px ({:5.1f}%){}".format(
        cname, lo, hi, count, pct, marker))
print("  Dryland (AI {}-{}): {:,} px ({:.1f}%)".format(
    DRYLAND_AI_MIN, DRYLAND_AI_MAX,
    int(dryland_mask.sum()), dryland_mask.sum() / valid_px * 100))

# ── A5. Save ───────────────────────────────────────────────────────────────
out_dir = Path(PROC_ROOT)
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "aridity_index.npy",  ai_smap.astype(np.float32))
np.save(out_dir / "aridity_class.npy",  aridity_class)
np.save(out_dir / "dryland_mask.npy",   dryland_mask)
np.save(out_dir / "semi_arid_mask.npy", semi_arid_mask)

lc = lats.astype(np.float32)
lo = lons.astype(np.float32)
ds_ai = xr.Dataset({
    "ai":        (["lat","lon"], ai_smap,
                  {"long_name": "Aridity Index (P/PET)", "units": "dimensionless",
                   "source": "Trabucco & Zomer 2019 CGIAR-CSI AI v3.1",
                   "grid": "Resampled to SMAP 0.09-deg via Resampling.average"}),
    "ai_class":  (["lat","lon"], aridity_class.astype(np.int8),
                  {"flag_values": [-1,0,1,2,3,4],
                   "flag_meanings": "nodata hyper_arid arid semi_arid dry_subhumid humid"}),
    "dryland":   (["lat","lon"], dryland_mask.astype(np.int8)),
    "semi_arid": (["lat","lon"], semi_arid_mask.astype(np.int8)),
}, coords={"lat": lc, "lon": lo})
ds_ai.to_netcdf(
    out_dir / "aridity_domain_masks.nc",
    encoding={v: {"zlib": True, "complevel": 5} for v in ds_ai.data_vars}
)
print("\nAridity outputs saved (shape {}x{}).".format(H, W))

# ── A6. QC plot ────────────────────────────────────────────────────────────
# Aridity zones:
# 0 = arid (<0.20)
# 1 = semi-arid [0.20, 0.50)
# 2 = dry sub-humid [0.50, 0.65)
# 3 = humid (>=0.65)
# -1 = nodata

CLASS_COLORS = {-1:"#d0d0d0", 0:"#a63c06", 1:"#e07b27",
                 2:"#f5c842", 3:"#8cc97e", 4:"#2a7ab8"}
CLASS_LABELS = {-1:"Ocean/NoData", 0:"Hyper-arid (<0.05)",
                 1:"Arid (0.05-0.20)", 2:"Semi-arid (0.20-0.50) *",
                 3:"Dry sub-humid (0.50-0.65)", 4:"Humid (>=0.65)"}
cmap_d = mcolors.ListedColormap([CLASS_COLORS[k] for k in sorted(CLASS_COLORS)])
norm_d = mcolors.BoundaryNorm([-1.5,-0.5,0.5,1.5,2.5,3.5,4.5], cmap_d.N)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(aridity_class,
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
               origin="upper", cmap=cmap_d, norm=norm_d, aspect="auto")
cbar = plt.colorbar(im, ax=ax, ticks=[-1,0,1,2,3,4], shrink=0.7)
cbar.set_ticklabels([CLASS_LABELS[k] for k in sorted(CLASS_LABELS)], fontsize=8)
ax.set_title(
    "UNEP/IPCC Aridity Classes (0.09-deg SMAP grid)  |  * = primary model domain\n"
    "Dryland (AI {}-{}): {:,} px ({:.1f}% of valid land)".format(
        DRYLAND_AI_MIN, DRYLAND_AI_MAX,
        int(dryland_mask.sum()), dryland_mask.sum() / valid_px * 100),
    fontsize=10)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(out_dir / "qc_aridity.png", dpi=150, bbox_inches="tight")
plt.close()
print("QC plot saved.")

# ═══════════════════════════════════════════════════════════════════════════
# B. TELECONNECTION INDICES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_text(url):
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", errors="ignore")


def load_nino34_cpc(url, col_name="Nino34"):
    """NOAA CPC sstoi.indices: column 9 = Nino3.4 anomaly."""
    records = []
    for line in fetch_text(url).splitlines():
        parts = line.split()
        if len(parts) < 10 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        records.append({
            "date":   pd.Timestamp(year=int(parts[0]), month=int(parts[1]), day=1),
            col_name: float(parts[9])
        })
    return (pd.DataFrame(records)
              .drop_duplicates(subset="date", keep="last")
              .set_index("date")[col_name].sort_index())


def load_amo_psl(url, col_name="AMO"):
    """NOAA PSL AMO: 13-column rows (YEAR + 12 months); -99 = NaN."""
    records = []
    for line in fetch_text(url).splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for month, val in enumerate(parts[1:13], 1):
            x = float(val)
            records.append({
                "date":   pd.Timestamp(year=year, month=month, day=1),
                col_name: np.nan if x <= -99 else x
            })
    return (pd.DataFrame(records)
              .drop_duplicates(subset="date", keep="last")
              .set_index("date")[col_name].sort_index())


def load_pdo(url, col_name="PDO"):
    """NCEI ERSST v5 PDO: 13-column rows (YEAR + 12 months).
    Column-count filter avoids 'Oct' ValueError on header lines."""
    records = []
    for line in fetch_text(url).splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for month, val in enumerate(parts[1:13], 1):
            records.append({
                "date":   pd.Timestamp(year=year, month=month, day=1),
                col_name: float(val)
            })
    return (pd.DataFrame(records)
              .drop_duplicates(subset="date", keep="last")
              .set_index("date")[col_name].sort_index())


print("\n── Downloading teleconnection indices ──")
nino34 = load_nino34_cpc(
    "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices")
amo    = load_amo_psl(
    "https://psl.noaa.gov/data/correlation/amon.us.long.data")
pdo    = load_pdo(
    "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat")

for name, s in [("Nino34", nino34), ("AMO", amo), ("PDO", pdo)]:
    print("  {:7s}: {:,} months  {:%Y-%m} to {:%Y-%m}  NaN={}".format(
        name, len(s), s.index.min(), s.index.max(), int(s.isna().sum())))

# ── Resample monthly -> daily ──────────────────────────────────────────────
# ffill: carry monthly value forward through each day of the month
# bfill: fill any remaining NaN (e.g. AMO 2024 after data ends 2023-12)
start     = pd.Timestamp("2015-01-01")
end       = pd.Timestamp("2024-12-31")
daily_idx = pd.date_range(start, end, freq="1D")

telecon = pd.DataFrame(index=daily_idx)
for name, series in [("Nino34", nino34), ("AMO", amo), ("PDO", pdo)]:
    s_clean          = series[~series.index.duplicated(keep="last")].sort_index()
    telecon[name]    = s_clean.reindex(daily_idx, method="ffill").ffill().bfill()

nan_counts = telecon.isna().sum()
if nan_counts.any():
    print("\nWARNING: residual NaN after ffill+bfill:")
    print(nan_counts[nan_counts > 0])
else:
    print("  All columns: 0 NaN after ffill+bfill  OK")

out_csv = out_dir / "teleconnection_indices_daily.csv"
telecon.to_csv(out_csv)
print("\nSaved: {}".format(out_csv))
print(telecon.head(3).to_string())
print("...\n" + telecon.tail(3).to_string())

# ── Final summary ──────────────────────────────────────────────────────────
print("\n── Step 4c complete ──")
for f in ["aridity_index.npy", "aridity_class.npy", "dryland_mask.npy",
          "semi_arid_mask.npy", "aridity_domain_masks.nc",
          "teleconnection_indices_daily.csv", "qc_aridity.png"]:
    p = out_dir / f
    print("  {} {}".format("OK" if p.exists() else "MISSING", f))
print("\nNext: Step 5 (grid harmonisation)")









