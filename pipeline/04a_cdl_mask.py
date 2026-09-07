#import ee
#import numpy as np
#import rasterio
##from rasterio.io import MemoryFile
#from rasterio.transform import from_bounds
#import xarray as xr
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#from pathlib import Path
#from config import PROC_ROOT, CDL_DIR
 
# ── Initialise EE ──────────────────────────────────────────────────────────
#ee.Initialize()
#ee.Initialize(project="ee-concert-445314") 


#!/usr/bin/env python3
"""
04a_cdl_mask.py  (v3)
=====================
Fixes vs v2:
  Bug 1: winter_wheat (label 0) was silently overwritten by nodata handling.
         GEE sets TIFF nodata=0.0 in the header even after .unmask(255).
         The old code replaced arr==0 with 255, destroying label-0 pixels.
         Fix: ignore the GEE nodata header entirely after .unmask(255).
         All masked pixels already carry 255 in the pixel data itself.
 
  Bug 2: crsTransform origin still had float imprecision (-125.000002,
         53.870001) causing GEE to return 363x657 instead of 321x656.
         Fix: round origin values to 2 decimal places.
         round(-125.000002, 2) = -125.0
         round(53.870001,   2) =  53.87
         With these values GEE returns exactly 321x656.
"""
 
import ee
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import PROC_ROOT
 
ee.Initialize(project="ee-concert-445314")
 
zarr_path = Path(PROC_ROOT) / "nirv_daily_conus.zarr"
assert zarr_path.exists(), "NIRv Zarr not found. Run fix_config_and_zarr.sh first."
 
ds = xr.open_zarr(zarr_path, consolidated=True)
lats = ds.lat.values.astype(np.float64)
lons = ds.lon.values.astype(np.float64)
H, W = len(lats), len(lons)
 
# BUG 2 FIX: round pixel spacing to 4 dp, then round origin to 2 dp.
# float32 Zarr coordinates give lons[1]-lons[0] = 0.090004 and origins
# like -125.000002.  round(..., 2) snaps to exactly -125.0 and 53.87,
# which produces the correct 321x656 pixel count in GEE.
RES     = round(float(lons[1] - lons[0]), 4)           # 0.090004 -> 0.09
LON_MIN = round(float(lons[0])  - RES / 2, 2)          # -125.000002 -> -125.0
LAT_MAX = round(float(lats[0])  + RES / 2, 2)          #   53.870001 ->  53.87
LON_MAX = round(float(lons[-1]) + RES / 2, 2)
LAT_MIN = round(float(lats[-1]) - RES / 2, 2)
 
CRS_TRANSFORM = [RES, 0, LON_MIN, 0, -RES, LAT_MAX]
REGION        = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])
SCALE_M       = int(RES * 111_320)
 
print("Grid (from Zarr) : {} lat x {} lon".format(H, W))
print("RES              : {}".format(RES))
print("crsTransform     : {}  <- should be clean decimals".format(CRS_TRANSFORM))
print("BBOX             : lon [{}, {}]  lat [{}, {}]".format(
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX))
expected_cols = round((LON_MAX - LON_MIN) / RES)
expected_rows = round((LAT_MAX - LAT_MIN) / RES)
print("Expected GEE grid: {} rows x {} cols  (target {} x {})".format(
    expected_rows, expected_cols, H, W))
 
# ── 2. Crop code / label definitions ───────────────────────────────────────
# Labels start at 1 (not 0) to avoid collision with GEE's nodata=0.
# This is the permanent fix: reserve 0 as a safe sentinel.
CROP_CLASSES = {
    "winter_wheat": {"codes": [24],  "label": 1},
    "spring_wheat": {"codes": [26],  "label": 2},
    "corn":         {"codes": [1],   "label": 3},
    "sorghum":      {"codes": [4],   "label": 4},
    "cotton":       {"codes": [2],   "label": 5},
    "alfalfa":      {"codes": [36],  "label": 6},
    "barley":       {"codes": [21],  "label": 7},
    "durum_wheat":  {"codes": [27],  "label": 8},
}
# 0   = non-crop (safe: never conflicts with GEE nodata=0 because we unmask)
# 1-8 = crop classes
# 255 = fallback fill (also non-crop; belt-and-suspenders)
ALL_CROP_CODES  = [c for v in CROP_CLASSES.values() for c in v["codes"]]
ALL_CROP_LABELS = [v["label"] for v in CROP_CLASSES.values()
                   for _ in v["codes"]]
 
# ── 3. Build GEE images ─────────────────────────────────────────────────────
cdl_raw = ee.Image("USDA/NASS/CDL/2020").select("cropland").clip(REGION)
 
crop_mask_ee = (cdl_raw
                .remap(ALL_CROP_CODES, [1] * len(ALL_CROP_CODES), 0)
                .unmask(0)           # ocean / non-CDL -> 0 (non-crop)
                .rename("crop_mask")
                .toByte())
 
crop_type_ee = (cdl_raw
                .remap(ALL_CROP_CODES, ALL_CROP_LABELS, 0)
                .unmask(0)           # ocean / non-CDL -> 0 (non-crop)
                .rename("crop_type")
                .toByte())
 
# ── 4. Download helper ──────────────────────────────────────────────────────
def download_ee_image(ee_image, region, crs_transform, scale_m):
    """
    Download single-band image via getDownloadURL.
 
    BUG 1 FIX: do NOT use rasterio's nodata header to replace pixel values.
    After .unmask(fill_value), the pixel data already contains the fill value
    for all masked areas.  GEE still writes nodata=0 in the TIFF header as a
    metadata artefact, but reading and acting on it would overwrite label-0
    pixels (or in this version, we've shifted labels to 1-8, so label 0 is
    now non-crop — but we still ignore the nodata header for robustness).
    """
    import urllib.request
    url = ee_image.getDownloadURL({
        "region":       region,
        "crs":          "EPSG:4326",
        "crsTransform": crs_transform,
        "format":       "GEO_TIFF",
        "scale":        scale_m,
    })
    with urllib.request.urlopen(url) as r:
        data = r.read()
    with MemoryFile(data) as mf:
        with mf.open() as ds_r:
            arr    = ds_r.read(1).astype(np.uint8)
            nodata = ds_r.nodata
            print("    shape={} nodata={} (header — ignored after .unmask())".format(
                arr.shape, nodata))
    # No nodata replacement: .unmask() already set the correct fill in pixels.
    return arr
 
# ── 5. Download ─────────────────────────────────────────────────────────────
print("\nDownloading CDL crop mask...")
crop_mask_raw = download_ee_image(crop_mask_ee, REGION, CRS_TRANSFORM, SCALE_M)
print("  unique values: {}".format(np.unique(crop_mask_raw)))
 
print("Downloading CDL crop type...")
crop_type_raw = download_ee_image(crop_type_ee, REGION, CRS_TRANSFORM, SCALE_M)
print("  unique values: {}".format(np.unique(crop_type_raw)))
 
# ── 6. Resize fallback (should not trigger after Bug 2 fix) ─────────────────
def resize_to_grid(arr, target_H, target_W, name):
    if arr.shape == (target_H, target_W):
        return arr
    from scipy.ndimage import zoom
    zh, zw = target_H / arr.shape[0], target_W / arr.shape[1]
    print("  WARNING: {} {} -> {}x{} (zoom {:.4f}, {:.4f})".format(
        name, arr.shape, target_H, target_W, zh, zw))
    return zoom(arr, (zh, zw), order=0)
 
crop_mask_arr = resize_to_grid(crop_mask_raw, H, W, "crop_mask")
crop_type_arr = resize_to_grid(crop_type_raw, H, W, "crop_type")
crop_mask_arr = (crop_mask_arr == 1).astype(bool)
crop_type_arr = crop_type_arr.astype(np.uint8)
 
# ── 7. Statistics ────────────────────────────────────────────────────────────
n_crop  = int(crop_mask_arr.sum())
n_total = crop_mask_arr.size
print("\nCrop mask summary (2020 CDL):")
print("  Cropland pixels : {:,} / {:,}  ({:.1f}%)".format(
    n_crop, n_total, n_crop / n_total * 100))
 
print("\nPer-crop pixel counts:")
for name, info in CROP_CLASSES.items():
    n   = int((crop_type_arr == info["label"]).sum())
    pct = n / n_total * 100
    print("  {:15s} (label {:d}): {:6,d} px  ({:.2f}%)".format(
        name, info["label"], n, pct))
n_noncrop = int((crop_type_arr == 0).sum())
print("  {:15s} (label 0): {:6,d} px  ({:.1f}%)".format(
    "non_crop", n_noncrop, n_noncrop / n_total * 100))
 
n_typed = int((crop_type_arr >= 1).sum())
print("\nSanity check:")
print("  crop_mask total     : {:,}".format(n_crop))
print("  crop_type typed(1-8): {:,}  (should be close)".format(n_typed))
diff = abs(n_crop - n_typed)
pct_diff = diff / max(n_crop, 1) * 100
if pct_diff > 2.0:
    print("  WARNING: {:.1f}% mismatch -- recheck CDL remap codes".format(pct_diff))
else:
    print("  OK  ({:.1f}% difference, within tolerance)".format(pct_diff))
 
# ── 8. Save ──────────────────────────────────────────────────────────────────
out_dir = Path(PROC_ROOT)
out_dir.mkdir(parents=True, exist_ok=True)
 
np.save(out_dir / "cdl_crop_mask.npy", crop_mask_arr)
np.save(out_dir / "cdl_crop_type.npy", crop_type_arr)
 
transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W, H)
 
with rasterio.open(
    out_dir / "cdl_crop_mask.tif", "w", driver="GTiff",
    height=H, width=W, count=1, dtype=np.uint8,
    crs="EPSG:4326", transform=transform, compress="lzw"
) as dst:
    dst.write(crop_mask_arr.astype(np.uint8), 1)
    dst.update_tags(1, description="CDL 2020 binary crop mask (1=crop 0=non-crop)")
 
with rasterio.open(
    out_dir / "cdl_crop_type.tif", "w", driver="GTiff",
    height=H, width=W, count=1, dtype=np.uint8,
    crs="EPSG:4326", transform=transform, compress="lzw"
) as dst:
    dst.write(crop_type_arr, 1)
    dst.update_tags(1, description=(
        "CDL 2020 crop type: 0=non-crop "
        "1=winter_wheat 2=spring_wheat 3=corn 4=sorghum "
        "5=cotton 6=alfalfa 7=barley 8=durum_wheat"))
 
# ── 9. QC plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
 
axes[0].imshow(crop_mask_arr,
               extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
               origin="upper", cmap="Greens", vmin=0, vmax=1, aspect="auto")
axes[0].set_title("CDL 2020 binary crop mask\n{:,} px  ({:.1f}%)".format(
    n_crop, n_crop / n_total * 100), fontsize=11)
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
 
n_cls = len(CROP_CLASSES)
cmap  = plt.colormaps.get_cmap("tab10").resampled(n_cls)
disp  = np.where(crop_type_arr == 0, np.nan, crop_type_arr.astype(float) - 1)
im    = axes[1].imshow(disp,
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                       origin="upper", cmap=cmap,
                       vmin=-0.5, vmax=n_cls - 0.5, aspect="auto")
cbar = plt.colorbar(im, ax=axes[1], ticks=list(range(n_cls)), shrink=0.8)
cbar.set_ticklabels(list(CROP_CLASSES.keys()), fontsize=8)
axes[1].set_title("CDL 2020 crop type classes\n(0=non-crop excluded)", fontsize=11)
axes[1].set_xlabel("Longitude")
 
plt.tight_layout()
plt.savefig(out_dir / "qc_cdl_crop_mask.png", dpi=150, bbox_inches="tight")
plt.close()
 
print("\nSaved to {}:".format(out_dir))
for f in ["cdl_crop_mask.npy", "cdl_crop_type.npy",
          "cdl_crop_mask.tif", "cdl_crop_type.tif", "qc_cdl_crop_mask.png"]:
    p = out_dir / f
    print("  {} {}".format("OK" if p.exists() else "MISSING", f))
print("\nStep 4a v3 complete.")
print("After running, confirm winter_wheat label 1 has >1000 pixels.")
print("Next: 04b_irrigation_fraction.py")
 



#old code
#import ee
#import numpy as np
#import rasterio
#from rasterio.warp import reproject, Resampling
#from pathlib import Path
#from config import BBOX, PROC_ROOT

#ee.Initialize(project="your-gcp-project-id")

# CDL crop codes for major CONUS semi-arid crops
#CROP_CODES = {
#    "winter_wheat": [24],
#    "spring_wheat": [26],
#    "corn":         [1],
#    "sorghum":      [4],
#    "cotton":       [2],
#    "alfalfa":      [36],
#    "barley":       [21],
#    "durum_wheat":  [27],
#}
#ALL_CROP_CODES = [c for codes in CROP_CODES.values() for c in codes]

#region = ee.Geometry.Rectangle(list(BBOX))

# Use 2020 CDL as static mask (update annually in operational system)
#cdl = ee.Image("USDA/NASS/CDL/2020").select("cropland").clip(region)

# Crop pixel mask: 1 = any crop, 0 = non-crop
#crop_mask_ee = cdl.remap(ALL_CROP_CODES, [1]*len(ALL_CROP_CODES), 0).rename("crop_mask")
# Crop type class (multi-class for per-crop heads)
#crop_type_ee = cdl.remap(ALL_CROP_CODES, list(range(len(ALL_CROP_CODES))), 0).rename("crop_type")

# Export at 0.01° (~1 km) — will be resampled to SMAP grid
#for name, img in [("crop_mask", crop_mask_ee), ("crop_type", crop_type_ee)]:
#    task = ee.batch.Export.image.toDrive(
#        image=img, description=f"CDL_{name}_2020",
#        folder="drought_forecast", scale=1000,
#        region=region, crs="EPSG:4326", maxPixels=1e12
#    )
#    task.start()
#    print(f"Exporting CDL {name}...")
