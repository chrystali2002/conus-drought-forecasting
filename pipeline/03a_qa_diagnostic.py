"""
03a_qa_diagnostic.py
====================
Before committing to the 10-year CONUS export, verify that the StateQA
masking produces physically reasonable valid-pixel fractions.

Expected results for a summer 8-day composite over the SGP tile:
  No mask (raw reflectance)     : ~85–95% finite pixels (land only)
  Correct cloud + shadow mask   : ~65–80%  (some cloud cover expected)
  Fire-bit mask (WRONG)         : ~3–10%   — this was the 3.77% bug source
  Correct FULL mask             : ~55–75%  (add land/water)

Run this as:  python 03a_qa_diagnostic.py
Inspect the output PNG before running 03a_gee_nirv_export.py.
"""

import ee
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from pathlib import Path
from config import PROC_ROOT

ee.Initialize()

# ── Test region: single SGP tile (small enough for getDownloadURL) ──
# Approx 7° × 5° — fits in GEE getDownloadURL limit at 0.01° scale
TEST_REGION = ee.Geometry.Rectangle([-104, 34, -97, 39])   # W Kansas / N Oklahoma
TEST_DATE   = "2020-07-01"
TEST_DATE_END = "2020-07-09"   # one 8-day composite
TEST_SCALE  = 0.09             # 0.09° = SMAP grid; use for QA check

out_dir = Path(PROC_ROOT)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load one 8-day image ──
img_col = (ee.ImageCollection("MODIS/061/MOD09A1")
             .filterDate(TEST_DATE, TEST_DATE_END)
             .filterBounds(TEST_REGION))

n = img_col.size().getInfo()
print(f"Images in test window: {n}")
assert n >= 1, "No image found — check TEST_DATE"

img = ee.Image(img_col.first())
qa  = img.select("StateQA")
nir = img.select("sur_refl_b02").multiply(0.0001)
red = img.select("sur_refl_b01").multiply(0.0001)

# ── Build progressive masks ──
# 1. No mask — raw NIRv (unmasked)
ndvi_raw  = nir.subtract(red).divide(nir.add(red).clamp(0.001, 2.0))
nirv_raw  = nir.multiply(ndvi_raw).rename("NIRv_noMask")

# 2. CORRECT cloud mask: bits 0-1 = 00 (clear)
cloud_ok     = qa.bitwiseAnd(3).eq(0)            # bits 0-1 = clear
nirv_cloud   = nirv_raw.updateMask(cloud_ok).rename("NIRv_cloudOnly")

# 3. CORRECT shadow mask: bit 2 = 0 (no shadow)
shadow_ok    = qa.bitwiseAnd(4).eq(0)            # bit 2 = 0
nirv_shadow  = nirv_raw.updateMask(cloud_ok.And(shadow_ok)).rename("NIRv_cloudShadow")

# 4. CORRECT land mask: bits 3-5 in {001,010,011,100} = non-ocean land/water
#    001=land, 010=coastline, 011=shallow inland water, 100=ephemeral water
#    Exclude 000=shallow ocean, 110=moderate ocean, 111=deep ocean
land_code = qa.rightShift(3).bitwiseAnd(7)        # extract bits 3-5
land_ok   = land_code.gte(1).And(land_code.lte(4))
nirv_full  = nirv_raw.updateMask(cloud_ok.And(shadow_ok).And(land_ok)).rename("NIRv_fullMask")

# 5. WRONG fire-bit mask (the original bug — for comparison)
fire_bit_ok  = qa.bitwiseAnd(1 << 10).eq(0)      # bit 10 = fire flag ← WRONG use
nirv_firebug = nirv_raw.updateMask(fire_bit_ok).rename("NIRv_fireBitBug")

# ── Download all variants as small numpy arrays ──
def ee_to_numpy(image, region, scale_deg):
    """Download a single-band GEE image as a numpy array via getDownloadURL."""
    url = image.getDownloadURL({
        "scale":  scale_deg * 111320,   # metres
        "region": region,
        "format": "GEO_TIFF",
        "crs":    "EPSG:4326",
    })
    import urllib.request, io
    with urllib.request.urlopen(url) as r:
        data = r.read()
    with MemoryFile(data) as mf:
        with mf.open() as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

print("\nDownloading test images from GEE (small region, ~5 s each)...")
variants = {
    "No mask (raw)"            : nirv_raw,
    "Cloud only (bits 0-1)"    : nirv_cloud,
    "Cloud + shadow (bits 0-2)": nirv_shadow,
    "CORRECT full mask"        : nirv_full,
    "WRONG fire-bit mask (BUG)": nirv_firebug,
}

arrays = {}
for name, img_v in variants.items():
    try:
        arr = ee_to_numpy(img_v, TEST_REGION, TEST_SCALE)
        arrays[name] = arr
        finite_pct = np.isfinite(arr).mean() * 100
        print(f"  {name:35s}: {finite_pct:5.1f}% valid pixels  "
              f"  mean NIRv = {np.nanmean(arr):.3f}")
    except Exception as exc:
        print(f"  {name}: FAILED — {exc}")

# ── QC Plot: side-by-side comparison ──
fig, axes = plt.subplots(1, len(arrays), figsize=(5*len(arrays), 5))
for ax, (name, arr) in zip(axes, arrays.items()):
    finite_pct = np.isfinite(arr).mean() * 100
    im = ax.imshow(arr, cmap="YlGn", vmin=-0.1, vmax=0.5, aspect="auto")
    ax.set_title(f"{name}\n{finite_pct:.1f}% valid", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="NIRv")

plt.suptitle(f"MOD09A1 StateQA mask comparison  |  SGP tile  |  {TEST_DATE}",
             fontsize=11, y=1.01)
plt.tight_layout()
qc_path = out_dir / "qc_nirv_qa_comparison.png"
plt.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nQC plot saved → {qc_path}")
print("\nDecision rule:")
print("  'CORRECT full mask' valid % should be 55–80% for a summer SGP composite.")
print("  If 'WRONG fire-bit mask' ~ your earlier result → confirmed bug source.")
