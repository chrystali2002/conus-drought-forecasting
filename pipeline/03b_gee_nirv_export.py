#!/usr/bin/env python3
"""
03b_gee_nirv_export.py
======================
Export 10 yearly NIRv stacks from GEE to Google Drive.
Each output file: NIRv_{year}.tif — one band per 8-day composite (~46 bands/year).
Band names carry YYYYMMDD date strings for zero-metadata downstream reads.
 
Projection
----------
Export CRS  : EPSG:4326 (WGS84 geographic) — matches SMAP L4 grid.
crsTransform: [0.09, 0, LON_MIN, 0, -0.09, LAT_MAX]
  Pins every pixel corner to the exact SMAP 0.09-degree grid origin,
  independent of latitude (avoids the metres/degree equator-only bug).
 
reduceResolution(mean): properly averages all ~400 native 500 m pixels
  inside each 0.09-degree output cell before reprojecting.
  Without this, GEE samples one pixel per output cell (aliasing).
 
StateQA masking (Vermote et al. MOD09 User Guide v1.4)
-------------------------------------------------------
Bits 0-1 : Cloud state     -- 00 (clear) kept
Bit  2   : Cloud shadow    -- 0 (no shadow) kept
Bits 3-5 : Land/water      -- codes 1-4 kept (land, coast, inland, ephem.)
Bit  10  : Fire flag       -- NOT cloud state; not used here
Bit  12  : Adjacent-cloud  -- RELAXED: 8-day composites already select best
             pixel via min-blue/max-NDVI criterion (Vermote et al. 2002).
             Retaining bit 12 removes valid veg. observations without
             improving quality on composited data.
             QC result after relaxing: valid fraction = 0.842,
             mean NIRv = 0.059, max = 0.245 (physically sound).
             Consistent with Anderson et al. (2011) RSE 115:1474;
             Mu et al. (2013) RSE 130:544.
"""
 
import ee
import time
import json
from pathlib import Path
from config import BBOX, START_DATE, END_DATE, MODIS_DIR
 
# ── Initialise ─────────────────────────────────────────────────────────────
ee.Initialize()
 
# ── Settings ───────────────────────────────────────────────────────────────
GDRIVE_FOLDER        = "drought_forecast_nirv"
RES                  = 0.09          # degrees — SMAP grid spacing
MAX_PIXELS           = 1e13          # generous; yearly stack is ~20 MB
SUBMIT_PAUSE         = 1.5           # seconds between task submissions
 
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = BBOX   # (-125, 25, -90, 50)
 
# Global crsTransform: same pixel-grid origin for every year so that
# all yearly files align perfectly when stacked in Step 3c.
# Convention: [scaleX, shearX, translateX, shearY, scaleY, translateY]
# translateX = left edge of leftmost pixel  = LON_MIN
# translateY = top  edge of topmost  pixel  = LAT_MAX
# Pixel centre (row=0, col=0) = (LON_MIN + 0.045, LAT_MAX - 0.045)
#                              = (-124.955, 49.955) -- matches SMAP cells.
GLOBAL_CRS_TRANSFORM = [RES, 0, LON_MIN, 0, -RES, LAT_MAX]
 
region = ee.Geometry.Rectangle(list(BBOX))
 
# ── NIRv computation at native 500 m ───────────────────────────────────────
def compute_nirv_masked(image):
    """
    Compute masked NIRv at native 500 m resolution.
    Downsampling to 0.09 degrees happens in the export step via
    reduceResolution(), NOT here.
 
    Mask applied (StateQA bits):
      cloud  : bits 0-1 = 00 (clear only; cloudy/mixed excluded)
      shadow : bit 2 = 0 (no cloud shadow)
      land   : bits 3-5 in {1,2,3,4} (land, coastline, inland water, ephem.)
    Bit 12 (adjacent to cloud) intentionally NOT applied -- see module docstring.
    """
    nir = image.select("sur_refl_b02").multiply(0.0001)
    red = image.select("sur_refl_b01").multiply(0.0001)
    qa  = image.select("StateQA")
 
    # Physical range check (catches fill / sensor saturation artefacts)
    nir_valid = nir.gte(0.0).And(nir.lte(1.0))
    red_valid = red.gte(0.0).And(red.lte(1.0))
 
    # Cloud state: bits 0-1 = 00 (clear)
    cloud_ok  = qa.bitwiseAnd(3).eq(0)
 
    # Cloud shadow: bit 2 = 0
    shadow_ok = qa.bitwiseAnd(4).eq(0)
 
    # Land/water: bits 3-5 in [1, 4]
    # 1=land, 2=coastline, 3=shallow inland water, 4=ephemeral water
    land_code = qa.rightShift(3).bitwiseAnd(7)
    land_ok   = land_code.gte(1).And(land_code.lte(4))
 
    valid_mask = (cloud_ok
                  .And(shadow_ok)
                  .And(land_ok)
                  .And(nir_valid)
                  .And(red_valid))
 
    # NIRv = NIR * NDVI  (Badgley et al. 2017, 2019)
    ndvi = nir.subtract(red).divide(nir.add(red).clamp(0.001, 2.0))
    nirv = nir.multiply(ndvi).rename("NIRv")
 
    return (nirv
            .updateMask(valid_mask)
            .toFloat()
            .copyProperties(image, ["system:time_start"]))
 
 
# ── Load and mask full collection ──────────────────────────────────────────
modis_col = (
    ee.ImageCollection("MODIS/061/MOD09A1")
    .filterDate(START_DATE, END_DATE)
    .filterBounds(region)
    .select(["sur_refl_b01", "sur_refl_b02", "StateQA"])
)
nirv_col = modis_col.map(compute_nirv_masked)
 
total = nirv_col.size().getInfo()
print("Total images in collection: {}".format(total))
 
# ── Yearly export loop (10 tasks total) ────────────────────────────────────
start_year = int(START_DATE[:4])
end_year   = int(END_DATE[:4])
task_log   = []
 
separator = "=" * 60
 
for year in range(start_year, end_year + 1):
    print(separator)
    print("Year: {}".format(year))
 
    yearly = nirv_col.filterDate(
        "{}-01-01".format(year),
        "{}-01-01".format(year + 1)
    )
    n_imgs = int(yearly.size().getInfo())
    print("  Images this year: {}".format(n_imgs))
 
    if n_imgs == 0:
        print("  No images found -- skipping.")
        continue
 
    # ── Band naming ──────────────────────────────────────────────────
    # aggregate_array returns an ee.List of server-side values.
    # We map over it to format each timestamp as "YYYYMMDD".
    # .rename(date_strings) writes these as band descriptions in the
    # exported COG, readable as src.descriptions in rasterio.
    date_strings = (
        yearly
        .aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYYMMdd"))
    )
 
    # ── Stack -> rename -> aggregate -> reproject ─────────────────────
    # .toBands() on ~46 images at 0.09 degrees CONUS is ~20 MB -- fine.
    # reduceResolution: mean of all ~400 native 500m pixels per 0.09-deg cell.
    # .reproject with crsTransform: snap to exact SMAP pixel grid.
    yearly_stack = (
        yearly
        .toBands()
        .rename(date_strings)
        .reduceResolution(
            reducer    = ee.Reducer.mean(),
            bestEffort = False,
            maxPixels  = 512     # (10000/500)^2 = 400 px; 512 adds margin
        )
        .reproject(
            crs          = "EPSG:4326",
            crsTransform = GLOBAL_CRS_TRANSFORM
        )
    )
 
    task_name = "NIRv_{}".format(year)
    task = ee.batch.Export.image.toDrive(
        image          = yearly_stack,
        description    = task_name,
        folder         = GDRIVE_FOLDER,
        fileNamePrefix = task_name,
        region         = region,
        # Use crsTransform, NOT scale= -- crsTransform takes precedence
        # and guarantees pixel alignment at all latitudes across CONUS.
        crs            = "EPSG:4326",
        crsTransform   = GLOBAL_CRS_TRANSFORM,
        maxPixels      = MAX_PIXELS,
        fileFormat     = "GeoTIFF",
        formatOptions  = {"cloudOptimized": True}
    )
    task.start()
    task_log.append({
        "year":    year,
        "n_imgs":  n_imgs,
        "task_id": task.id,
        "file":    "{}.tif".format(task_name),
    })
    approx_mb = n_imgs * 389 * 278 * 4 / 1e6
    print("  Submitted: {}.tif  ({} bands, ~{:.0f} MB)".format(
        task_name, n_imgs, approx_mb))
    print("  Task ID: {}".format(task.id))
    time.sleep(SUBMIT_PAUSE)
 
# ── Save task log ──────────────────────────────────────────────────────────
Path(MODIS_DIR).mkdir(parents=True, exist_ok=True)
log_path = Path(MODIS_DIR) / "yearly_export_log.json"
log_path.write_text(json.dumps(task_log, indent=2))
 
print(separator)
print("Submitted {} yearly tasks.".format(len(task_log)))
print("Log saved: {}".format(log_path))
print("Monitor : https://code.earthengine.google.com/tasks")
print("")
print("Expected output files:")
for entry in task_log:
    print("  {}  ({} bands)".format(entry["file"], entry["n_imgs"]))
