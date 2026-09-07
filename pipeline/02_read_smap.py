#!/usr/bin/env python3
"""
02_read_smap.py

Read downloaded daily SMAP Level-4 root-zone soil moisture granules and
convert them into one gridded NetCDF file over the CONUS processing domain.

Input
-----
Downloaded SMAP SPL4SMGP HDF5 files from:

    data/raw/smap/

These files were downloaded by 01_download_smap.py using one daily
21:00 UTC granule per day. granule spanning 21:00–00:00 UTC, filename-stamped 22:30 UTC" — but the data and the attributes are defensible as they stand

Output
------
    data/processed/smap_daily_2230utc_conus.nc

Main variable
-------------
rzsm : Root-zone soil moisture, 0–100 cm, units m3/m3

Notes
-----
Because 01_download_smap.py selects only one granule per day, this script
creates a daily 21:00 UTC snapshot dataset, not a true 24-hour daily mean.

If duplicate files exist for the same day, this script averages them.
"""

import re
import h5py
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from config import BBOX, DATA_ROOT, PROC_ROOT


# ============================================================
# Helper functions
# ============================================================

def extract_datetime_from_filename(filepath: Path) -> pd.Timestamp:
    """
    Extract timestamp from SMAP filename.

    Example filename:
    SMAP_L4_SM_gph_20240101T210000_Vv8010_001.h5
    """
    match = re.search(r"(\d{8})T(\d{6})", filepath.name)

    if match is None:
        raise ValueError(f"Could not extract timestamp from filename: {filepath.name}")

    date_part = match.group(1)
    time_part = match.group(2)

    return pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M%S")


def read_smap_granule(filepath: Path) -> dict:
    """
    Read a single SPL4SMGP HDF5 granule.

    Returns
    -------
    dict containing:
        rzsm : root-zone soil moisture array
        sfsm : surface soil moisture array
        lat  : latitude grid
        lon  : longitude grid
    """

    with h5py.File(filepath, "r") as f:
        rzsm = f["Geophysical_Data/sm_rootzone"][:]
        sfsm = f["Geophysical_Data/sm_surface"][:]

        lat = f["cell_lat"][:]
        lon = f["cell_lon"][:]

        # Replace SMAP fill values with NaN.
        # SMAP fill values are usually very negative, e.g. -9999.
        rzsm = rzsm.astype(np.float32)
        sfsm = sfsm.astype(np.float32)

        rzsm[rzsm < -100] = np.nan
        sfsm[sfsm < -100] = np.nan

    return {
        "rzsm": rzsm,
        "sfsm": sfsm,
        "lat": lat,
        "lon": lon,
    }


def get_bbox_masks(lat: np.ndarray, lon: np.ndarray, bbox):
    """
    Build row/column masks for the target bounding box.

    bbox format:
        lon_min, lat_min, lon_max, lat_max
    """

    lon_min, lat_min, lon_max, lat_max = bbox

    col_mask = (lon[0, :] >= lon_min) & (lon[0, :] <= lon_max)
    row_mask = (lat[:, 0] >= lat_min) & (lat[:, 0] <= lat_max)

    if row_mask.sum() == 0 or col_mask.sum() == 0:
        raise ValueError(
            "BBOX produced an empty SMAP subset. "
            "Check BBOX order: should be lon_min, lat_min, lon_max, lat_max."
        )

    return row_mask, col_mask


def crop_to_bbox(data: np.ndarray, row_mask, col_mask) -> np.ndarray:
    """
    Crop a 2D SMAP array to the target bounding box.
    """
    return data[np.ix_(row_mask, col_mask)]


# ============================================================
# Main processing
# ============================================================

def build_smap_stack(smap_dir: Path, bbox) -> xr.Dataset:
    """
    Stack downloaded daily SMAP granules into an xarray Dataset.

    If one file exists per day:
        the daily value is that 21:00 UTC snapshot.

    If multiple files exist for the same day:
        the daily value is the mean across duplicate files.
    """

    files = sorted(smap_dir.glob("*.h5"))

    if len(files) == 0:
        raise FileNotFoundError(f"No SMAP .h5 files found in {smap_dir}")

    print(f"Found {len(files)} downloaded SMAP files in {smap_dir}")

    # Read first granule to define spatial grid
    g0 = read_smap_granule(files[0])

    row_mask, col_mask = get_bbox_masks(
        g0["lat"],
        g0["lon"],
        bbox,
    )

    lats = g0["lat"][:, 0][row_mask].astype(np.float32)
    lons = g0["lon"][0, :][col_mask].astype(np.float32)

    print(f"Subset grid size: {len(lats)} rows × {len(lons)} cols")
    print(f"Latitude range : {np.nanmin(lats):.3f} to {np.nanmax(lats):.3f}")
    print(f"Longitude range: {np.nanmin(lons):.3f} to {np.nanmax(lons):.3f}")

    records = []

    print("\nReading and cropping SMAP granules...")

    for f in tqdm(files):
        try:
            dt = extract_datetime_from_filename(f)
            day = dt.normalize()

            g = read_smap_granule(f)

            rzsm_crop = crop_to_bbox(g["rzsm"], row_mask, col_mask)

            records.append(
                {
                    "time": day,
                    "datetime_utc": dt,
                    "rzsm": rzsm_crop.astype(np.float32),
                    "filename": f.name,
                }
            )

        except Exception as e:
            print(f"\nWARNING: Skipping file due to error: {f.name}")
            print(f"Reason: {e}")

    if len(records) == 0:
        raise RuntimeError("No valid SMAP granules were read.")

    # Group by date.
    df = pd.DataFrame(records)

    print("\nTemporal coverage before daily grouping:")
    print(f"First date: {df['time'].min().date()}")
    print(f"Last date : {df['time'].max().date()}")
    print(f"Unique days: {df['time'].nunique()}")

    times = []
    rzsm_stack = []

    print("\nBuilding daily stack...")

    for day, grp in tqdm(df.groupby("time")):
        arrays = np.stack(grp["rzsm"].values).astype(np.float32)

        # Usually only one file per day.
        # If duplicates exist, average them.
        daily_rzsm = np.nanmean(arrays, axis=0).astype(np.float32)

        times.append(day)
        rzsm_stack.append(daily_rzsm)

    rzsm_arr = np.stack(rzsm_stack).astype(np.float32)

    print("\nFinal SMAP array:")
    print(f"Shape: {rzsm_arr.shape}")
    print(f"RZSM min : {np.nanmin(rzsm_arr):.4f}")
    print(f"RZSM max : {np.nanmax(rzsm_arr):.4f}")
    print(f"RZSM mean: {np.nanmean(rzsm_arr):.4f}")
    print(f"NaN fraction: {np.isnan(rzsm_arr).mean() * 100:.2f}%")

    ds = xr.Dataset(
        data_vars={
            "rzsm": (
                ["time", "lat", "lon"],
                rzsm_arr,
            )
        },
        coords={
            "time": pd.to_datetime(times),
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "Daily 21:00 UTC SMAP Level-4 Root-Zone Soil Moisture over CONUS",
            "source_product": "SPL4SMGP Version 008",
            "processing_note": (
                "One 21:00 UTC SMAP Level-4 granule was selected per day. "
                "This is a daily snapshot dataset, not a 24-hour daily mean."
            ),
            "bbox": str(bbox),
        },
    )

    ds["rzsm"].attrs = {
        "long_name": "Root-zone soil moisture",
        "description": "SMAP Level-4 root-zone soil moisture, 0-100 cm",
        "units": "m3 m-3",
        "daily_representation": "21:00 UTC snapshot",
    }

    return ds


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":

    smap_dir = Path(DATA_ROOT) / "smap"
    out_dir = Path(PROC_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_smap = build_smap_stack(
        smap_dir=smap_dir,
        bbox=BBOX,
    )

    out_path = out_dir / "smap_daily_2230utc_conus.nc"

    encoding = {
        "rzsm": {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "_FillValue": -9999.0,
        }
    }

    ds_smap.to_netcdf(
        out_path,
        encoding=encoding,
    )

    print(f"\nSaved → {out_path}")
    print(f"Output shape: {ds_smap['rzsm'].shape}")
    print("Step 02 complete.")
