#!/usr/bin/env python3
"""
01_download_smap.py

Download one daily SMAP Level-4 root-zone soil moisture granule for CONUS.

Dataset
-------
NASA SMAP Level-4 Global 3-hourly Soil Moisture Product:
    short_name = SPL4SMGP
    version    = 008

Why only one granule per day?
-----------------------------
SPL4SMGP is available every 3 hours, giving about 8 granules per day.
For a long-term drought forecasting workflow targeting 8–40 day lead times,
root-zone soil moisture evolves mainly on weekly-to-monthly timescales.
Therefore, one consistently timed daily snapshot is sufficient and greatly
reduces storage and preprocessing burden.

A diagnostic using 01_check_smap_times.py check showed that the 21:00 UTC granule has complete coverage
for tested years, so this script selects only 21:00 UTC granules.

Important
---------
Earthaccess may return an extra 21:00 UTC granule from the day before
START_DATE because the temporal search window overlaps that file.
This script explicitly filters selected granules back to START_DATE–END_DATE.
"""

import earthaccess
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import BBOX, START_DATE, END_DATE, DATA_ROOT


# ============================================================
# User settings
# ============================================================

SHORT_NAME = "SPL4SMGP"
VERSION = "008"

# Selected daily SMAP analysis time
TARGET_HOUR = 21   # 21:00 UTC


# ============================================================
# Helper function
# ============================================================

def get_granule_datetime(granule):
    """
    Extract beginning datetime from an Earthaccess SMAP granule.
    Returns timezone-aware UTC datetime.
    """
    t = granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


# ============================================================
# Login
# ============================================================

earthaccess.login(strategy="netrc")


# ============================================================
# Search SMAP granules
# ============================================================

print("Searching for SMAP granules...")
print(f"Dataset      : {SHORT_NAME}.v{VERSION}")
print(f"Date range   : {START_DATE} to {END_DATE}")
print(f"BBOX         : {BBOX}")
print(f"Target hour  : {TARGET_HOUR:02d}:00 UTC")

results = earthaccess.search_data(
    short_name=SHORT_NAME,
    version=VERSION,
    temporal=(START_DATE, END_DATE),
    bounding_box=BBOX,
)

print(f"\nFound {len(results)} total granules")

if len(results) == 0:
    raise RuntimeError(
        "No SMAP granules found. Check short_name, version, date range, BBOX, "
        "or Earthdata authentication."
    )


# ============================================================
# Inspect first few timestamps
# ============================================================

print("\nFirst few granule timestamps returned by Earthaccess:")
for g in results[:5]:
    print(" ", get_granule_datetime(g))


# ============================================================
# Select one granule per day
# ============================================================

start_date = pd.to_datetime(START_DATE).date()
end_date = pd.to_datetime(END_DATE).date()

selected = []

for g in results:
    dt = get_granule_datetime(g)
    day = dt.date()

    # Keep only 21:00 UTC granules inside the requested date range.
    # This removes any extra previous-day granule returned by Earthaccess.
    if dt.hour == TARGET_HOUR and start_date <= day <= end_date:
        selected.append(g)

selected = sorted(selected, key=get_granule_datetime)

print(
    f"\nSelected {len(selected)} daily granules at "
    f"{TARGET_HOUR:02d}:00 UTC within {START_DATE}–{END_DATE}"
)

if len(selected) == 0:
    raise RuntimeError(
        f"No granules found at {TARGET_HOUR:02d}:00 UTC within the requested date range."
    )


# ============================================================
# Optional sanity check
# ============================================================

expected_days = len(pd.date_range(start=start_date, end=end_date, freq="D"))

print(f"Expected number of daily timesteps: {expected_days}")
print(f"Selected number of granules       : {len(selected)}")

if len(selected) != expected_days:
    print(
        "\nWARNING: Selected granule count does not match expected number of days."
    )
    print(
        "This may indicate missing SMAP files for some days. "
        "The download can still proceed, but check gaps before modeling."
    )


# ============================================================
# Download
# ============================================================

out_dir = Path(DATA_ROOT) / "smap"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading to: {out_dir}")
print("Using 4 threads to reduce NASA/NSIDC 502/503 server errors.\n")

earthaccess.download(
    selected,
    str(out_dir),
    threads=4,
)

print("\nDownload step complete.")
print(f"Downloaded daily {TARGET_HOUR:02d}:00 UTC SMAP granules to: {out_dir}")
