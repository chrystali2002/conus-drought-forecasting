#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smap_coverage_stats.py
======================
Quantify the spatial coverage of the retained SMAP L4 granules, so the methods
can say how complete the record is rather than asserting completeness.

Reports coverage of the PROCESSED daily file, which is what the model actually
consumed - more relevant than a pre-download survey of granule windows.

Two views:
  domain    coverage over the 3,093 water-limited cropland cells. This is the
            number to quote: it is what the forecasts depend on.
  grid      coverage over the full 321 x 656 box, which includes ocean and
            Canada and is therefore always low. Reported only so the two are
            not confused.

USAGE
    python smap_coverage_stats.py
    python smap_coverage_stats.py --raw    # also scan the raw granules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    sys.exit("needs xarray:  pip install xarray")

try:
    from config import PROC_ROOT
except ImportError:
    PROC_ROOT = "data/processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="also scan data/raw/smap granule filenames")
    args = ap.parse_args()
    proc = Path(PROC_ROOT)

    # ---- domain mask
    ms = xr.open_zarr(proc / "master_dataset.zarr", consolidated=True)
    wlc = ms["water_limited_crop"].values > 0.5
    print(f"domain (water_limited_crop): {int(wlc.sum())} cells\n")

    # ---- processed daily RZSM
    nc = proc / "smap_daily_2230utc_conus.nc"
    if not nc.exists():
        sys.exit(f"not found: {nc}")
    d = xr.open_dataset(nc)
    rz = d["rzsm"]
    t = d["time"].values
    print(f"file      {nc.name}")
    print(f"steps     {len(t)}   {str(t[0])[:10]} to {str(t[-1])[:10]}")
    print(f"grid      {rz.shape[1:]}")

    # the processed file may be on the pre-harmonisation grid
    same_grid = rz.shape[1:] == wlc.shape
    if not same_grid:
        print(f"NOTE: file grid {rz.shape[1:]} != domain grid {wlc.shape};"
              f" domain coverage will be taken from anomalies.zarr instead")

    # ---- per-timestep coverage over the whole box
    print("\nGRID COVERAGE (full box, includes ocean and non-CONUS land)")
    frac = []
    for i in range(len(t)):
        v = rz.isel(time=i).values
        frac.append(float(np.isfinite(v).mean()))
    frac = np.array(frac)
    print(f"  mean   {frac.mean()*100:.2f} %")
    print(f"  min    {frac.min()*100:.2f} %  on {str(t[int(frac.argmin())])[:10]}")
    print(f"  steps below 90% of the median: "
          f"{int((frac < 0.9*np.median(frac)).sum())} of {len(frac)}")

    # ---- domain coverage, from the harmonised anomalies if needed
    print("\nDOMAIN COVERAGE (water-limited cropland - quote this one)")
    src = rz if same_grid else xr.open_zarr(
        proc / "anomalies.zarr", consolidated=True)["rzsm_anom"]
    if not same_grid:
        print("  (measured on anomalies.zarr rzsm_anom)")
    n = src.sizes["time"]
    dfrac = np.empty(n)
    for i in range(n):
        v = src.isel(time=i).values
        dfrac[i] = float(np.isfinite(v[wlc]).mean())
    print(f"  mean per timestep        {dfrac.mean()*100:.3f} %")
    print(f"  median per timestep      {np.median(dfrac)*100:.3f} %")
    print(f"  minimum                  {dfrac.min()*100:.3f} %")
    print(f"  timesteps at 100 %       {int((dfrac >= 0.9999).sum())} of {n} "
          f"({(dfrac >= 0.9999).mean()*100:.1f} %)")
    print(f"  timesteps below 99 %     {int((dfrac < 0.99).sum())}")

    # per-cell completeness: how many cells are ever missing
    stack = np.empty((n, int(wlc.sum())), dtype=bool)
    for i in range(n):
        stack[i] = np.isfinite(src.isel(time=i).values[wlc])
    per_cell = stack.mean(axis=0)
    print(f"\n  per-cell completeness:")
    print(f"    cells complete for every timestep  {int((per_cell >= 0.9999).sum())} "
          f"of {len(per_cell)}")
    print(f"    worst cell                         {per_cell.min()*100:.2f} %")
    print(f"    mean across cells                  {per_cell.mean()*100:.3f} %")

    # ---- raw granule filenames
    if args.raw:
        raw = Path("data/raw/smap")
        files = sorted(raw.glob("SMAP_L4_SM_gph_*.h5"))
        print(f"\nRAW GRANULES: {len(files)} files in {raw}")
        import re
        from collections import Counter
        stamps = Counter()
        days = set()
        for f in files:
            m = re.search(r"_(\d{8})T(\d{6})_", f.name)
            if m:
                stamps[m.group(2)] += 1
                days.add(m.group(1))
        for s, c in sorted(stamps.items()):
            print(f"  timestamp {s[:2]}:{s[2:4]} UTC   {c} files")
        print(f"  distinct days {len(days)}")
        if days:
            print(f"  span {min(days)} to {max(days)}")
            import datetime as dt
            d0 = dt.datetime.strptime(min(days), "%Y%m%d").date()
            d1 = dt.datetime.strptime(max(days), "%Y%m%d").date()
            expected = (d1 - d0).days + 1
            print(f"  expected days in span {expected}, missing "
                  f"{expected - len(days)}")

    print("\nSUGGESTED METHODS WORDING")
    print(f'  "Root-zone soil moisture was available for {dfrac.mean()*100:.1f} % of')
    print(f'   domain cell-timesteps, with {(dfrac >= 0.9999).mean()*100:.0f} % of timesteps')
    print( '   fully covered."   (adjust to match the numbers above)')


if __name__ == "__main__":
    main()
