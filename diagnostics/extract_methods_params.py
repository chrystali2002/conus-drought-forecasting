#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_methods_params.py
=========================
Gather the numbers the methods section needs, from the scripts and the data
themselves rather than from memory.

Two kinds of output:

  MEASURED   read directly from config.py, the zarr stores, or the .npy files.
             These are facts and can be pasted into the manuscript.

  INSPECT    printed as the relevant source lines from the pipeline scripts, for
             you to read and paraphrase. Anything involving an algorithm - how
             the irrigation correction is formulated, how 8-day NIRv becomes
             daily - cannot be reduced to a number by a script, so this prints
             the code and leaves the wording to you.

USAGE
    python extract_methods_params.py            # everything
    python extract_methods_params.py --measured # skip the source listings
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    sys.exit("needs xarray:  pip install xarray zarr")


def rule(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show(path, patterns, context=2, limit=40):
    """Print lines matching any pattern, with a little context."""
    p = Path(path)
    if not p.exists():
        print(f"  [{path} not found]")
        return
    lines = p.read_text(errors="replace").splitlines()
    hits, shown = [], 0
    for i, ln in enumerate(lines):
        if any(re.search(pat, ln, re.I) for pat in patterns):
            hits.append(i)
    if not hits:
        print(f"  [no matches in {path}]")
        return
    printed = set()
    for i in hits:
        if shown >= limit:
            print(f"  ... ({len(hits) - shown} more matches)")
            break
        for j in range(max(0, i - context), min(len(lines), i + context + 1)):
            if j not in printed:
                mark = ">>" if j == i else "  "
                print(f"  {mark} {j+1:4d}| {lines[j].rstrip()[:110]}")
                printed.add(j)
        shown += 1
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", action="store_true",
                    help="only the values that can be measured")
    args = ap.parse_args()

    # ---------------------------------------------------------------- measured
    rule("MEASURED - paste these into the manuscript")

    try:
        import config
        print("\nFrom config.py:")
        for k in ("CONTEXT_LEN", "LEAD_DAYS", "LR", "EPOCHS", "HIDDEN_DIM",
                  "KERNEL_SIZE", "NUM_LAYERS", "SEEDS", "PROC_ROOT"):
            v = getattr(config, k, "NOT DEFINED")
            if k == "SEEDS" and isinstance(v, list):
                print(f"  {k:14s} = {len(v)} seeds: {v}")
            else:
                print(f"  {k:14s} = {v}")
        proc = Path(config.PROC_ROOT)
    except Exception as exc:
        print(f"  could not import config.py: {exc}")
        proc = Path("data/processed")

    # grid, period, splits
    try:
        ms = xr.open_zarr(proc / "master_dataset.zarr", consolidated=True)
        t = ms["time"].values
        lat, lon = ms["lat"].values, ms["lon"].values
        print("\nGrid and period (master_dataset.zarr):")
        print(f"  grid            {len(lat)} x {len(lon)}")
        print(f"  lat             {lat.min():.4f} to {lat.max():.4f}  "
              f"(step {abs(np.diff(lat).mean()):.5f} deg)")
        print(f"  lon             {lon.min():.4f} to {lon.max():.4f}  "
              f"(step {abs(np.diff(lon).mean()):.5f} deg)")
        print(f"  time            {str(t[0])[:10]} to {str(t[-1])[:10]}  "
              f"({len(t)} steps)")
        print(f"  variables       {sorted(ms.data_vars)}")
    except Exception as exc:
        print(f"  could not open master_dataset.zarr: {exc}")
        ms = None

    # domain composition
    if ms is not None:
        try:
            ai = ms["aridity_index"].values
            wlc = ms["water_limited_crop"].values > 0.5
            crop = ms["crop_mask"].values > 0.5
            irr = ms["irr_frac"].values
            if np.nanmax(irr) > 1.5:
                irr = irr / 100.0
            print("\nDomain composition:")
            print(f"  crop_mask cells            {int(crop.sum())}")
            print(f"  water_limited_crop cells   {int(wlc.sum())}")
            for nm, lo, hi in (("arid", 0.05, 0.20), ("semi_arid", 0.20, 0.50),
                               ("dry_subhumid", 0.50, 0.65)):
                n = int((wlc & np.isfinite(ai) & (ai >= lo) & (ai < hi)).sum())
                print(f"    of which {nm:14s} {n}")
            print(f"  irrigated  (>= 0.15)       {int((wlc & (irr >= 0.15)).sum())}")
            print(f"  rainfed    (<= 0.05)       {int((wlc & (irr <= 0.05)).sum())}")
            print(f"  unassigned (0.05-0.15)     "
                  f"{int((wlc & (irr > 0.05) & (irr < 0.15)).sum())}")
            print(f"  irr_frac range on domain   {np.nanmin(irr[wlc]):.3f} to "
                  f"{np.nanmax(irr[wlc]):.3f}")
        except Exception as exc:
            print(f"  domain composition failed: {exc}")

    # sample counts per split, recomputed the way the Dataset does it
    try:
        import config as cfg
        import pandas as pd
        an = xr.open_zarr(proc / "anomalies.zarr", consolidated=True)
        times = pd.DatetimeIndex(an["time"].values)
        CTX, LEADS, STRIDE = cfg.CONTEXT_LEN, cfg.LEAD_DAYS, 8
        TRAIN_END = pd.Timestamp("2021-12-31")
        VAL_END = pd.Timestamp("2022-12-31")
        all_t0 = np.arange(CTX, len(times) - max(LEADS), STRIDE)
        tr = all_t0[times[all_t0] <= TRAIN_END]
        va = all_t0[(times[all_t0] > TRAIN_END) & (times[all_t0] <= VAL_END)]
        te = all_t0[times[all_t0] > VAL_END]
        print("\nSample counts (stride 8):")
        print(f"  train  {len(tr):4d}   {times[tr[0]].date()} to {times[tr[-1]].date()}")
        print(f"  val    {len(va):4d}   {times[va[0]].date()} to {times[va[-1]].date()}")
        print(f"  test   {len(te):4d}   {times[te[0]].date()} to {times[te[-1]].date()}")
        print(f"  context {CTX} steps, leads {LEADS}")
    except Exception as exc:
        print(f"  sample counts failed: {exc}")

    # correction statistics over the full record
    try:
        o = xr.open_zarr(proc / "anomalies.zarr", consolidated=True)["rzsm_anom"]
        c = xr.open_zarr(proc / "anomalies_corrected.zarr",
                         consolidated=True)["rzsm_anom_rainfed"]
        irr = ms["irr_frac"].values
        if np.nanmax(irr) > 1.5:
            irr = irr / 100.0
        m = irr >= 0.15
        step = 8
        d = []
        for i in range(0, o.sizes["time"], step):
            x = (c.isel(time=i).values - o.isel(time=i).values)[m]
            d.append(x[np.isfinite(x)])
        d = np.concatenate(d)
        print("\nIrrigation correction magnitude (every 8th step, irr_frac >= 0.15):")
        print(f"  cells corrected            {int(m.sum())}")
        print(f"  mean correction            {d.mean():+.4f} sigma")
        print(f"  mean |correction|          {np.abs(d).mean():.4f} sigma")
        print(f"  median |correction|        {np.median(np.abs(d)):.4f} sigma")
        print(f"  95th pct |correction|      {np.percentile(np.abs(d), 95):.4f} sigma")
        print(f"  fraction exactly zero      {np.mean(np.abs(d) < 1e-9):.3f}")
        print(f"  one-sided (all <= 0)?      {bool((d <= 1e-9).all())}")
    except Exception as exc:
        print(f"  correction statistics failed: {exc}")

    if args.measured:
        return

    # ---------------------------------------------------------------- inspect
    rule("INSPECT - read these source lines and paraphrase")

    todo = [
        ("CDL cropland threshold and year", "04a_cdl_mask.py",
         [r"thresh", r"CDL", r">\s*0\.", r">=\s*0\.", r"year", r"20\d\d"]),
        ("SMAP daily aggregation rule", "02_read_smap.py",
         [r"2230|22:30|utc", r"resample|sel\(|nearest|mean\(", r"daily"]),
        ("NIRv formulation, QA, 8-day to daily", "03c_nirv_assemble.py",
         [r"nirv|ndvi|nir\b", r"qa|quality|cloud", r"interp|resample|reindex|ffill"]),
        ("LANID years and aggregation", "04b_irrigation_fraction_V3.py",
         [r"lanid", r"20\d\d", r"mean|sum|fraction|coarsen|resample"]),
        ("Anomaly window and climatology", "06_compute_anomalies.py",
         [r"doy|day_of_year", r"window|\+/-|halfwidth|delta", r"clim", r"std|mean"]),
        ("Irrigation correction formulation", "07_irrigation_deconvolution.py",
         [r"def |excess|regional|rainfed|reference|quantile|percentile",
          r"0\.15|thresh"]),
    ]
    for title, path, pats in todo:
        rule(f"{title}   ({path})")
        show(path, pats)

    rule("NEXT")
    print("""
  Paste the MEASURED block into the methods directly.
  For each INSPECT block, write one or two sentences describing what the code
  does - not what it is called. "A regional excess correction was applied" tells
  a reader nothing; "the irrigation-attributable component was estimated as the
  difference between each irrigated cell and the median of rainfed cells within
  X km, and subtracted where positive" is reproducible.
""")


if __name__ == "__main__":
    main()
