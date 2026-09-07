#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperarid_test.py
=================
Test whether the aridity index has been diluted by water during regridding.

THE HYPOTHESIS
    Global-AI_ET0 v3.1 arrives at 30 arcsec. Aggregating to 0.09 deg averages
    ~117 source pixels per cell. If nodata (water) was filled with zero before
    averaging rather than excluded, then

        AI_ours ~= AI_true * land_fraction

    Every affected value moves DOWNWARD, and the more water in the footprint,
    the further down. Near a humid coast this pushes cells below the 0.65
    dryland ceiling and falsely INTO the study domain; near an arid coast or
    lake it pushes them below the 0.05 floor and falsely OUT of it.

WHY HYPER-ARID IS THE SHARPEST TEST
    CONUS has essentially no genuinely hyper-arid land. Death Valley sits around
    0.03-0.05 and is marginal; almost nothing else comes close. So a large
    population of cells at AI < 0.05 is already suspicious. If those cells sit
    disproportionately adjacent to water, dilution is close to established - and
    unlike a domain-wide regression of AI on distance, this needs no confound
    control, because there is no legitimate reason for true hyper-aridity to
    cluster at coastlines.

WHAT THIS CAN AND CANNOT SHOW
    It can establish the pattern and size the exposure. It CANNOT confirm the
    mechanism, because sub-cell land fraction is not recoverable from the zarr:
    a cell is only flagged nodata when it is essentially all water, so a cell
    that is 60 % water at source resolution carries no marker here. Only reading
    the source raster footprint - counting valid pixels per target cell - can
    confirm dilution. Treat a positive result here as grounds for that check,
    not a substitute for it.

USAGE
    cd /mnt/scratch/olusegu3/drought/conus_drought
    python hyperarid_test.py
    python hyperarid_test.py --max-dist 8 --plot
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import xarray as xr
except ImportError:
    sys.exit("needs xarray:  pip install xarray zarr")

try:
    from scipy import ndimage
except ImportError:
    sys.exit("needs scipy:  pip install scipy")

try:
    from config import PROC_ROOT
except ImportError:
    PROC_ROOT = "data/processed"

AI_MIN, AI_MAX = 0.05, 0.65      # dryland_mask thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=int, default=8,
                    help="tabulate out to this distance in grid cells")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ms = xr.open_zarr(f"{PROC_ROOT}/master_dataset.zarr", consolidated=True)
    ai = ms["aridity_index"].values
    acl = ms["aridity_class"].values
    crop = ms["crop_mask"].values > 0.5
    wlc = ms["water_limited_crop"].values > 0.5
    lat, lon = ms["lat"].values, ms["lon"].values
    H, W = ai.shape

    # aridity_class == -1 marks nodata / water in the source raster
    water = acl == -1
    land = ~water
    print(f"grid {H} x {W} = {H*W:,} cells")
    print(f"  nodata/water (aridity_class == -1)  {int(water.sum()):,}")
    print(f"  land                                {int(land.sum()):,}")

    # distance in CELLS to the nearest water cell. Cell units, not km: the
    # footprint is what matters, and 0.09 deg is ~10 km in latitude but ~7.9 km
    # in longitude at 38N, so a metric distance would mix the two.
    dist = ndimage.distance_transform_cdt(land, metric="chessboard")
    print(f"  max distance to water {int(dist[land].max())} cells")

    # ---------------------------------------------------------------- test 1
    print("\n" + "=" * 72)
    print("TEST 1  where do the hyper-arid cells sit?")
    print("=" * 72)
    hyper = land & np.isfinite(ai) & (ai < AI_MIN)
    n_hyper = int(hyper.sum())
    print(f"cells with AI < {AI_MIN} on land: {n_hyper:,}")
    print("CONUS has essentially no genuinely hyper-arid land, so a large "
          "population here is already a warning.\n")

    print(f"{'dist':>5} {'land cells':>12} {'hyper-arid':>11} {'% hyper':>9} "
          f"{'enrichment':>11}")
    print("-" * 54)
    base = n_hyper / max(int(land.sum()), 1)
    rows = []
    for d in range(1, args.max_dist + 1):
        at = land & (dist == d)
        n = int(at.sum())
        h = int((at & hyper).sum())
        pct = h / max(n, 1)
        rows.append((d, n, h, pct, pct / base if base else np.nan))
        print(f"{d:>5} {n:>12,} {h:>11,} {pct*100:>8.2f}% "
              f"{pct/base if base else float('nan'):>10.2f}x")
    far = land & (dist > args.max_dist)
    n_far = int(far.sum()); h_far = int((far & hyper).sum())
    print(f"{'>'+str(args.max_dist):>5} {n_far:>12,} {h_far:>11,} "
          f"{h_far/max(n_far,1)*100:>8.2f}% "
          f"{(h_far/max(n_far,1))/base if base else float('nan'):>10.2f}x")

    at1 = rows[0][3]
    print(f"\n  share of ALL hyper-arid cells within 1 cell of water: "
          f"{rows[0][2]/max(n_hyper,1)*100:.1f}%")
    print(f"  share of ALL hyper-arid cells within 2 cells: "
          f"{sum(r[2] for r in rows[:2])/max(n_hyper,1)*100:.1f}%")
    if base and at1 / base > 3:
        print("\n  VERDICT: hyper-arid cells are strongly concentrated next to")
        print("  water. There is no legitimate reason for true hyper-aridity to")
        print("  cluster at coastlines. This is consistent with nodata-as-zero")
        print("  dilution during regridding.")
    elif base and at1 / base > 1.5:
        print("\n  VERDICT: moderate enrichment near water. Suggestive but not")
        print("  decisive on its own; the threshold-asymmetry test below carries")
        print("  more weight.")
    else:
        print("\n  VERDICT: no enrichment near water. Dilution is NOT supported")
        print("  at this scale, and the eastern cells need another explanation.")

    # ---------------------------------------------------------------- test 2
    print("\n" + "=" * 72)
    print("TEST 2  threshold asymmetry - the mechanism discriminator")
    print("=" * 72)
    print("Dilution moves AI DOWNWARD only. So near-water cells whose")
    print("neighbourhood is humid should be pushed INTO the dryland band, while")
    print("near-water cells in an arid neighbourhood should be pushed OUT of it,")
    print("below the floor. A registration error relocates values and produces")
    print("no such threshold-dependent sign flip.\n")

    # local surround: median AI over a ring 3-8 cells away, land only, excluding
    # other near-water cells so the reference is not itself contaminated
    clean = land & (dist >= 3) & np.isfinite(ai)
    ai0 = np.where(clean, ai, np.nan)
    from scipy.ndimage import uniform_filter
    valid = clean.astype(np.float32)
    filled = np.where(clean, ai, 0.0).astype(np.float32)
    K = 17           # ~8-cell radius
    s = uniform_filter(filled, size=K, mode="constant", cval=0.0)
    c = uniform_filter(valid, size=K, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        surround = np.where(c > 0.02, s / np.maximum(c, 1e-9), np.nan)

    near = land & (dist <= 2) & np.isfinite(ai) & np.isfinite(surround)
    humid_sur = near & (surround > AI_MAX)
    arid_sur = near & (surround <= AI_MAX) & (surround > AI_MIN)

    for label, sel in (("humid surround (AI_sur > 0.65)", humid_sur),
                       ("arid  surround (0.05 < AI_sur <= 0.65)", arid_sur)):
        n = int(sel.sum())
        if n == 0:
            print(f"  {label}: no cells")
            continue
        anom = np.nanmedian(ai[sel] - surround[sel])
        into = int((sel & (ai >= AI_MIN) & (ai <= AI_MAX)).sum())
        below = int((sel & (ai < AI_MIN)).sum())
        print(f"  {label}")
        print(f"    near-water cells        {n:,}")
        print(f"    median AI - surround    {anom:+.3f}")
        print(f"    now inside 0.05-0.65    {into:,} ({into/n*100:.1f}%)")
        print(f"    now below 0.05          {below:,} ({below/n*100:.1f}%)")

    # ---------------------------------------------------------------- test 3
    print("\n" + "=" * 72)
    print("TEST 3  exposure of the study domain")
    print("=" * 72)
    for name, m in (("crop_mask", crop), ("water_limited_crop", wlc)):
        n = int(m.sum())
        print(f"\n{name}: {n:,} cells")
        for d in (1, 2, 3):
            k = int((m & (dist <= d)).sum())
            print(f"  within {d} cell(s) of water: {k:,} ({k/max(n,1)*100:.1f}%)")
        sel = m & (dist <= 2) & np.isfinite(surround) & (surround > AI_MAX)
        print(f"  within 2 cells AND humid surround: {int(sel.sum()):,}"
              f"   <- candidates for false inclusion")
        if int(sel.sum()):
            yy, xx = np.where(sel)
            print(f"    lat {lat[yy].min():.2f} to {lat[yy].max():.2f}, "
                  f"lon {lon[xx].min():.2f} to {lon[xx].max():.2f}")
            print(f"    AI range {np.nanmin(ai[sel]):.3f} to {np.nanmax(ai[sel]):.3f}")

    print("\n" + "=" * 72)
    print("NEXT")
    print("=" * 72)
    print("""
  If tests 1 and 2 both point to dilution, confirm the mechanism by reading the
  source raster footprint - this file cannot do it, because sub-cell land
  fraction is not recoverable here:

    for each affected 0.09 deg cell, open data/raw/aridity/ai_v31_yr.tif,
    read the full footprint window, and record (a) the centre value, (b) the
    mean over VALID pixels only, (c) the count of nodata pixels. Check the
    scale factor first - v3.1 ships AI as integers x1e-4 with nodata -9999.

    Then plot AI_ours / AI_rawmean_validonly against valid_fraction.
    Slope ~1 through the origin confirms nodata-as-zero in our regridding.

  A useful sanity check on any reconstruction: AI_ours / land_fraction should
  recover a PHYSICALLY PLAUSIBLE value (~0.9-1.1 for the Virginia Piedmont).
  Proportionality alone is weaker evidence than proportionality that recovers
  the right number.
""")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.4))
        d = [r[0] for r in rows]; e = [r[4] for r in rows]
        ax.bar(d, e, color="#18453B")
        ax.axhline(1, color="#A33A1E", ls="--", lw=1.2,
                   label="no enrichment")
        ax.set_xlabel("Distance to nearest nodata/water cell (grid cells)")
        ax.set_ylabel("Hyper-arid enrichment\n(x domain baseline)")
        ax.set_title("Do hyper-arid cells cluster at water margins?",
                     loc="left", color="#18453B", fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("hyperarid_enrichment.png", dpi=180, facecolor="white")
        print("wrote hyperarid_enrichment.png")


if __name__ == "__main__":
    main()
