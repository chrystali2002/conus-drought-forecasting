#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_post_figures.py
====================
Generate every figure for the blog post from the current data, in one pass, so
none of them can silently be a stale copy from an earlier run.

Produces
--------
  domain_map.png     the study domain shaded by irrigated fraction
  skill_vs_lead.png  forecast skill against lead time, with the seed spread
  dacc_by_mask.png   change in skill by pixel group - the null result
  dose_response.png  change in skill against irrigated fraction - the gradient

Sources
-------
  data/processed/master_dataset.zarr            irr_frac, water_limited_crop
  outputs/metrics/corrected/multiseed_stratified.csv
  outputs/metrics/uncorrected/multiseed_stratified.csv

All skill numbers are recomputed here from the stratified CSVs rather than
copied, and the script prints every value it plots so the post text can be
checked against it.

USAGE
    cd /mnt/scratch/olusegu3/drought/conus_drought
   python make_post_figures.py --outdir /mnt/scratch/olusegu3/personal-site/blog/posts/irrigation-drought-forecasting

    python make_post_figures.py --outdir out --skip-map    # if the zarr is elsewhere
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from scipy import stats as st
except ImportError:
    st = None

INK, MSU, GOLD = "#0F241F", "#18453B", "#B8860B"
GREY, MID, PALE = "#8C9A96", "#3E7C6A", "#EAF2EE"
DPI = 200


# ---------------------------------------------------------------- helpers
def paired(df, mask, lead, treat="corrected", ref="uncorrected"):
    """Mean and 95% t interval of the PER-SEED difference.

    Paired, not a difference of independent means: the two arms share seeds, and
    seed-to-seed variation is the same order as the effect, so discarding the
    pairing would bury it.
    """
    sub = df[(df["mask"] == mask) & (df["lead"] == lead)]
    a = sub[sub.arm == treat].set_index("seed")["acc"]
    b = sub[sub.arm == ref].set_index("seed")["acc"]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return np.nan, np.nan, np.nan, 0
    d = (a.loc[common] - b.loc[common]).values
    m = float(d.mean())
    if len(d) < 2 or st is None:
        return m, np.nan, np.nan, len(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    h = st.t.ppf(0.975, len(d) - 1) * se
    return m, m - h, m + h, len(d)


def load_metrics(root):
    frames = []
    for arm in ("corrected", "uncorrected"):
        p = os.path.join(root, arm, "multiseed_stratified.csv")
        if not os.path.exists(p):
            sys.exit(f"not found: {p}\nRun 16_evaluate_multiseed.py for both arms first.")
        d = pd.read_csv(p)
        if "arm" not in d.columns:
            d["arm"] = arm
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"  {len(df)} rows | {df.seed.nunique()} seeds | "
          f"leads {sorted(df.lead.unique())}")
    print(f"  masks: {sorted(df['mask'].unique())}")
    return df


def finish(fig, ax_or_axes, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- figure 1
def fig_domain_map(zarr_path, out):
    """Domain classed by irrigated fraction, on a state-border basemap.

    Two decisions worth recording. First, the classes are the evaluation's own
    strata (< 0.15, 0.15-0.25, 0.25-0.50, > 0.50) rather than a continuous ramp:
    a linear 0-1 colour scale spends most of its range on values the data never
    takes, and renders the 2,539 sub-0.15 cells as pale noise when they are in
    fact the control group. Classing the map this way also makes it legible
    against dose_response.png, which splits on exactly these bins.

    Second, state borders. Without them the reader cannot name a single feature,
    so "the Snake River Plain is barely in this domain" is unreadable - and that
    absence is a real finding, not a rendering artifact: the Columbia Basin
    contributes 10 cells, none of them majority-irrigated.
    """
    try:
        import xarray as xr
    except ImportError:
        print("  xarray not available; skipping domain map")
        return
    if not os.path.exists(zarr_path):
        print(f"  {zarr_path} not found; skipping domain map")
        return
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        ccrs = cfeature = None
        print("  cartopy not available; drawing without the basemap")

    ms = xr.open_zarr(zarr_path, consolidated=True)
    lat, lon = ms["lat"].values, ms["lon"].values
    wlc = ms["water_limited_crop"].values > 0.5
    # Match the evaluated set: 15 cells over the Great Salt Lake, Chesapeake Bay
    # and a Texas lagoon are in the cropland mask but have no SMAP data at any
    # timestep, and are excluded from every skill figure. Showing them here
    # would make the map disagree with the results.
    try:
        an = xr.open_zarr(zarr_path.replace("master_dataset", "anomalies"),
                          consolidated=True)["rzsm_anom"]
        valid = np.isfinite(an.isel(time=slice(0, 60)).values).all(axis=0)
        dropped = int((wlc & ~valid).sum())
        wlc = wlc & valid
        print(f"    excluded {dropped} cells with no SMAP data -> {int(wlc.sum())}")
    except Exception as exc:
        print(f"    could not apply SMAP mask ({exc}); showing all domain cells")
    irr = ms["irr_frac"].values
    if np.nanmax(irr) > 1.5:
        irr = irr / 100.0
    yy, xx = np.where(wlc)
    plon, plat, pval = lon[xx], lat[yy], irr[yy, xx]

    # lo, hi, colour, marker size, alpha, label. The split at 0.05 is not
    # cosmetic: the rainfed control mask is irr <= 0.05 (n = 2,164), so the
    # 360 cells in 0.05-0.15 belong to no evaluation stratum at all. Labelling
    # the whole sub-0.15 population "rainfed control" would overstate the
    # control group by a sixth.
    CLASSES = [
        (0.00, 0.0500001, "#CBD5D1",  3.0, 0.70, "\u2264 0.05  rainfed control"),
        (0.0500001, 0.15, "#A9BDB6",  5.0, 0.85, "0.05 - 0.15  in neither stratum"),
        (0.15, 0.25, "#7FB3A0",  9.0, 0.95, "0.15 - 0.25  irrigated"),
        (0.25, 0.50, "#D99A2B", 16.0, 1.00, "0.25 - 0.50  irrigated"),
        (0.50, 1.01, "#8C3B00", 30.0, 1.00, "> 0.50  majority irrigated"),
    ]
    # label x/y, then the point the leader touches, then the count to report
    REGIONS = [
        ("Columbia Basin",           -121.3, 48.6, -119.4, 46.9),
        ("Snake River Plain",        -115.2, 47.9, -113.8, 43.3),
        ("Central Valley & Salinas", -119.6, 33.4, -121.4, 36.5),
        ("High Plains Aquifer",       -88.5, 41.8,  -98.6, 41.2),
    ]
    BOXES = {"Columbia Basin":           (-121, -117, 45.0, 48.5),
             "Snake River Plain":        (-117, -111, 42.0, 45.0),
             "Central Valley & Salinas": (-123, -118, 34.5, 41.0),
             "High Plains Aquifer":      (-104,  -96, 31.0, 44.0)}

    fig = plt.figure(figsize=(9.4, 5.6))
    if ccrs is not None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-124.5, -74.0, 26.0, 49.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), fc="#FAFAF8", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), fc="#EFF4F6", zorder=0)
        ax.add_feature(cfeature.STATES.with_scale("50m"), ec="#C8CFCC", lw=0.55, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), ec="#9AA5A1", lw=0.7, zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), ec="#9AA5A1", lw=0.7, zorder=1)
        tf = {"transform": ccrs.PlateCarree()}
    else:
        ax = plt.axes()
        ax.set_xlim(-124.5, -74.0); ax.set_ylim(26.0, 49.5)
        ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
        tf = {}

    handles = []
    for lo, hi, colour, size, alpha, label in CLASSES:
        m = (pval >= lo) & (pval < hi)
        ax.scatter(plon[m], plat[m], s=size, c=colour, marker="s", linewidths=0,
                   alpha=alpha, zorder=3 if lo >= 0.15 else 2, **tf)
        handles.append(Line2D([], [], ls="", marker="s", ms=np.sqrt(size) + 1.5,
                              mfc=colour, mec="none",
                              label=f"{label}   (n = {int(m.sum()):,})"))

    # Annotate each cluster with what the domain actually contains there, so a
    # reader looking for the Columbia Basin learns why they cannot find it.
    for name, tx, ty, px, py in REGIONS:
        x0, x1, y0, y1 = BOXES[name]
        n = int(((plon >= x0) & (plon <= x1) & (plat >= y0) & (plat <= y1)
                 & (pval >= 0.15)).sum())
        ax.annotate(f"{name}\n{n} cells \u2265 0.15", xy=(px, py), xytext=(tx, ty),
                    fontsize=7.8, color=INK, ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#D5DCD9", lw=0.6, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color="#95A29D", lw=0.8,
                                    shrinkA=2, shrinkB=2), **tf)

    n = int(wlc.sum())
    n_irr = int((wlc & (irr >= 0.15)).sum())
    ax.set_title(f"Only {n_irr} of {n:,} water-limited cropland cells "
                 f"are more than 15 % irrigated",
                 loc="left", color=MSU, fontweight="bold", fontsize=12.5, pad=9)
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8.2, frameon=True,
                    framealpha=0.94, edgecolor="#D5DCD9", borderpad=0.7,
                    labelspacing=0.7, title="Irrigated fraction of cell (LANID)")
    leg.get_title().set_fontsize(8.4); leg.get_title().set_color(INK)
    ax.text(0.995, 0.975,
            "Marker size and colour follow the evaluation's own strata.\n"
            "Only the 2,164 cells at or below 0.05 are the rainfed control.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
            color=GREY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.8))
    if ccrs is not None:
        gl = ax.gridlines(draw_labels=True, lw=0.4, color="#E3E8E6", alpha=0.8)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7.5, "color": "#5A6B66"}
    else:
        ax.grid(alpha=0.15, lw=0.5)
    finish(fig, ax, os.path.join(out, "domain_map.png"))

    print(f"    domain {n} cells | >=0.15 irrigated {n_irr} | "
          f">=0.50 {int((wlc & (irr >= 0.50)).sum())}")
    for lo, hi, _, _, _, label in CLASSES:
        print(f"      {label:28s} {int(((pval >= lo) & (pval < hi)).sum()):5d}")


# ---------------------------------------------------------------- figure 2
def fig_skill_vs_lead(df, out):
    """Skill against lead, corrected arm, with the across-seed spread."""
    leads = sorted(df["lead"].unique())
    sub = df[(df["mask"] == "all_wlc") & (df.arm == "corrected")]
    mean = [sub[sub.lead == L]["acc"].mean() for L in leads]
    sd = [sub[sub.lead == L]["acc"].std(ddof=1) for L in leads]
    lo = [m - s for m, s in zip(mean, sd)]
    hi = [m + s for m, s in zip(mean, sd)]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(leads, mean, "-o", color=MSU, lw=2.2, ms=7, label="Full model")
    ax.fill_between(leads, lo, hi, color=MSU, alpha=0.15, lw=0,
                    label=f"±1 SD across {sub.seed.nunique()} seeds")
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("Anomaly correlation coefficient (ACC)")
    ax.set_xticks(leads)
    ax.set_ylim(0, max(hi) * 1.15)
    ax.set_title("Vegetation drought forecast skill declines with lead time",
                 loc="left", color=MSU, fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for L, m in zip(leads, mean):
        ax.annotate(f"{m:.3f}", (L, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=INK)
    finish(fig, ax, os.path.join(out, "skill_vs_lead.png"))

    print("    ACC by lead (corrected, whole domain):")
    for L, m, s in zip(leads, mean, sd):
        print(f"      {L:>2}d  {m:.4f} +/- {s:.4f}")
    print(f"      mean across leads {np.mean(mean):.4f}")


# ---------------------------------------------------------------- figure 3
def fig_dacc_by_mask(df, out):
    """The null: irrigated and rainfed differences are the same size."""
    leads = sorted(df["lead"].unique())
    series = [("rainfed", "Rainfed — control, identical input", GREY, "--", "s"),
              ("irrigated", "Irrigated — where the correction acts", GOLD, "-", "o"),
              ("all_wlc", "Whole domain", MSU, "-", "^")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mask, label, colour, ls, mk in series:
        if mask not in df["mask"].values:
            continue
        m, lo, hi = [], [], []
        for L in leads:
            a, b, c, _ = paired(df, mask, L)
            m.append(a); lo.append(b); hi.append(c)
        ax.plot(leads, m, ls, color=colour, marker=mk, lw=2.0, ms=6, label=label)
        ax.fill_between(leads, lo, hi, color=colour, alpha=0.12, lw=0)
        print(f"    {mask:10s} " +
              "  ".join(f"{L}d {v:+.4f}" for L, v in zip(leads, m)))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("Change in skill, ΔACC\n(corrected − uncorrected)")
    ax.set_xticks(leads)
    ax.set_title("Irrigation correction does not change skill overall",
                 loc="left", color=MSU, fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.03,
            "Shading: 95 % CI across seeds. Rainfed cells receive identical input\n"
            "in both runs, so their spread is the empirical noise floor.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=GREY)
    finish(fig, ax, os.path.join(out, "dacc_by_mask.png"))


# ---------------------------------------------------------------- figure 4
def fig_dose_response(df, out):
    """The gradient: effect grows with irrigated fraction and with lead."""
    bins = [("irr_015_025", 0.20, "0.15–0.25"),
            ("irr_025_050", 0.375, "0.25–0.50"),
            ("irr_050_plus", 0.70, "> 0.50")]
    bins = [b for b in bins if b[0] in df["mask"].values]
    if not bins:
        print("  no irrigation-fraction bins present; skipping dose response")
        return
    leads = sorted(df["lead"].unique())
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(leads)))
    x = [b[1] for b in bins]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for L, colour in zip(leads, cols):
        m, lo, hi = [], [], []
        for mask, _, _ in bins:
            a, b, c, _ = paired(df, mask, L)
            m.append(a); lo.append(b); hi.append(c)
        ax.errorbar(x, m, yerr=[np.array(m) - np.array(lo),
                                np.array(hi) - np.array(m)],
                    fmt="o-", color=colour, capsize=3, lw=1.9, ms=6,
                    label=f"{L}-day lead")
        print(f"    {L:>2}d lead  " +
              "  ".join(f"{b[2]} {v:+.4f}" for b, v in zip(bins, m)))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlabel("Irrigated fraction of the cell")
    ax.set_ylabel("Change in skill, ΔACC\n(corrected − uncorrected)")
    ax.set_title("The correction helps where irrigation dominates the cell",
                 loc="left", color=MSU, fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    y0 = ax.get_ylim()[0]
    for mask, xi, _ in bins:
        n = int(df[df["mask"] == mask]["n_pixels"].iloc[0])
        ax.annotate(f"n = {n}", (xi, y0), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color=GREY)
    finish(fig, ax, os.path.join(out, "dose_response.png"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/metrics")
    ap.add_argument("--zarr", default="data/processed/master_dataset.zarr")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading metrics")
    df = load_metrics(args.root)

    print("\nFigure 1: domain map")
    if args.skip_map:
        print("  skipped")
    else:
        fig_domain_map(args.zarr, args.outdir)

    print("\nFigure 2: skill vs lead")
    fig_skill_vs_lead(df, args.outdir)

    print("\nFigure 3: change in skill by pixel group")
    fig_dacc_by_mask(df, args.outdir)

    print("\nFigure 4: dose response on irrigated fraction")
    fig_dose_response(df, args.outdir)

    print("\nAll figures regenerated from current data. Every number printed "
          "above should match the post text; if it does not, the text is stale.")


if __name__ == "__main__":
    main()
