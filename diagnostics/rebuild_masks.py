#!/usr/bin/env python3
"""
rebuild_masks.py
================
Rebuild reporting-zone and aridity masks on the current (321, 656) grid.

The existing mask_*.npy files are (278, 656) - built before the grid was
extended northward, a 43-row difference. Rather than guess an offset, rebuild
from coordinates and from aridity_index, both already correct in
master_dataset.zarr.

Zone boundaries are IMPORTED from config.REPORTING_ZONES rather than retyped, so
they cannot drift from the definition the rest of the pipeline uses. Note that
config stores them as (lon_min, lat_min, lon_max, lat_max).

As config.py states, these zones are for post-masking disaggregation only: they
do not replace the AI + CDL domain definition. Every mask written here is
therefore reported intersected with water_limited_crop as well as standalone.

Writes mask_<name>_v2.npy, leaving the stale files untouched for comparison.
"""
import numpy as np
import xarray as xr
from pathlib import Path

try:
    from config import PROC_ROOT, REPORTING_ZONES
except ImportError as exc:
    raise SystemExit(f"could not import from config.py: {exc}")

OUT = Path(PROC_ROOT)
ms = xr.open_zarr(OUT / "master_dataset.zarr", consolidated=True)
lat = ms["lat"].values
lon = ms["lon"].values
ai = ms["aridity_index"].values
wlc = ms["water_limited_crop"].values > 0.5
H, W = wlc.shape
print(f"grid {H} x {W}   lat {lat.min():.2f} to {lat.max():.2f}   "
      f"lon {lon.min():.2f} to {lon.max():.2f}")
print(f"water_limited_crop: {int(wlc.sum())} px\n")

LA = lat[:, None] * np.ones((1, W))
LO = np.ones((H, 1)) * lon[None, :]

# ---------------------------------------------------------------- zones
print("REPORTING ZONES (from config.REPORTING_ZONES)")
print(f"{'zone':26s} {'zone px':>9} {'WLC in zone':>12} {'irrigated':>10}")
irr = ms["irr_frac"].values
if np.nanmax(irr) > 1.5:
    irr = irr / 100.0

for name, (w, s, e, n) in REPORTING_ZONES.items():
    m = (LA >= s) & (LA <= n) & (LO >= w) & (LO <= e)
    np.save(OUT / f"mask_{name}_v2.npy", m.astype(np.uint8))
    in_dom = m & wlc
    print(f"  {name:24s} {int(m.sum()):9d} {int(in_dom.sum()):12d} "
          f"{int((in_dom & (irr >= 0.15)).sum()):10d}")

# ---------------------------------------------------------------- aridity
# UNEP/IPCC classes. The study domain is AI 0.05-0.65, i.e. arid through
# dry-subhumid, which is why hyper_arid and humid fall outside it.
ARIDITY = {
    "hyper_arid":   (0.00, 0.05),
    "arid":         (0.05, 0.20),
    "semi_arid":    (0.20, 0.50),
    "dry_subhumid": (0.50, 0.65),
    "humid":        (0.65, 99.0),
}
print("\nARIDITY CLASSES (from aridity_index)")
print(f"{'class':26s} {'class px':>9} {'WLC in class':>12}")
for name, (lo_, hi_) in ARIDITY.items():
    m = np.isfinite(ai) & (ai >= lo_) & (ai < hi_)
    np.save(OUT / f"mask_{name}_v2.npy", m.astype(np.uint8))
    print(f"  {name:24s} {int(m.sum()):9d} {int((m & wlc).sum()):12d}")

# ---------------------------------------------------------------- checks
print("\nCONSISTENCY CHECKS")
dom_classes = np.isfinite(ai) & (ai >= 0.05) & (ai < 0.65)
overlap = int((dom_classes & wlc).sum())
print(f"  WLC pixels inside AI 0.05-0.65: {overlap} of {int(wlc.sum())}")
if overlap != int(wlc.sum()):
    print("    NOTE: some WLC pixels fall outside the stated AI range. Check")
    print("    whether the domain used different class boundaries.")

print("\nOLD vs NEW (old masks are on the stale 278-row grid)")
for name in list(REPORTING_ZONES) + list(ARIDITY):
    old_p = OUT / f"mask_{name}.npy"
    if not old_p.exists():
        continue
    old = np.load(old_p) > 0.5
    new = np.load(OUT / f"mask_{name}_v2.npy") > 0.5
    flag = "" if abs(int(old.sum()) - int(new.sum())) < 0.2 * max(int(new.sum()), 1) \
           else "   <- large change, check the bounds"
    print(f"  {name:24s} old {int(old.sum()):7d} {str(old.shape):12s} "
          f"new {int(new.sum()):7d}{flag}")

print("\nWrote mask_<name>_v2.npy for every zone and class.")
print("If you adopt these, update the filenames wherever the old masks are read")
print("(16_evaluate_multiseed.py loads mask_Central_Valley_CA.npy by name).")
