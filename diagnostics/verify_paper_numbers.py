#!/usr/bin/env python3
"""Verify the two numbers quoted in the manuscript against the actual data.

1. Pixel counts: irrigated + rainfed = 2733, not the 3093 evaluation domain.
   The gap is pixels with 0.05 < irr_frac < 0.15, which belong to neither mask.
   Confirm the arithmetic so the paper's numbers reconcile.

2. Mean correction magnitude: the 0.035 sigma figure came from a single date
   (2020-07-01). A number cited as general must be computed over the full record.
"""
import numpy as np, xarray as xr

ms = xr.open_zarr("data/processed/master_dataset.zarr", consolidated=True)
irr = ms["irr_frac"].values
if np.nanmax(irr) > 1.5:
    irr = irr / 100.0
wlc = ms["water_limited_crop"].values > 0.5

print("=" * 66)
print("1. PIXEL COUNTS WITHIN THE EVALUATION DOMAIN")
print("=" * 66)
tot = int(wlc.sum())
hi  = int((wlc & (irr >= 0.15)).sum())
lo  = int((wlc & (irr <= 0.05)).sum())
mid = int((wlc & (irr > 0.05) & (irr < 0.15)).sum())
print(f"  evaluation domain (water_limited_crop)   {tot:5d}")
print(f"  irrigated  (irr_frac >= 0.15)            {hi:5d}")
print(f"  rainfed    (irr_frac <= 0.05)            {lo:5d}")
print(f"  unassigned (0.05 < irr_frac < 0.15)      {mid:5d}")
print(f"  check: {hi} + {lo} + {mid} = {hi+lo+mid}  (should equal {tot})")
print(f"\n  FOOTNOTE TEXT:")
print(f"  \"Of the {tot} water-limited cropland pixels, {hi} have irrigated")
print(f"   fraction >= 0.15 and {lo} <= 0.05; the remaining {mid} fall between")
print(f"   these thresholds and are excluded from the stratified comparison.\"")

print("\n" + "=" * 66)
print("2. MEAN CORRECTION MAGNITUDE OVER THE FULL RECORD")
print("=" * 66)
o = xr.open_zarr("data/processed/anomalies.zarr", consolidated=True)["rzsm_anom"]
c = xr.open_zarr("data/processed/anomalies_corrected.zarr",
                 consolidated=True)["rzsm_anom_rainfed"]

# stride in time to keep memory sane; every 8th step matches the NIRv cadence
step = 8
mask = irr >= 0.15
print(f"  sampling every {step}th timestep over {o.sizes['time']} steps")
diffs, frac_zero = [], []
for i in range(0, o.sizes["time"], step):
    a = o.isel(time=i).values[mask]
    b = c.isel(time=i).values[mask]
    d = b - a
    d = d[np.isfinite(d)]
    if d.size:
        diffs.append(d)
        frac_zero.append(float(np.mean(np.abs(d) < 1e-9)))
alld = np.concatenate(diffs)
print(f"\n  pixels with irr_frac >= 0.15 (domain-wide): {int(mask.sum())}")
print(f"  samples: {alld.size:,}")
print(f"  mean correction            {alld.mean():+.4f} sigma")
print(f"  mean |correction|          {np.abs(alld).mean():.4f} sigma")
print(f"  median |correction|        {np.median(np.abs(alld)):.4f} sigma")
print(f"  95th pct |correction|      {np.percentile(np.abs(alld),95):.4f} sigma")
print(f"  max |correction|           {np.abs(alld).max():.4f} sigma")
print(f"  fraction receiving no correction  {np.mean(frac_zero):.3f}")
print(f"\n  Growing season only (May-Sep):")
t = o["time"].values[::step]
gs = (t.astype("datetime64[M]").astype(int) % 12 + 1)
gs = (gs >= 5) & (gs <= 9)
gsd = np.concatenate([d for d, g in zip(diffs, gs) if g])
print(f"    mean |correction|        {np.abs(gsd).mean():.4f} sigma")
print(f"\n  Quote the growing-season figure if the paper discusses the growing")
print(f"  season, the all-season figure otherwise. Say which.")
