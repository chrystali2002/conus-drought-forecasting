#!/usr/bin/env python3
"""Verify the Step 7 irrigation correction behaves as the QC figure implies.

Panels 1 and 2 of qc_irrigation_deconvolution.png appear to differ over large
areas (the central US is redder, the Southeast bluer in the corrected panel),
yet panel 3 shows the correction is near zero and confined to irrigated pixels.
Those two readings are inconsistent: if the correction only touches pixels with
irr_frac >= 0.15, the two fields must be IDENTICAL everywhere else. This checks
that numerically instead of by eye.
"""
import numpy as np, xarray as xr, sys

date = sys.argv[1] if len(sys.argv) > 1 else "2020-07-01"
orig = xr.open_zarr("data/processed/anomalies.zarr", consolidated=False)
corr = xr.open_zarr("data/processed/anomalies_corrected.zarr", consolidated=False)
mst  = xr.open_zarr("data/processed/master_dataset.zarr", consolidated=False)

o = orig["rzsm_anom"].sel(time=date, method="nearest").values
c = corr["rzsm_anom_rainfed"].sel(time=date, method="nearest").values
irr = mst["irr_frac"].values
if irr.max() > 1.5:
    irr = irr / 100.0

d = c - o
valid = np.isfinite(d)
inside  = valid & (irr >= 0.15)
outside = valid & (irr <  0.15)

print(f"date {date}   grid {o.shape}")
print(f"\nINSIDE  irr_frac >= 0.15   ({inside.sum():,} px)")
print(f"  mean correction {np.nanmean(d[inside]):+.4f}  "
      f"min {np.nanmin(d[inside]):+.4f}  max {np.nanmax(d[inside]):+.4f}")
print(f"  fraction negative (drier after correction) {np.mean(d[inside] < 0):.3f}")

print(f"\nOUTSIDE irr_frac <  0.15   ({outside.sum():,} px)")
print(f"  max |correction| {np.nanmax(np.abs(d[outside])):.6f}")
n_nonzero = int((np.abs(d[outside]) > 1e-6).sum())
print(f"  pixels changed   {n_nonzero:,}")
if n_nonzero == 0:
    print("  OK - untouched outside irrigated pixels, as intended.")
else:
    print("  PROBLEM - the correction altered non-irrigated pixels. Most likely")
    print("  cause: the anomaly was RE-STANDARDIZED after correction, so every")
    print("  pixel's z-score shifted because the domain mean and sd changed. That")
    print("  would contaminate your rainfed control and inflate the apparent")
    print("  rainfed-vs-irrigated skill difference. Check Step 7 for a second")
    print("  standardization applied after the deconvolution.")

print(f"\nWHOLE FIELD")
print(f"  mean |correction| {np.nanmean(np.abs(d[valid])):.4f} sigma")
print(f"  95th pct |correction| {np.nanpercentile(np.abs(d[valid]),95):.4f} sigma")
