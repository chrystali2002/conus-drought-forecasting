#!/usr/bin/env python3
"""Pre-flight: does the regridded precipitation actually cover the pixels the
diagnostic will use?

finite_fraction=0.496 for the precipitation field is expected - the target grid
is a lat/lon box containing the Gulf, both oceans, southern Canada and northern
Mexico, and gridMET is CONUS land only. What matters is not the global fraction
but whether precipitation exists AT THE CROPLAND PIXELS that will be paired. If
it does not, those pixels drop out silently and the diagnostic quietly shrinks.
"""
import numpy as np, xarray as xr

pr = xr.open_zarr("data/processed/precip_smap_grid.zarr", consolidated=False)
an = xr.open_zarr("data/processed/anomalies.zarr", consolidated=False)
ms = xr.open_zarr("data/processed/master_dataset.zarr", consolidated=False)

print("TIME AXIS")
tp, ta = pr["time"].values, an["time"].values
print(f"  precip {len(tp)} steps, anomalies {len(ta)} steps")
same = len(tp) == len(ta) and bool((np.asarray(tp, 'datetime64[D]') ==
                                   np.asarray(ta, 'datetime64[D]')).all())
print(f"  identical: {same}")
if not same:
    print("  -> reindex precip onto the anomalies time axis before proceeding")

irr = ms["irr_frac"].values
if irr.max() > 1.5:
    irr = irr / 100.0
crop = ms["crop_mask"].values > 0.5
wlc = ms["water_limited_crop"].values > 0.5

# a pixel is usable if precipitation is present most of the time
pfin = np.isfinite(pr["precip"].isel(time=slice(0, 400)).values).mean(axis=0) > 0.9

print("\nCOVERAGE AT THE PIXELS THAT MATTER")
for name, m in (("all cropland", crop), ("water-limited cropland", wlc)):
    tot = int(m.sum())
    cov = int((m & pfin).sum())
    print(f"  {name:24s} {tot:7,} px, precip at {cov:7,} ({cov/max(tot,1)*100:5.1f}%)")

for name, m in (("irrigated cropland (>=0.5)", crop & (irr >= 0.5)),
                ("rainfed cropland (<=0.05)", crop & (irr <= 0.05))):
    tot = int(m.sum())
    cov = int((m & pfin).sum())
    print(f"  {name:24s} {tot:7,} px, precip at {cov:7,} ({cov/max(tot,1)*100:5.1f}%)")

lost = int((crop & ~pfin).sum())
if lost:
    lat = ms["lat"].values
    rows = np.where((crop & ~pfin).any(axis=1))[0]
    print(f"\n  {lost:,} cropland pixels have no precipitation.")
    if len(rows):
        print(f"  affected latitudes {lat[rows].min():.2f} to {lat[rows].max():.2f}")
        print("  If these are all above 49.42N they are outside gridMET and outside")
        print("  CONUS; that is acceptable. If they are inside CONUS, investigate")
        print("  before running the diagnostic.")
else:
    print("\n  Every cropland pixel has precipitation. Good to proceed.")

print("\nPRECIP SANITY")
v = pr["precip"].isel(time=slice(0, 400)).values
print(f"  mean {np.nanmean(v):.3f} mm/day -> {np.nanmean(v)*365:.0f} mm/yr")
print(f"  max daily {np.nanmax(v):.1f} mm, negatives: {int((v < 0).sum())}")
