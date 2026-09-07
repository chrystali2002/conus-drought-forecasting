#!/usr/bin/env python3
"""
07_irrigation_deconvolution.py
==============================
Remove the irrigation-induced RZSM signal from the standardised anomaly
in pixels where the LANID irrigation fraction exceeds the low threshold.

Scientific rationale
--------------------
In irrigated areas SMAP L4 RZSM reflects a mixture of rainfall-driven
and management-driven soil moisture. Using the raw RZSM anomaly in those
pixels would attribute artificial moisture to the drought predictor signal,
biasing the ConvLSTM away from physical drought dynamics.

CONUS extension over Africa study: the Africa source study trained on
predominantly rainfed cropland. For CONUS we explicitly correct the RZSM
signal in the 6,454 pixels (3.1% of domain) with irr_frac > 0.15.

Method (linear regional excess correction)
-------------------------------------------
For each time step t:
  1. For each irrigated pixel p (irr_frac > IRR_HIGH_THRESHOLD):
     a. Compute the rainfed regional background rzsm_regional(p) as the
        spatial mean of non-irrigated pixels within a ~1-degree window
        using scipy.ndimage.uniform_filter, masking irrigated pixels out.
     b. Estimate the irrigation excess:
          irr_excess(p) = rzsm_obs(p) - rzsm_regional(p)
     c. Apply a scaled correction:
          correction(p) = irr_frac(p) * max(irr_excess(p), 0)
          (only correct positive excess — don't amplify droughts)
  2. rzsm_rainfed = rzsm_obs - correction
  3. Clip to [-4, 4] sigma.

This is a first-order correction. A full data-assimilation approach
(ensemble Kalman filter with irrigation withdrawal priors) is an open
research direction noted in the paper.

References
----------
Xie Y et al. (2021) LANID ESSD 13:5689 (irrigation fraction source).
Brocca L et al. (2018) Geophys. Res. Lett. (RZSM irrigation signature).

Thresholds (calibrated to LANID output from Step 4b)
------------------------------------------------------
  IRR_HIGH_THRESHOLD = 0.15   (6,454 px, 3.1% of grid)
  IRR_LOW_THRESHOLD  = 0.05   (13,403 px, 6.4% of grid)
  RAINFED_THRESHOLD  = 0.05   (pixels < 0.05 treated as rainfed)

Outputs
-------
  data/processed/anomalies_corrected.zarr
    rzsm_anom_rainfed : (T, 321, 656)  irrigation-corrected RZSM anomaly
    nirv_anom         : (T, 321, 656)  unchanged NIRv anomaly (copied)
"""

import numpy as np
import xarray as xr
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from pathlib import Path
from tqdm import tqdm
from config import PROC_ROOT

IRR_HIGH_THRESHOLD = 0.15   # strongly irrigation-affected
IRR_LOW_THRESHOLD  = 0.05   # weakly affected (included in correction)
FILTER_SIZE        = 11     # ~1-degree spatial window at 0.09-deg resolution
CLIP_SIGMA         = 4.0    # clip corrected anomalies to ±4σ

out_dir = Path(PROC_ROOT)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load inputs
# ═══════════════════════════════════════════════════════════════════════════
anom_zarr   = out_dir / "anomalies.zarr"
master_zarr = out_dir / "master_dataset.zarr"
assert anom_zarr.exists(),   "anomalies.zarr not found. Run Step 6."
assert master_zarr.exists(), "master_dataset.zarr not found. Run Step 5."

ds_anom   = xr.open_zarr(anom_zarr,   consolidated=True)
ds_master = xr.open_zarr(master_zarr, consolidated=True)

rzsm_anom = ds_anom["rzsm_anom"].values.astype(np.float32)   # (T, H, W)
nirv_anom = ds_anom["nirv_anom"].values.astype(np.float32)
irr_frac  = ds_master["irr_frac"].values.astype(np.float32)  # (H, W)

T, H, W = rzsm_anom.shape
print("Anomaly shape    : {} x {} x {}".format(T, H, W))
print("Irrigation frac  : range [{:.3f}, {:.3f}]".format(
    float(irr_frac.min()), float(irr_frac.max())))

irr_high_mask  = irr_frac >= IRR_HIGH_THRESHOLD    # (H, W) bool
irr_any_mask   = irr_frac >= IRR_LOW_THRESHOLD
rainfed_mask   = irr_frac <  IRR_LOW_THRESHOLD

print("High-irr pixels  : {:,}  (irr_frac >= {})".format(
    int(irr_high_mask.sum()), IRR_HIGH_THRESHOLD))
print("Rainfed pixels   : {:,}  (irr_frac <  {})".format(
    int(rainfed_mask.sum()),  IRR_LOW_THRESHOLD))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Irrigation deconvolution (time-step loop)
# ═══════════════════════════════════════════════════════════════════════════
rzsm_rainfed = rzsm_anom.copy()

print("\nApplying irrigation deconvolution ({} time steps) ...".format(T))

corrections_applied = 0
for t in tqdm(range(T), desc="Deconvolve", ncols=70):
    slice_t = rzsm_anom[t]   # (H, W)

    # Regional rainfed background: spatial mean of non-irrigated pixels
    # within ~1-degree window. Irrigated pixels are excluded from average.
    rainfed_vals = np.where(~irr_high_mask, slice_t, np.nan)

    # uniform_filter with NaN handling: replace NaN with 0, filter both
    # signal and a valid-count mask, then divide (weighted mean)
    valid_mask   = np.isfinite(rainfed_vals).astype(np.float32)
    signal_filled = np.where(np.isfinite(rainfed_vals), rainfed_vals, 0.0)

    signal_smooth = uniform_filter(signal_filled, size=FILTER_SIZE,
                                   mode="constant", cval=0.0)
    count_smooth  = uniform_filter(valid_mask,   size=FILTER_SIZE,
                                   mode="constant", cval=0.0)

    # Avoid division by zero (no rainfed neighbours)
    with np.errstate(invalid="ignore", divide="ignore"):
        rzsm_regional = np.where(
            count_smooth > 0.05,
            signal_smooth / count_smooth,
            slice_t   # fallback: use pixel's own value if no rainfed neighbours
        )

    # Irrigation excess = how much wetter irrigated pixels are vs surroundings
    irr_excess = slice_t - rzsm_regional   # (H, W)

    # Correction = irr_frac * positive excess only
    # We only remove artificial wetness (excess > 0).
    # If irrigated pixels are DRIER than surroundings (drought), no correction.
    correction = irr_frac * np.where(
        irr_high_mask & (irr_excess > 0),
        irr_excess,
        0.0
    )

    # Apply clip ONLY to pixels where a correction was actually made.
    # Rainfed pixels (irr_frac < IRR_LOW_THRESHOLD) must be bit-for-bit
    # identical to the input; the clip must never touch them.
    # Without this guard, np.clip modifies rainfed pixels whose raw anomaly
    # falls outside ±CLIP_SIGMA (possible for sparse DOY samples).
    corrected = slice_t - correction
    rzsm_rainfed[t] = np.where(
        irr_any_mask,                              # irrigated pixels: apply + clip
        np.clip(corrected, -CLIP_SIGMA, CLIP_SIGMA),
        slice_t                                    # rainfed pixels: original unchanged
    )

    if t == 0:
        # Diagnostic on first step
        n_corrected = int((correction[irr_high_mask] > 0.01).sum())
        mean_correction = float(correction[irr_high_mask & (correction > 0.01)].mean()) \
            if (correction[irr_high_mask] > 0.01).any() else 0.0
        print("  Step t=0: {:,} high-irr pixels corrected, mean correction={:.3f}σ".format(
            n_corrected, mean_correction))
    corrections_applied += 1

print("Deconvolution complete.")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════
wlc_mask = ds_master["water_limited_crop"].values.astype(bool)

print("\nCorrection magnitude diagnostics (high-irr pixels, irr_frac >= 0.15):")
if irr_high_mask.sum() > 0:
    orig_mean  = float(np.nanmean(rzsm_anom[:, irr_high_mask]))
    corr_mean  = float(np.nanmean(rzsm_rainfed[:, irr_high_mask]))
    orig_std   = float(np.nanstd(rzsm_anom[:, irr_high_mask]))
    corr_std   = float(np.nanstd(rzsm_rainfed[:, irr_high_mask]))
    print("  RZSM original   : mean={:.3f}  std={:.3f}".format(orig_mean, orig_std))
    print("  RZSM corrected  : mean={:.3f}  std={:.3f}".format(corr_mean, corr_std))
    print("  Mean shift      : {:.3f}σ  (correction brought irrigated pixels closer to 0)".format(
        orig_mean - corr_mean))

print("\nRainfed domain (irr_frac < 0.05) — should be UNCHANGED:")
if rainfed_mask.sum() > 0:
    diff = rzsm_rainfed[:, rainfed_mask] - rzsm_anom[:, rainfed_mask]
    print("  Max absolute diff: {:.6f}  (should be 0.0)".format(
        float(np.nanmax(np.abs(diff)))))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Save corrected anomalies
# ═══════════════════════════════════════════════════════════════════════════
corr_zarr  = out_dir / "anomalies_corrected.zarr"
compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)

import pandas as pd
time_coords = pd.DatetimeIndex(ds_anom.time.values).values.astype("datetime64[ns]")

ds_corr = xr.Dataset(
    {
        "rzsm_anom_rainfed": (["time","lat","lon"], rzsm_rainfed,
                              {"long_name": ("Irrigation-corrected RZSM anomaly "
                                             "(z-score)"),
                               "units": "sigma",
                               "method": ("Regional rainfed excess correction; "
                                          "irr_frac threshold = {}".format(
                                              IRR_HIGH_THRESHOLD)),
                               "source": "SMAP L4 + LANID 2020 correction"}),
        "nirv_anom":         (["time","lat","lon"], nirv_anom,
                              {"long_name": "NIRv standardised anomaly (z-score)",
                               "units": "sigma",
                               "note": "Unchanged from Step 6"}),
    },
    coords={"time": time_coords,
            "lat":  ds_anom.lat.values,
            "lon":  ds_anom.lon.values},
    attrs={
        "title":           "CONUS Drought Forecast — Corrected Anomalies",
        "irr_threshold":   IRR_HIGH_THRESHOLD,
        "filter_size_deg": FILTER_SIZE * 0.09,
    }
).chunk({"time": 10, "lat": H, "lon": W})

print("\nWriting anomalies_corrected.zarr ...")
ds_corr.to_zarr(
    corr_zarr, mode="w", consolidated=True,
    encoding={
        "rzsm_anom_rainfed": {"dtype": "float32", "compressor": compressor},
        "nirv_anom":         {"dtype": "float32", "compressor": compressor},
    }
)
size_mb = sum(f.stat().st_size for f in corr_zarr.rglob("*") if f.is_file()) / 1e6
print("Saved: {}  ({:.0f} MB)".format(corr_zarr, size_mb))

# ═══════════════════════════════════════════════════════════════════════════
# 5. QC plot: before vs after correction for a summer day
# ═══════════════════════════════════════════════════════════════════════════
times  = pd.DatetimeIndex(ds_anom.time.values)
ref_lons = ds_anom.lon.values
ref_lats = ds_anom.lat.values
extent = [float(ref_lons[0]), float(ref_lons[-1]),
          float(ref_lats[-1]), float(ref_lats[0])]

# Find a July date (peak irrigation season)
july_mask = (times.month == 7)
july_idx  = np.where(july_mask)[0]
plot_t    = july_idx[len(july_idx)//2]   # mid-series July
plot_date = str(times[plot_t].date())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

before = rzsm_anom[plot_t].copy()
after  = rzsm_rainfed[plot_t].copy()
diff   = after - before

# Mask non-irrigated pixels for the diff panel
diff_show = np.where(irr_high_mask, diff, np.nan)

for ax, data, title, cmap, vmin, vmax in [
    (axes[0], before,    "RZSM anom (original)\n{}".format(plot_date),
     "RdBu", -3, 3),
    (axes[1], after,     "RZSM anom (irrigat. corrected)\n{}".format(plot_date),
     "RdBu", -3, 3),
    (axes[2], diff_show, "Correction  (corrected − original)\nirr_frac ≥ {} only".format(
        IRR_HIGH_THRESHOLD),
     "PuOr_r", -1, 1),
]:
    im = ax.imshow(data, extent=extent, origin="upper",
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="σ", shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

plt.suptitle("Step 7 — Irrigation deconvolution QC", fontsize=12)
plt.tight_layout()
qc_path = out_dir / "qc_irrigation_deconvolution.png"
plt.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.close()
print("QC plot: {}".format(qc_path))

print("\n── Step 7 complete ──")
print("Output: anomalies_corrected.zarr")
print("  rzsm_anom_rainfed : irrigation-corrected  (T={}, H={}, W={})".format(T,H,W))
print("  nirv_anom         : unchanged NIRv anomaly")
print("\nExpected QC:")
print("  Irrigated pixels: mean correction 0.1–0.5σ (reduced positive bias)")
print("  Rainfed pixels  : max diff = 0.000 (completely unchanged)")
print("  Correction map  : negative values (blue) in Snake River Plain,")
print("    Central Valley CA, Kansas High Plains, Columbia Basin")
print("\nNext: Step 8 (aridity stratification and domain mask assembly)")
