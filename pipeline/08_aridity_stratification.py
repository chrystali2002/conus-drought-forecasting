#!/usr/bin/env python3
"""
08_domain_masks.py
==================
Assemble the final analysis masks used by the ConvLSTM training pipeline.
All masks are on the 321x656 SMAP/NIRv 0.09-degree grid.

Masks produced
--------------
  water_limited_crop   : dryland (AI 0.05-0.65) AND CDL crop mask
                         Primary RZSM-skill evaluation domain (mirrors Africa)
  rainfed_crop         : water_limited_crop AND irr_frac < 0.05
                         Rainfed-dominant pixels; used for primary model training
  irrigated_crop       : water_limited_crop AND irr_frac >= 0.15
                         Irrigation-affected pixels; Step 7 corrected RZSM used
  semi_arid_crop       : semi-arid only (AI 0.20-0.50) AND CDL crop
                         Strict match to Africa study domain for ACC comparison
  energy_limited_crop  : humid (AI >= 0.65) AND CDL crop
                         Control domain; RZSM skill expected minimal
  Reporting-zone masks : intersection of above with NOAA/USDA reporting zones

Scientific basis
----------------
Dryland stratification follows UNEP (1992) / IPCC AR6 AI framework.
Water-limitation vs energy-limitation split mirrors the Africa study's
finding that RZSM advantage concentrates in water-limited (AI < 0.65)
croplands and is negligible in energy-limited (AI >= 0.65) regions.

Outputs
-------
  data/processed/domain_masks.npz      all boolean masks as compressed array
  data/processed/domain_masks.nc       labelled NetCDF for inspection
  data/processed/qc_domain_masks.png   QC figure
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from config import PROC_ROOT, REPORTING_ZONES

out_dir = Path(PROC_ROOT)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load component layers (all 321x656 from previous steps)
# ═══════════════════════════════════════════════════════════════════════════
ds_master = xr.open_zarr(out_dir / "master_dataset.zarr", consolidated=True)

aridity_class = ds_master["aridity_class"].values       # int8  (H, W)
crop_mask     = ds_master["crop_mask"].values.astype(bool)  # bool
irr_frac      = ds_master["irr_frac"].values            # float32
ref_lats      = ds_master.lat.values
ref_lons      = ds_master.lon.values
H, W          = aridity_class.shape

print("Grid            : {} x {}".format(H, W))
print("Total pixels    : {:,}".format(H * W))
print("CDL crop pixels : {:,}  ({:.1f}%)".format(
    int(crop_mask.sum()), crop_mask.mean() * 100))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Aridity-based sub-masks
# ═══════════════════════════════════════════════════════════════════════════
dryland_mask   = (aridity_class >= 1) & (aridity_class <= 3)  # AI 0.05-0.65
semi_arid_mask = (aridity_class == 2)                          # AI 0.20-0.50
humid_mask     = (aridity_class == 4)                          # AI >= 0.65

# Irrigation sub-masks (calibrated to LANID Step 4b output)
IRR_HIGH = 0.15
IRR_LOW  = 0.05
rainfed_mask   = irr_frac <  IRR_LOW    # < 5%  irrigated
irrigated_mask = irr_frac >= IRR_HIGH   # >= 15% irrigated

# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined domain masks
# ═══════════════════════════════════════════════════════════════════════════
masks = {
    # Primary evaluation domain (mirrors Africa study)
    "water_limited_crop":  dryland_mask   & crop_mask,
    # Training sub-domains
    "rainfed_crop":        dryland_mask   & crop_mask & rainfed_mask,
    "irrigated_crop":      dryland_mask   & crop_mask & irrigated_mask,
    # Strict semi-arid for direct Africa ACC comparison
    "semi_arid_crop":      semi_arid_mask & crop_mask,
    # Control domain (energy-limited; RZSM advantage expected minimal)
    "energy_limited_crop": humid_mask     & crop_mask,
    # Full dryland (no crop filter) — for domain-wide RZSM statistics
    "dryland_all":         dryland_mask,
}

print("\nDomain mask pixel counts:")
print("{:28s} {:>8s} {:>7s}".format("Mask", "Pixels", "% grid"))
print("-" * 45)
for name, mask in masks.items():
    n = int(mask.sum())
    print("{:28s} {:8,d}  {:6.2f}%".format(name, n, n / (H*W) * 100))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Reporting-zone sub-masks
#    Applied AFTER aridity mask — zones only select already-qualified drylands
# ═══════════════════════════════════════════════════════════════════════════
lon_grid, lat_grid = np.meshgrid(ref_lons, ref_lats)

zone_masks = {}
print("\nReporting-zone dryland cropland pixels:")
for zone_name, (zlon_min, zlat_min, zlon_max, zlat_max) in REPORTING_ZONES.items():
    spatial = ((lon_grid >= zlon_min) & (lon_grid <= zlon_max) &
               (lat_grid >= zlat_min) & (lat_grid <= zlat_max))
    zone_masks[zone_name] = spatial & masks["water_limited_crop"]
    n = int(zone_masks[zone_name].sum())
    print("  {:35s}: {:,} px".format(zone_name, n))
    masks["zone_" + zone_name] = zone_masks[zone_name]

# ═══════════════════════════════════════════════════════════════════════════
# 5. Save
# ═══════════════════════════════════════════════════════════════════════════
# Compressed numpy archive (fast load for training pipeline)
np.savez_compressed(
    out_dir / "domain_masks.npz",
    **{k: v.astype(np.int8) for k, v in masks.items()}
)
print("\nSaved: domain_masks.npz  ({} masks)".format(len(masks)))

# Labelled NetCDF for visual inspection
ds_masks = xr.Dataset(
    {k: (["lat","lon"], v.astype(np.int8),
         {"long_name": k.replace("_"," "),
          "flag_values": [0, 1],
          "flag_meanings": "outside domain"})
     for k, v in masks.items()},
    coords={"lat": ref_lats.astype(np.float32),
            "lon": ref_lons.astype(np.float32)},
    attrs={"title": "CONUS Drought Forecast — Domain Masks",
           "grid":  "321 x 656  0.09-deg EPSG:4326",
           "basis": "UNEP/IPCC AI + USDA CDL + USGS LANID 2020"}
)
ds_masks.to_netcdf(
    out_dir / "domain_masks.nc",
    encoding={v: {"zlib": True, "complevel": 5} for v in ds_masks.data_vars}
)
print("Saved: domain_masks.nc")

# ═══════════════════════════════════════════════════════════════════════════
# 6. QC plot — all primary masks on one figure
# ═══════════════════════════════════════════════════════════════════════════
extent = [float(ref_lons[0]), float(ref_lons[-1]),
          float(ref_lats[-1]), float(ref_lats[0])]

primary_masks = [
    ("water_limited_crop",  "Water-limited cropland\n(primary domain)",  "#2d6a0f"),
    ("rainfed_crop",        "Rainfed dryland crop\n(primary training)",  "#185FA5"),
    ("irrigated_crop",      "Irrigated dryland crop\n(deconv. applied)", "#BA7517"),
    ("semi_arid_crop",      "Semi-arid crop only\n(Africa comparison)",  "#f5c842"),
    ("energy_limited_crop", "Energy-limited crop\n(humid control)",      "#6baed6"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes_flat = axes.flatten()   # numpy array, not iterator -- safe to index after zip

for (name, title, color), ax in zip(primary_masks, axes_flat):
    mask_data = masks[name].astype(np.float32)
    mask_data[mask_data == 0] = np.nan

    cmap_bin = mcolors.ListedColormap(["white", color])
    ax.imshow(np.where(np.isnan(mask_data), 0, mask_data),
              extent=extent, origin="upper",
              cmap=cmap_bin, vmin=0, vmax=1, aspect="auto", alpha=0.9)
    n = int(masks[name].sum())
    ax.set_title("{}\n{:,} px ({:.1f}%)".format(
        title, n, n / (H*W) * 100), fontsize=9)
    ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

# Last panel: all zones overlaid
ax_last = list(axes_flat)[5]
zone_colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00"]
for (zname, _), zcol in zip(REPORTING_ZONES.items(), zone_colors):
    zm = zone_masks.get(zname)
    if zm is None: continue
    overlay = np.where(zm, 1.0, np.nan)
    cmap_z = mcolors.ListedColormap(["white", zcol])
    ax_last.imshow(np.where(np.isnan(overlay), 0, overlay),
                   extent=extent, origin="upper",
                   cmap=cmap_z, vmin=0, vmax=1,
                   aspect="auto", alpha=0.7)
ax_last.set_title("Reporting zones\n(water-limited cropland)", fontsize=9)
ax_last.set_xlabel("Lon"); ax_last.set_ylabel("Lat")

plt.suptitle("Step 8 — Domain masks (321×656 SMAP grid)", fontsize=12)
plt.tight_layout()
plt.savefig(out_dir / "qc_domain_masks.png", dpi=150, bbox_inches="tight")
plt.close()
print("QC plot: {}".format(out_dir / "qc_domain_masks.png"))

print("\n── Step 8 complete ──")
print("Ready for ConvLSTM dataset construction (Step 9).")
print("\nKey domain sizes for model paper:")
for name in ["water_limited_crop", "rainfed_crop", "semi_arid_crop"]:
    n = int(masks[name].sum())
    print("  {:28s}: {:,} px  ({:.2f}%)".format(name, n, n/(H*W)*100))
