#!/usr/bin/env python3
"""How much of the EVALUATION domain does the irrigation correction touch?

This decides whether a corrected-vs-uncorrected ablation can show anything. If
only a small share of the evaluation pixels are irrigated, a domain-averaged
ACC difference is diluted toward zero BY CONSTRUCTION - not because the
correction is ineffective, but because most pixels are identical in both arms.
The ablation must then be scored on irrigated pixels, with the domain average
reported as context rather than as the headline.
"""
import numpy as np, xarray as xr

ms = xr.open_zarr("data/processed/master_dataset.zarr", consolidated=False)
irr = ms["irr_frac"].values
if irr.max() > 1.5:
    irr = irr / 100.0
wlc = ms["water_limited_crop"].values > 0.5
crop = ms["crop_mask"].values > 0.5

print(f"evaluation domain (water_limited_crop): {int(wlc.sum()):,} px\n")
print(f"{'threshold':>12} {'all crop':>10} {'in WLC':>8} {'% of WLC':>9}")
for th in (0.05, 0.10, 0.15, 0.25, 0.50):
    a = int((crop & (irr >= th)).sum())
    w = int((wlc & (irr >= th)).sum())
    print(f"  irr >= {th:<5.2f} {a:10,} {w:8,} {w/max(wlc.sum(),1)*100:8.1f}%")

n15 = int((wlc & (irr >= 0.15)).sum())
frac = n15 / max(int(wlc.sum()), 1)
print(f"\nStep 7 corrects pixels with irr_frac >= 0.15.")
print(f"Within the evaluation domain that is {n15:,} px ({frac*100:.1f}%).")
print(f"\nIMPLICATION")
if frac < 0.15:
    print(f"  A domain-averaged ACC difference will be roughly {frac:.2f} x the")
    print(f"  effect on irrigated pixels. If the correction improves irrigated-pixel")
    print(f"  ACC by 0.04, the domain average moves by about "
          f"{0.04*frac:.3f} - indistinguishable from noise.")
    print(f"  Score the ablation ON IRRIGATED PIXELS. Report the domain average")
    print(f"  as context, and say plainly why it is small.")
else:
    print(f"  A domain-averaged comparison is defensible, but still report the")
    print(f"  irrigated-pixel result separately - that is where the mechanism acts.")

print(f"\nAlso note for the diagnostic: irr_frac >= 0.50 gives only "
      f"{int((crop & (irr >= 0.50)).sum()):,} cropland px domain-wide.")
print("  At 9 km a cell is rarely more than half irrigated, so the default")
print("  --irr-thresh 0.5 starves the pairing. Use 0.15 to match Step 7.")
