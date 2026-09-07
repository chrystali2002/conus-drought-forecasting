#!/usr/bin/env python3
"""
09_build_dataset.py
===================
Construct the ConvLSTM training dataset from preprocessed anomalies.

Architecture of each sample
-----------------------------
Input  (context window, 80 days before forecast date t0):
  rzsm_seq   : (80, 1, 321, 656)  irrigation-corrected RZSM anomaly
  nirv_seq   : (80, 1, 321, 656)  NIRv anomaly
  irr_static : (1,  321, 656)     static irrigation fraction (LANID 2020)
  telecon    : (80, 3)            Nino3.4, AMO, PDO scalar indices

Target (NIRv anomaly at each lead time):
  target : (5, 321, 656)  NIRv anomaly at +8, +16, +24, +32, +40 days

Loss mask:
  mask   : (321, 656) bool  water_limited_crop — only these pixels
                             contribute to training loss and evaluation

Time split
----------
  Train : 2015-04-07 → 2021-12-31  (valid t0 positions, stride=8)
  Val   : 2022-01-01 → 2022-12-31
  Test  : 2023-01-01 → 2024-12-26  (held out — never seen during training)

Sample counts (approx, stride=8)
---------------------------------
  Train  : ~290 samples
  Val    : ~40  samples
  Test   : ~80  samples

Memory note
-----------
Loading both anomaly arrays (each ~3 GB) simultaneously uses ~6 GB.
All data is pinned in RAM for fast DataLoader access on GPU training runs.
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS, BATCH_SIZE

out_dir = Path(PROC_ROOT)
STRIDE  = 8   # sample every 8 days — matches MODIS 8-day composite cadence

TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END   = pd.Timestamp("2022-12-31")
# TEST_END  = pd.Timestamp("2024-12-26")   (remainder)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load anomaly arrays into RAM
# ═══════════════════════════════════════════════════════════════════════════
print("Loading anomalies_corrected.zarr ...")
ds_corr  = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
rzsm_arr = ds_corr["rzsm_anom_rainfed"].values.astype(np.float32)  # (T, H, W)
nirv_arr = ds_corr["nirv_anom"].values.astype(np.float32)
times    = pd.DatetimeIndex(ds_corr.time.values)
T, H, W  = rzsm_arr.shape
print("  Loaded: {} x {} x {}  ({:.1f} GB each)".format(
    T, H, W, rzsm_arr.nbytes / 1e9))

# Replace NaN with 0.0 in inputs — NaN pixels lie outside the domain mask
# and will be zeroed-out by the loss mask; 0 is safe as anomaly mid-point
rzsm_arr = np.nan_to_num(rzsm_arr, nan=0.0)
nirv_arr = np.nan_to_num(nirv_arr, nan=0.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load static features
# ═══════════════════════════════════════════════════════════════════════════
ds_master = xr.open_zarr(out_dir / "master_dataset.zarr", consolidated=True)
irr_frac  = ds_master["irr_frac"].values.astype(np.float32)    # (H, W)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load teleconnection indices
# ═══════════════════════════════════════════════════════════════════════════
telecon_df = pd.read_csv(
    out_dir / "teleconnection_indices_daily.csv",
    index_col=0, parse_dates=True
)
telecon_df = telecon_df.reindex(times, method="ffill").fillna(0.0)
telecon_arr = telecon_df[["Nino34", "AMO", "PDO"]].values.astype(np.float32)
N_TELECON   = telecon_arr.shape[1]
print("Teleconnection shape: {}  cols={}".format(
    telecon_arr.shape, list(telecon_df.columns[:3])))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Load domain mask
# ═══════════════════════════════════════════════════════════════════════════
domain_npz = np.load(out_dir / "domain_masks.npz")
wlc_mask   = domain_npz["water_limited_crop"].astype(bool)   # (H, W)
print("Water-limited cropland pixels: {:,}".format(int(wlc_mask.sum())))

MAX_LEAD = max(LEAD_DAYS)

# ═══════════════════════════════════════════════════════════════════════════
# 5. PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════
class DroughtForecastDataset(Dataset):
    """
    Sliding-window dataset for ConvLSTM sub-seasonal NIRv anomaly forecasting.

    Each sample:
      rzsm  : (ctx, 1, H, W)   RZSM anomaly context window
      nirv  : (ctx, 1, H, W)   NIRv anomaly context window
      irr   : (1, H, W)        static irrigation fraction
      tc    : (ctx, n_tc)      teleconnection indices
      target: (n_leads, H, W)  NIRv anomaly at lead times
      mask  : (H, W) bool      water_limited_crop loss mask
    """
    def __init__(self,
                 rzsm:     np.ndarray,
                 nirv:     np.ndarray,
                 irr:      np.ndarray,
                 telecon:  np.ndarray,
                 mask:     np.ndarray,
                 times:    pd.DatetimeIndex,
                 context_len: int,
                 lead_days:   list,
                 stride:      int,
                 split:       str,
                 train_end:   pd.Timestamp,
                 val_end:     pd.Timestamp):

        self.rzsm     = rzsm
        self.nirv     = nirv
        self.irr      = irr[None]          # (1, H, W)
        self.telecon  = telecon
        self.mask     = mask
        self.ctx      = context_len
        self.leads    = np.array(lead_days)
        self.max_lead = max(lead_days)

        # Build valid start indices
        all_t0 = np.arange(context_len, len(times) - self.max_lead, stride)

        if split == "train":
            self.starts = all_t0[times[all_t0] <= train_end]
        elif split == "val":
            self.starts = all_t0[(times[all_t0] > train_end) &
                                  (times[all_t0] <= val_end)]
        elif split == "test":
            self.starts = all_t0[times[all_t0] > val_end]
        else:
            raise ValueError("split must be train/val/test")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]

        # Context window: [t0-ctx, t0)
        rzsm_seq = self.rzsm[t0 - self.ctx : t0]   # (ctx, H, W)
        nirv_seq = self.nirv[t0 - self.ctx : t0]
        tc_seq   = self.telecon[t0 - self.ctx : t0] # (ctx, n_tc)

        # Targets: NIRv anomaly at each lead time
        # lead=8 means 8 days ahead → index t0 + 8 - 1 = t0 + 7
        target = np.stack([
            self.nirv[t0 + lead - 1] for lead in self.leads
        ])  # (n_leads, H, W)

        return {
            "rzsm":    torch.from_numpy(rzsm_seq[:, None]),  # (ctx, 1, H, W)
            "nirv":    torch.from_numpy(nirv_seq[:, None]),
            "irr":     torch.from_numpy(self.irr),            # (1, H, W)
            "tc":      torch.from_numpy(tc_seq),              # (ctx, n_tc)
            "target":  torch.from_numpy(target),              # (n_leads, H, W)
            "mask":    torch.from_numpy(self.mask),           # (H, W) bool
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Instantiate splits
# ═══════════════════════════════════════════════════════════════════════════
common_kwargs = dict(
    rzsm=rzsm_arr, nirv=nirv_arr, irr=irr_frac,
    telecon=telecon_arr, mask=wlc_mask, times=times,
    context_len=CONTEXT_LEN, lead_days=LEAD_DAYS, stride=STRIDE,
    train_end=TRAIN_END, val_end=VAL_END,
)

train_ds = DroughtForecastDataset(**common_kwargs, split="train")
val_ds   = DroughtForecastDataset(**common_kwargs, split="val")
test_ds  = DroughtForecastDataset(**common_kwargs, split="test")

print("\nDataset splits:")
print("  Train : {:4d} samples  ({} to {})".format(
    len(train_ds),
    str(times[train_ds.starts[0]])[:10],
    str(times[train_ds.starts[-1]])[:10]))
print("  Val   : {:4d} samples  ({} to {})".format(
    len(val_ds),
    str(times[val_ds.starts[0]])[:10],
    str(times[val_ds.starts[-1]])[:10]))
print("  Test  : {:4d} samples  ({} to {})".format(
    len(test_ds),
    str(times[test_ds.starts[0]])[:10],
    str(times[test_ds.starts[-1]])[:10]))

# ═══════════════════════════════════════════════════════════════════════════
# 7. Verify one sample shape
# ═══════════════════════════════════════════════════════════════════════════
sample = train_ds[0]
print("\nSample tensor shapes (train[0]):")
for k, v in sample.items():
    print("  {:8s}: {}  dtype={}".format(k, tuple(v.shape), v.dtype))

assert sample["rzsm"].shape   == (CONTEXT_LEN, 1, H, W),    "rzsm shape wrong"
assert sample["nirv"].shape   == (CONTEXT_LEN, 1, H, W),    "nirv shape wrong"
assert sample["irr"].shape    == (1, H, W),                  "irr shape wrong"
assert sample["tc"].shape     == (CONTEXT_LEN, N_TELECON),   "tc shape wrong"
assert sample["target"].shape == (len(LEAD_DAYS), H, W),     "target shape wrong"
assert sample["mask"].shape   == (H, W),                     "mask shape wrong"
print("  All shapes verified OK")

# ═══════════════════════════════════════════════════════════════════════════
# 8. DataLoaders
# ═══════════════════════════════════════════════════════════════════════════
# pin_memory=True: speeds up CPU→GPU transfer
# persistent_workers: keeps worker processes alive between epochs
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4,
                          pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2,
                          pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=1,
                          shuffle=False, num_workers=2,
                          pin_memory=True)

print("\nDataLoaders:")
print("  Train batches/epoch : {}".format(len(train_loader)))
print("  Val   batches/epoch : {}".format(len(val_loader)))
print("  Test  batches       : {}".format(len(test_loader)))

# ═══════════════════════════════════════════════════════════════════════════
# 9. Dataset statistics (for model paper / reproducibility)
# ═══════════════════════════════════════════════════════════════════════════
# Compute mean/std over the TRAINING split only (no data leakage)
print("\nComputing train-split statistics for model paper ...")
wlc_flat = wlc_mask.ravel()

def masked_stats(arr, mask):
    """Mean and std over water-limited cropland pixels across time."""
    flat = arr.reshape(arr.shape[0], -1)[:, mask.ravel()]
    return float(np.nanmean(flat)), float(np.nanstd(flat))

train_t0_start = train_ds.starts[0]
train_t0_end   = train_ds.starts[-1] + MAX_LEAD

rzsm_train = rzsm_arr[train_t0_start : train_t0_end + 1]
nirv_train = nirv_arr[train_t0_start : train_t0_end + 1]

stats = {
    "train_period": {
        "start": str(times[train_t0_start].date()),
        "end":   str(times[train_t0_end].date()),
    },
    "rzsm_anom": {
        "mean": masked_stats(rzsm_train, wlc_mask)[0],
        "std":  masked_stats(rzsm_train, wlc_mask)[1],
    },
    "nirv_anom": {
        "mean": masked_stats(nirv_train, wlc_mask)[0],
        "std":  masked_stats(nirv_train, wlc_mask)[1],
    },
    "irr_frac": {
        "mean": float(irr_frac[wlc_mask].mean()),
        "std":  float(irr_frac[wlc_mask].std()),
    },
    "n_train":   len(train_ds),
    "n_val":     len(val_ds),
    "n_test":    len(test_ds),
    "context_len": CONTEXT_LEN,
    "lead_days":   LEAD_DAYS,
    "stride_days": STRIDE,
    "grid":        [H, W],
    "n_wlc_pixels": int(wlc_mask.sum()),
}

stats_path = out_dir / "dataset_statistics.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print("  Saved: {}".format(stats_path))

for k in ["rzsm_anom", "nirv_anom"]:
    print("  {:12s}: mean={:.4f}  std={:.4f}".format(
        k, stats[k]["mean"], stats[k]["std"]))
print("  Expected: mean ~0, std ~1 (already z-scored in Steps 6-7)")

print("\n── Step 9 complete ──")
print("Dataset is ready for ConvLSTM training (Step 10).")
print("\nSummary for model paper:")
print("  Training samples  : {:,}".format(len(train_ds)))
print("  Validation samples: {:,}".format(len(val_ds)))
print("  Test samples      : {:,}".format(len(test_ds)))
print("  Input channels    : 2 dynamic (RZSM, NIRv) + 1 static (irr_frac)")
print("                    + {} teleconnection scalars".format(N_TELECON))
print("  Context window    : {} days".format(CONTEXT_LEN))
print("  Lead times        : {} days".format(LEAD_DAYS))
print("  Domain pixels     : {:,} water-limited cropland".format(int(wlc_mask.sum())))
