#!/usr/bin/env python3
"""
12_evaluate.py
==============
Evaluate the best ConvLSTM checkpoint on the held-out test set (2023-2024).

Checkpoint used
---------------
  models/best_convlstm_conus.pt  (epoch 7, val ACC=0.5813)
  Confirmed optimal by sensitivity tests:
    - patience=25: best also epoch 7, ACC=0.5810
    - 3-epoch smooth: best epoch 14, ACC=0.5793
  All three within 0.002 ACC -- result is robust to stopping criterion.

Metrics
-------
  ACC  : Anomaly Correlation Coefficient (Pearson r, domain-averaged)
  RMSE : Root mean square error (sigma units)
  SS   : Skill score = 1 - RMSE_model / RMSE_climatology
         > 0 means model beats the climatological (zero-anomaly) baseline

Baselines
---------
  Climatology : predict zero anomaly everywhere (trivial baseline)
  Persistence : carry last observed NIRv anomaly forward to all lead times

Stratification
--------------
  1. Full water_limited_crop domain (primary result)
  2. Rainfed vs irrigated sub-domains
  3. Strict semi-arid (AI 0.20-0.50) -- Africa paper comparison
  4. NOAA/USDA reporting zones (SGP, NGP, DSW, IMW, CV-CA)

Outputs
-------
  outputs/metrics/evaluation_summary.csv
  outputs/metrics/evaluation_by_zone.csv
  outputs/metrics/qc_evaluation.png

Paper significance of each output
-----------------------------------
evaluation_summary.csv
  ACC, RMSE, and skill score at each of the 5 lead times against two
  baselines: climatology (predict zero anomaly everywhere) and persistence
  (carry the last observed NIRv forward to all lead times).
  Column RMSE_reduction_pct is the primary model-skill metric.
  The Africa source paper reports 22-24% RMSE reduction vs climatology;
  this column gives the CONUS equivalent for direct comparison in Table 1.

evaluation_by_zone.csv
  ACC stratified by rainfed vs irrigated cropland, semi-arid vs dry
  sub-humid, and each NOAA/USDA reporting zone (SGP, NGP, DSW, IMW,
  CV-CA). The key validation the paper requires:
    rainfed ACC > irrigated ACC at 40-day lead
  This confirms that the Step 7 irrigation deconvolution removed the
  artificial RZSM wetness signal in managed pixels rather than adding
  noise. If irrigated ACC is higher, the deconvolution overshot.

qc_evaluation.png — three panels:
  Panel 1 (ACC vs lead): ConvLSTM skill curve against persistence and
    the Africa benchmark (0.555 dashed line). Should show clear
    advantage over persistence at all lead times.
  Panel 2 (RMSE reduction bar chart): skill score per lead time.
    Positive bars = model beats climatology. Bars should be largest
    at short lead (8d) and taper toward 40d.
  Panel 3 (spatial ACC map, 40-day lead): pixel-level Pearson r over
    the water-limited cropland domain. Analogous to Africa paper
    Figure 3. Highest skill expected in the Southern Great Plains
    (Kansas/Oklahoma winter wheat belt) where RZSM-NIRv coupling is
    tightest under water-limited conditions.

Sensitivity test summary (for Methods section)
------------------------------------------------
Three training configurations were compared on the validation set:
  Original  (patience=15, raw ACC)     : best epoch 7, ACC=0.5813
  pat25     (patience=25, raw ACC)     : best epoch 7, ACC=0.5810
  3-epoch   (patience=15, smooth ACC) : best epoch 14, ACC=0.5793
All three converge to ACC in the band 0.5793-0.5813 (<0.002 range),
confirming the result is insensitive to the early stopping criterion.
The original model (epoch 7, ACC=0.5813) is used for all evaluations.
"""

import os
import json
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from pathlib import Path
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS

# ── Import model from 10_model.py ──────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "model_module", Path(__file__).parent / "10_model.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
DroughtForecastModel = _mod.DroughtForecastModel

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8   # larger batch OK at inference — no gradient storage
STRIDE     = 8
TRAIN_END  = pd.Timestamp("2021-12-31")
VAL_END    = pd.Timestamp("2022-12-31")

out_dir     = Path(PROC_ROOT)
#metrics_dir = Path("outputs/metrics")
metrics_dir = Path("outputs/metrics") / ARM.tag

metrics_dir.mkdir(parents=True, exist_ok=True)

print("Device          : {}".format(DEVICE))

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data ...")
#ds_rzsm   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
#rzsm_arr  = np.nan_to_num(
#    ds_rzsm["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)

# was:
#ds_rzsm   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
#rzsm_arr  = np.nan_to_num(ds_rzsm["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)

# with:
from arm_config import resolve_arm, apply_zeroing
ARM = resolve_arm()
ds_rzsm   = xr.open_zarr(out_dir / ARM.store, consolidated=True)
rzsm_arr  = np.nan_to_num(ds_rzsm[ARM.var].values.astype(np.float32), nan=0.0)
ARM.check_input(rzsm_arr, out_dir)
rzsm_arr  = apply_zeroing(rzsm_arr, ARM)



nirv_arr  = np.nan_to_num(
    ds_rzsm["nirv_anom"].values.astype(np.float32), nan=0.0)
times     = pd.DatetimeIndex(ds_rzsm.time.values)
T, H, W   = rzsm_arr.shape

ds_master  = xr.open_zarr(out_dir / "master_dataset.zarr", consolidated=True)
irr_frac   = ds_master["irr_frac"].values.astype(np.float32)

telecon_df  = pd.read_csv(
    out_dir / "teleconnection_indices_daily.csv",
    index_col=0, parse_dates=True)
telecon_arr = telecon_df.reindex(times, method="ffill").fillna(0.0)[
    ["Nino34", "AMO", "PDO"]].values.astype(np.float32)

domain_npz = np.load(out_dir / "domain_masks.npz")
masks      = {k: domain_npz[k].astype(bool) for k in domain_npz.files}
wlc        = masks["water_limited_crop"]
print("  Test period: {} to {}".format(
    str(times[times > VAL_END][0])[:10],
    str(times[-1])[:10]))
print("  WLC pixels : {:,}".format(int(wlc.sum())))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Dataset — test split only
# ═══════════════════════════════════════════════════════════════════════════
class DroughtForecastDataset(Dataset):
    def __init__(self, rzsm, nirv, irr, telecon, mask, times,
                 context_len, lead_days, stride, split, train_end, val_end):
        self.rzsm = rzsm; self.nirv = nirv
        self.irr  = irr[None]; self.telecon = telecon; self.mask = mask
        self.ctx  = context_len; self.leads = np.array(lead_days)
        self.max_lead = max(lead_days)
        all_t0 = np.arange(context_len, len(times) - self.max_lead, stride)
        if split == "train":
            self.starts = all_t0[times[all_t0] <= train_end]
        elif split == "val":
            self.starts = all_t0[(times[all_t0] > train_end) &
                                   (times[all_t0] <= val_end)]
        else:
            self.starts = all_t0[times[all_t0] > val_end]

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        return {
            "rzsm":   torch.from_numpy(self.rzsm[t0-self.ctx:t0][:, None]),
            "nirv":   torch.from_numpy(self.nirv[t0-self.ctx:t0][:, None]),
            "irr":    torch.from_numpy(self.irr),
            "tc":     torch.from_numpy(self.telecon[t0-self.ctx:t0]),
            "target": torch.from_numpy(
                np.stack([self.nirv[t0 + lead - 1] for lead in self.leads])),
            "mask":   torch.from_numpy(self.mask),
            "t0":     torch.tensor(t0),
        }


kw = dict(rzsm=rzsm_arr, nirv=nirv_arr, irr=irr_frac, telecon=telecon_arr,
          mask=wlc, times=times, context_len=CONTEXT_LEN,
          lead_days=LEAD_DAYS, stride=STRIDE,
          train_end=TRAIN_END, val_end=VAL_END)

test_ds     = DroughtForecastDataset(**kw, split="test")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)
print("  Test samples: {}  ({} to {})".format(
    len(test_ds),
    str(times[test_ds.starts[0]])[:10],
    str(times[test_ds.starts[-1]])[:10]))

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load best checkpoint (epoch 7, val ACC=0.5813)
# ═══════════════════════════════════════════════════════════════════════════

ckpt_path = Path("models") / ARM.tag / "best_convlstm_conus.pt"      # ← READ
assert ARM.tag in str(ckpt_path), f"checkpoint path not arm-specific: {ckpt_path}"
#ckpt_path = Path("models/best_convlstm_conus.pt")
assert ckpt_path.exists(), "Checkpoint not found: {ckpt_path}. Run Step 11 for arm {ARM.name}."


ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
# use_checkpoint=False at inference: no gradient storage needed
model = DroughtForecastModel(use_checkpoint=False).to(DEVICE)

assert ckpt.get("arm") == ARM.name, \
    f"checkpoint was trained as arm {ckpt.get('arm')!r} but is being evaluated as {ARM.name!r}"
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Checkpoint      : epoch {}  val ACC={:.4f}".format(
    ckpt["epoch"], ckpt["val_acc"]))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Run inference on test set
# ═══════════════════════════════════════════════════════════════════════════
print("\nRunning inference ...")
all_preds, all_targets, all_t0s = [], [], []

with torch.no_grad():
    for batch in test_loader:
        with autocast("cuda" if DEVICE.type == "cuda" else "cpu"):
            pred = model(
                batch["rzsm"].to(DEVICE),
                batch["nirv"].to(DEVICE),
                batch["irr"].to(DEVICE),
                batch["tc"].to(DEVICE),
            ).cpu()
        # .float() converts bfloat16 (A100 AMP default) to float32
        # before .numpy() — numpy does not support bfloat16
        all_preds.append(pred.float().numpy())
        all_targets.append(batch["target"].float().numpy())
        all_t0s.extend(batch["t0"].tolist())

all_preds   = np.concatenate(all_preds,   axis=0)   # (N, n_leads, H, W)
all_targets = np.concatenate(all_targets, axis=0)
N           = all_preds.shape[0]
print("  Predictions : {}".format(all_preds.shape))

# ═══════════════════════════════════════════════════════════════════════════
# 5. Baseline predictions
# ═══════════════════════════════════════════════════════════════════════════
clim_preds   = np.zeros_like(all_preds)    # climatology: zero anomaly

persist_preds = np.zeros_like(all_preds)   # persistence: last observed NIRv
for i, t0 in enumerate(all_t0s):
    nirv_at_t0 = nirv_arr[t0 - 1]         # last value before forecast window
    for li in range(len(LEAD_DAYS)):
        persist_preds[i, li] = nirv_at_t0

# ═══════════════════════════════════════════════════════════════════════════
# 6. Metric functions
# ═══════════════════════════════════════════════════════════════════════════
def domain_acc(preds, targets, mask, lead_idx):
    p = preds[:, lead_idx][:, mask].ravel()
    t = targets[:, lead_idx][:, mask].ravel()
    fin = np.isfinite(p) & np.isfinite(t)
    if fin.sum() < 10:
        return np.nan
    r, _ = pearsonr(p[fin], t[fin])
    return float(r)

def domain_rmse(preds, targets, mask, lead_idx):
    p = preds[:, lead_idx][:, mask].ravel()
    t = targets[:, lead_idx][:, mask].ravel()
    return float(np.sqrt(np.nanmean((p - t) ** 2)))

def skill_score(rmse_model, rmse_base):
    return float(1.0 - rmse_model / max(rmse_base, 1e-8))

# ═══════════════════════════════════════════════════════════════════════════
# 7. Primary evaluation — water_limited_crop
# ═══════════════════════════════════════════════════════════════════════════
print("\nPrimary evaluation — water_limited_crop domain")
print("{:>6s} {:>7s} {:>7s} {:>8s} {:>8s} {:>8s}".format(
    "Lead", "ACC", "RMSE", "SS_clim", "ACC_pers", "RMSE_red%"))
print("-" * 55)

results = []
for li, lead in enumerate(LEAD_DAYS):
    acc_m   = domain_acc( all_preds,     all_targets, wlc, li)
    acc_p   = domain_acc( persist_preds, all_targets, wlc, li)
    rmse_m  = domain_rmse(all_preds,     all_targets, wlc, li)
    rmse_c  = domain_rmse(clim_preds,    all_targets, wlc, li)
    rmse_p  = domain_rmse(persist_preds, all_targets, wlc, li)
    ss_c    = skill_score(rmse_m, rmse_c)
    ss_p    = skill_score(rmse_m, rmse_p)
    red_pct = (1.0 - rmse_m / max(rmse_c, 1e-8)) * 100

    results.append({
        "lead_days":         lead,
        "ACC":               acc_m,
        "RMSE_model":        rmse_m,
        "RMSE_clim":         rmse_c,
        "RMSE_pers":         rmse_p,
        "SS_climatology":    ss_c,
        "SS_persistence":    ss_p,
        "ACC_persistence":   acc_p,
        "RMSE_reduction_pct": red_pct,
    })
    print("{:>5d}d {:>7.3f} {:>7.3f} {:>8.3f} {:>8.3f} {:>7.1f}%".format(
        lead, acc_m, rmse_m, ss_c, acc_p, red_pct))

df_results = pd.DataFrame(results)
df_results.to_csv(metrics_dir / "evaluation_summary.csv", index=False)
print("Mean ACC (all leads): {:.3f}".format(df_results["ACC"].mean()))
print("Africa benchmark    : 0.555  (concurrent RZSM, domain-averaged)")

# ═══════════════════════════════════════════════════════════════════════════
# 8. Stratified evaluation
# ═══════════════════════════════════════════════════════════════════════════
print("\nStratified evaluation (mean ACC across all leads):")
strat_masks = {
    "water_limited_crop":  masks.get("water_limited_crop",  wlc),
    "rainfed_crop":        masks.get("rainfed_crop",        wlc),
    "irrigated_crop":      masks.get("irrigated_crop",      wlc),
    "semi_arid_crop":      masks.get("semi_arid_crop",      wlc),
    "energy_limited_crop": masks.get("energy_limited_crop", wlc),
}
zone_masks = {k: v for k, v in masks.items() if k.startswith("zone_")}
strat_masks.update(zone_masks)

zone_rows = []
for mname, m in strat_masks.items():
    if int(m.sum()) < 5:
        continue
    accs    = [domain_acc(all_preds, all_targets, m, li)
               for li in range(len(LEAD_DAYS))]
    mean_acc = float(np.nanmean(accs))
    zone_rows.append({
        "domain":    mname,
        "n_pixels":  int(m.sum()),
        "ACC_mean":  mean_acc,
        "ACC_8d":    accs[0],
        "ACC_16d":   accs[1],
        "ACC_24d":   accs[2],
        "ACC_32d":   accs[3],
        "ACC_40d":   accs[4],
    })
    print("  {:35s}: mean={:.3f}  8d={:.3f}  40d={:.3f}  ({:,}px)".format(
        mname, mean_acc, accs[0], accs[4], int(m.sum())))

df_zones = pd.DataFrame(zone_rows)
df_zones.to_csv(metrics_dir / "evaluation_by_zone.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 9. QC plot — three panels
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ref_lons = ds_rzsm.lon.values
ref_lats = ds_rzsm.lat.values
extent   = [float(ref_lons[0]), float(ref_lons[-1]),
            float(ref_lats[-1]), float(ref_lats[0])]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: ACC vs lead time
accs_m = [domain_acc(all_preds,     all_targets, wlc, li) for li in range(len(LEAD_DAYS))]
accs_p = [domain_acc(persist_preds, all_targets, wlc, li) for li in range(len(LEAD_DAYS))]
ax = axes[0]
ax.plot(LEAD_DAYS, accs_m, "o-",  color="#185FA5", lw=2, ms=7,
        label="ConvLSTM (this study)")
ax.plot(LEAD_DAYS, accs_p, "s--", color="#BA7517", lw=1.5, ms=6,
        label="Persistence baseline")
ax.axhline(0.555, color="gray", lw=1.0, ls=":", label="Africa benchmark (0.555)")
ax.axhline(0, color="lightgray", lw=0.8)
ax.set_xlabel("Lead time (days)"); ax.set_ylabel("ACC (Pearson r)")
ax.set_title("ACC vs lead time\nwater-limited cropland, 2023-2024")
ax.legend(fontsize=9); ax.set_ylim(-0.1, 1.0); ax.set_xticks(LEAD_DAYS)

# Panel 2: RMSE reduction % vs climatology
rmse_red = [r["RMSE_reduction_pct"] for r in results]
ax = axes[1]
bars = ax.bar(LEAD_DAYS, rmse_red, width=5, color="#2d6a0f", alpha=0.8,
              label="This study")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Lead time (days)")
ax.set_ylabel("RMSE reduction vs climatology (%)")
ax.set_title("Skill score (RMSE reduction)\nvs climatological baseline")
ax.set_xticks(LEAD_DAYS)
for bar, v in zip(bars, rmse_red):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            "{:.1f}%".format(v), ha="center", va="bottom", fontsize=8)

# Panel 3: Pixel-level ACC at 40-day lead (vectorized — no pixel loop)
li40 = len(LEAD_DAYS) - 1
p_all = all_preds[:, li40][:, wlc]    # (N, n_wlc)
t_all = all_targets[:, li40][:, wlc]  # (N, n_wlc)
p_c   = p_all - p_all.mean(axis=0)
t_c   = t_all - t_all.mean(axis=0)
denom = np.sqrt((p_c**2).sum(axis=0) * (t_c**2).sum(axis=0)) + 1e-8
r_vals = (p_c * t_c).sum(axis=0) / denom
px_acc = np.full((H, W), np.nan)
px_acc[wlc] = r_vals

ax  = axes[2]
im  = ax.imshow(px_acc, extent=extent, origin="upper",
                cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="ACC (Pearson r)", shrink=0.8)
ax.set_title("Pixel-level ACC — 40-day lead\nwater-limited cropland (test 2023-2024)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

plt.suptitle(
    "Step 12 Evaluation — ConvLSTM CONUS Drought Forecast  "
    "(epoch 7, val ACC=0.5813)",
    fontsize=11)
plt.tight_layout()
plt.savefig(metrics_dir / "qc_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 10. Final summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Step 12 complete ──")
print("\nTest-set results (2023-2024) for paper Table 1:")
print("{:>8s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
    "Lead", "ACC", "RMSE_red%", "SS_clim", "SS_pers"))
for r in results:
    print("{:>7d}d {:>8.3f} {:>9.1f}% {:>10.3f} {:>10.3f}".format(
        r["lead_days"], r["ACC"], r["RMSE_reduction_pct"],
        r["SS_climatology"], r["SS_persistence"]))

mean_acc = df_results["ACC"].mean()
print("\nDomain-averaged ACC (all leads): {:.3f}".format(mean_acc))
print("Africa paper benchmark         : 0.555")
print("Difference                     : {:+.3f}".format(mean_acc - 0.555))

print("\nOutputs saved:")
for f in ["evaluation_summary.csv", "evaluation_by_zone.csv", "qc_evaluation.png"]:
    p = metrics_dir / f
    print("  {} {}".format("OK" if p.exists() else "MISSING", p))
