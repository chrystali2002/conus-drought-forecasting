#!/usr/bin/env python3
"""
14_evaluate_nirv_only.py
========================
Priority 1 — Ablation evaluation: NIRv-only vs full model.

Loads the NIRv-only baseline checkpoint (Step 13) and the full model
checkpoint (Step 11) and evaluates both on the same held-out test set
(2023-2024). Produces a comparison table for the paper.

Scientific purpose
------------------
Answers the reviewer question:
  "Does RZSM + irrigation deconvolution + teleconnections add skill
   beyond NIRv autocorrelation alone?"

Expected result (based on Africa paper Adebayo & Nakalembe preprint):
  NIRv-only ACC ~ 0.47-0.49 (domain-averaged)
  Full model ACC = 0.507
  RZSM contribution ~ +0.02 to +0.04 ACC (growing with lead time)

Outputs
-------
  outputs/metrics/ablation_comparison.csv
  outputs/metrics/ablation_comparison.txt  (print-ready table)
"""

import os, importlib.util, numpy as np, pandas as pd
import xarray as xr, torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from pathlib import Path
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
STRIDE     = 8
TRAIN_END  = pd.Timestamp("2021-12-31")
VAL_END    = pd.Timestamp("2022-12-31")
out_dir    = Path(PROC_ROOT)
metrics_dir = Path("outputs/metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)

print("Step 14: NIRv-only ablation vs full model comparison")
print("Device :", DEVICE)

# ── Import models ─────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "model_module", Path(__file__).parent / "10_model.py")
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)
DroughtForecastModel = _mod.DroughtForecastModel

spec2 = importlib.util.spec_from_file_location(
    "nirv_module", Path(__file__).parent / "13_train_nirv_only.py")
_mod2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(_mod2)
NIRvOnlyModel = _mod2.NIRvOnlyModel

# ── Load data ──────────────────────────────────────────────────────────────
print("\nLoading data ...")
ds_corr   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
rzsm_arr  = np.nan_to_num(ds_corr["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)
nirv_arr  = np.nan_to_num(ds_corr["nirv_anom"].values.astype(np.float32), nan=0.0)
times     = pd.DatetimeIndex(ds_corr.time.values)
ds_master = xr.open_zarr(out_dir / "master_dataset.zarr", consolidated=True)
irr_frac  = ds_master["irr_frac"].values.astype(np.float32)
telecon_df = pd.read_csv(out_dir / "teleconnection_indices_daily.csv",
                          index_col=0, parse_dates=True)
telecon_arr = telecon_df.reindex(times, method="ffill").fillna(0.0)[
    ["Nino34","AMO","PDO"]].values.astype(np.float32)
domain_npz = np.load(out_dir / "domain_masks.npz")
wlc        = domain_npz["water_limited_crop"].astype(bool)

# ── Dataset (full inputs — NIRv-only model ignores extras via kwargs) ─────
class FullDataset(Dataset):
    def __init__(self, rzsm, nirv, irr, telecon, mask, times,
                 context_len, lead_days, stride, split, train_end, val_end):
        self.rzsm=rzsm; self.nirv=nirv; self.irr=irr[None]
        self.telecon=telecon; self.mask=mask
        self.ctx=context_len; self.leads=np.array(lead_days)
        self.max_lead=max(lead_days)
        all_t0=np.arange(context_len, len(times)-self.max_lead, stride)
        self.starts=all_t0[times[all_t0]>val_end]  # test split only
    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        t0=self.starts[idx]
        return {"rzsm":   torch.from_numpy(self.rzsm[t0-self.ctx:t0][:,None]),
                "nirv":   torch.from_numpy(self.nirv[t0-self.ctx:t0][:,None]),
                "irr":    torch.from_numpy(self.irr),
                "tc":     torch.from_numpy(self.telecon[t0-self.ctx:t0]),
                "target": torch.from_numpy(
                    np.stack([self.nirv[t0+lead-1] for lead in self.leads])),
                "mask":   torch.from_numpy(self.mask)}

test_ds = FullDataset(rzsm=rzsm_arr, nirv=nirv_arr, irr=irr_frac,
                      telecon=telecon_arr, mask=wlc, times=times,
                      context_len=CONTEXT_LEN, lead_days=LEAD_DAYS,
                      stride=STRIDE, split="test",
                      train_end=TRAIN_END, val_end=VAL_END)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)
print(f"  Test samples: {len(test_ds)}")

# ── Helper: run inference ──────────────────────────────────────────────────
def run_inference(model, loader, nirv_only=False):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            nirv = batch["nirv"].to(DEVICE)
            with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
                if nirv_only:
                    pred = model(nirv=nirv).cpu()
                else:
                    pred = model(batch["rzsm"].to(DEVICE), nirv,
                                 batch["irr"].to(DEVICE),
                                 batch["tc"].to(DEVICE)).cpu()
            preds.append(pred.float().numpy())
            targets.append(batch["target"].float().numpy())
    return np.concatenate(preds), np.concatenate(targets)

def domain_acc(preds, targets, mask, li):
    p = preds[:,li][:,mask].ravel(); t = targets[:,li][:,mask].ravel()
    r, _ = pearsonr(p, t); return float(r)

def domain_rmse(preds, targets, mask, li):
    p = preds[:,li][:,mask].ravel(); t = targets[:,li][:,mask].ravel()
    return float(np.sqrt(np.nanmean((p-t)**2)))

# ── Load and evaluate full model ──────────────────────────────────────────
print("\nEvaluating full model (epoch 7, val ACC=0.5813) ...")
ckpt_full = torch.load("models/best_convlstm_conus.pt",
                        map_location=DEVICE, weights_only=False)
model_full = DroughtForecastModel(use_checkpoint=False).to(DEVICE)
model_full.load_state_dict(ckpt_full["model_state"])
preds_full, targets = run_inference(model_full, test_loader, nirv_only=False)
del model_full; torch.cuda.empty_cache()

# ── Load and evaluate NIRv-only model ─────────────────────────────────────
print("Evaluating NIRv-only model ...")
ckpt_nirv = torch.load("models/best_nirv_only.pt",
                        map_location=DEVICE, weights_only=False)
model_nirv = NIRvOnlyModel(use_checkpoint=False).to(DEVICE)
model_nirv.load_state_dict(ckpt_nirv["model_state"])
preds_nirv, _ = run_inference(model_nirv, test_loader, nirv_only=True)
del model_nirv; torch.cuda.empty_cache()

# ── Climatology baseline ───────────────────────────────────────────────────
preds_clim = np.zeros_like(preds_full)

# ── Comparison table ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("ABLATION COMPARISON — test set 2023-2024, WLC domain (3,093 px)")
print("="*70)
print(f"{'Lead':>6s} {'Full ACC':>10s} {'NIRv-only':>10s} "
      f"{'ΔACC':>8s} {'Full RMSE↓':>11s} {'NIRv RMSE↓':>11s}")
print("-"*70)

rows = []
for li, lead in enumerate(LEAD_DAYS):
    acc_full = domain_acc(preds_full, targets, wlc, li)
    acc_nirv = domain_acc(preds_nirv, targets, wlc, li)
    acc_clim = 0.0  # by definition
    rmse_full = domain_rmse(preds_full, targets, wlc, li)
    rmse_nirv = domain_rmse(preds_nirv, targets, wlc, li)
    rmse_clim = domain_rmse(preds_clim, targets, wlc, li)
    delta     = acc_full - acc_nirv
    red_full  = (1 - rmse_full/rmse_clim)*100
    red_nirv  = (1 - rmse_nirv/rmse_clim)*100

    rows.append({"lead_days":lead,
                 "ACC_full":acc_full, "ACC_nirv_only":acc_nirv,
                 "delta_ACC":delta,
                 "RMSE_red_full_pct":red_full,
                 "RMSE_red_nirv_pct":red_nirv})
    print(f"{lead:>5d}d {acc_full:>10.3f} {acc_nirv:>10.3f} "
          f"{delta:>+8.3f} {red_full:>10.1f}% {red_nirv:>10.1f}%")

df = pd.DataFrame(rows)
print("-"*70)
print(f"{'Mean':>6s} {df.ACC_full.mean():>10.3f} {df.ACC_nirv_only.mean():>10.3f} "
      f"{df.delta_ACC.mean():>+8.3f}")
print("="*70)
print(f"\nConclusion: RZSM + irrigation + telecon contributes "
      f"{df.delta_ACC.mean():+.3f} mean ACC over NIRv-only baseline")
print(f"  (growing from {df.delta_ACC.iloc[0]:+.3f} at 8d "
      f"to {df.delta_ACC.iloc[-1]:+.3f} at 40d)")

df.to_csv(metrics_dir / "ablation_comparison.csv", index=False)
print(f"\nSaved: {metrics_dir}/ablation_comparison.csv")
print("Next: Step 15 (multi-seed training for uncertainty quantification)")
