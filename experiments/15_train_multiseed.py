#!/usr/bin/env python3
"""
15_train_multiseed.py
=====================
Priority 2 — Multi-seed training for uncertainty quantification.

Trains the full irrigation-aware ConvLSTM with 15 different random seeds.
Reports ACC as mean ± SD — standard requirement for deep learning
peer review in hydrology and remote sensing journals.

Why 3 seeds
-----------
With only 298 training samples, weight initialisation can affect the
final val ACC by ±0.003–0.008. Reporting a single seed result (0.507)
without uncertainty bounds is a common reviewer criticism. Three seeds
provides a stable estimate without excessive compute.

Outputs
-------
  models/seed_{0,1,2}_convlstm.pt     per-seed best checkpoints
  outputs/metrics/multiseed_results.json
"""

import os, sys, json, importlib.util
import numpy as np, pandas as pd, xarray as xr
import torch, torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS, LR, EPOCHS, HIDDEN_DIM, KERNEL_SIZE, NUM_LAYERS, SEEDS
from arm_config import resolve_arm, apply_zeroing
ARM = resolve_arm()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

spec = importlib.util.spec_from_file_location(
    "model_module", Path(__file__).parent / "10_model.py")
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)
DroughtForecastModel = _mod.DroughtForecastModel

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
STRIDE     = 8
PATIENCE   = 15
#SEEDS      = [42, 123, 777, 2024, 31337]
TRAIN_END  = pd.Timestamp("2021-12-31")
VAL_END    = pd.Timestamp("2022-12-31")
out_dir    = Path(PROC_ROOT)
#models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
#metrics_dir = Path("outputs/metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)

models_dir  = Path("models") / ARM.tag;          models_dir.mkdir(parents=True, exist_ok=True)
metrics_dir = Path("outputs/metrics") / ARM.tag; metrics_dir.mkdir(parents=True, exist_ok=True)

print(f"Step 15: Multi-seed training ({len(SEEDS)} seeds)")
print(f"Device : {DEVICE}")
print(f"Seeds  : {SEEDS}")

# ── Load data ──────────────────────────────────────────────────────────────
print("\nLoading data ...")
#ds_rzsm   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
#rzsm_arr  = np.nan_to_num(ds_rzsm["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)

# was:
#ds_rzsm   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
#rzsm_arr  = np.nan_to_num(ds_rzsm["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)

# with:
ds_rzsm   = xr.open_zarr(out_dir / ARM.store, consolidated=True)
rzsm_arr  = np.nan_to_num(ds_rzsm[ARM.var].values.astype(np.float32), nan=0.0)
ARM.check_input(rzsm_arr, out_dir)
rzsm_arr  = apply_zeroing(rzsm_arr, ARM)



nirv_arr  = np.nan_to_num(ds_rzsm["nirv_anom"].values.astype(np.float32), nan=0.0)
times     = pd.DatetimeIndex(ds_rzsm.time.values)
ds_master = xr.open_zarr(out_dir / "master_dataset.zarr", consolidated=True)
irr_frac  = ds_master["irr_frac"].values.astype(np.float32)
telecon_df = pd.read_csv(out_dir / "teleconnection_indices_daily.csv",
                          index_col=0, parse_dates=True)
telecon_arr = telecon_df.reindex(times, method="ffill").fillna(0.0)[
    ["Nino34","AMO","PDO"]].values.astype(np.float32)
domain_npz = np.load(out_dir / "domain_masks.npz")
wlc        = domain_npz["water_limited_crop"].astype(bool)

class DroughtDataset(Dataset):
    def __init__(self, rzsm, nirv, irr, telecon, mask, times,
                 context_len, lead_days, stride, split, train_end, val_end):
        self.rzsm=rzsm; self.nirv=nirv; self.irr=irr[None]
        self.telecon=telecon; self.mask=mask
        self.ctx=context_len; self.leads=np.array(lead_days)
        self.max_lead=max(lead_days)
        all_t0=np.arange(context_len, len(times)-self.max_lead, stride)
        if split=="train": self.starts=all_t0[times[all_t0]<=train_end]
        elif split=="val": self.starts=all_t0[(times[all_t0]>train_end)&(times[all_t0]<=val_end)]
        else:              self.starts=all_t0[times[all_t0]>val_end]
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

kw = dict(rzsm=rzsm_arr, nirv=nirv_arr, irr=irr_frac, telecon=telecon_arr,
          mask=wlc, times=times, context_len=CONTEXT_LEN,
          lead_days=LEAD_DAYS, stride=STRIDE,
          train_end=TRAIN_END, val_end=VAL_END)

def masked_mse(pred, target, mask):
    m = mask.unsqueeze(1).expand_as(pred)
    return ((pred-target)**2*m).sum()/m.sum().clamp(min=1)

def acc_mean(pred, target, mask):
    accs=[]; mask2=mask[0].bool()
    for li in range(pred.shape[1]):
        p=pred[:,li][:,mask2].reshape(-1); t=target[:,li][:,mask2].reshape(-1)
        p_c=p-p.mean(); t_c=t-t.mean()
        accs.append(float((p_c*t_c).sum()/(p_c.norm()*t_c.norm()).clamp(min=1e-8)))
    return float(np.mean(accs)), accs

# ── Train each seed ────────────────────────────────────────────────────────

all_results = []

for seed in SEEDS:
    ckpt_out = models_dir / f"seed_{seed}_convlstm.pt"
    if ckpt_out.exists() and not os.environ.get("FORCE_RETRAIN"):
        _c = torch.load(ckpt_out, map_location="cpu", weights_only=False)
        print(f"  seed {seed}: checkpoint exists (epoch {_c['epoch']}, "
              f"ACC={_c['val_acc']:.4f}) — skipping")
        all_results.append({"seed": seed, "best_epoch": _c["epoch"],
                            "val_acc": _c["val_acc"]})
        continue

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed)

    np.random.seed(seed)

    train_ds = DroughtDataset(**kw, split="train")
    val_ds   = DroughtDataset(**kw, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    model     = DroughtForecastModel(use_checkpoint=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler("cuda" if DEVICE.type=="cuda" else "cpu")

    best_acc = -1.0; patience_ctr = 0; best_ckpt = None

    for epoch in range(1, EPOCHS+1):
        model.train(); tlosses=[]
        for batch in tqdm(train_loader, desc=f"S{seed} Ep{epoch:03d} trn",
                          leave=False, ncols=70):
            optimizer.zero_grad()
            with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
                pred = model(batch["rzsm"].to(DEVICE), batch["nirv"].to(DEVICE),
                             batch["irr"].to(DEVICE), batch["tc"].to(DEVICE))
                loss = masked_mse(pred, batch["target"].to(DEVICE),
                                  batch["mask"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tlosses.append(float(loss))

        model.eval(); vaccs=[]
        with torch.no_grad():
            for batch in val_loader:
                with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
                    pred = model(batch["rzsm"].to(DEVICE), batch["nirv"].to(DEVICE),
                                 batch["irr"].to(DEVICE), batch["tc"].to(DEVICE))
                am, per = acc_mean(pred.cpu(), batch["target"], batch["mask"])
                vaccs.append(am)

        scheduler.step()
        val_acc = float(np.mean(vaccs))
        print(f"  Ep {epoch:03d} | trn={np.mean(tlosses):.4f} | "
              f"val ACC={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.1e}")

        if val_acc > best_acc:
            best_acc = val_acc; patience_ctr = 0
            best_ckpt = {"epoch":epoch, "model_state":model.state_dict(),
                         "arm": ARM.name,
                         "val_acc":val_acc, "seed":seed}
            torch.save(best_ckpt, models_dir/f"seed_{seed}_convlstm.pt")
            print(f"    ✓ Best (seed {seed}): epoch {epoch}, ACC={best_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    all_results.append({"seed":seed, "best_epoch":best_ckpt["epoch"],
                         "val_acc":best_acc})
    print(f"Seed {seed} done: epoch {best_ckpt['epoch']}, val ACC={best_acc:.4f}")




# ── Summary ────────────────────────────────────────────────────────────────
accs = [r["val_acc"] for r in all_results]
print(f"\n{'='*60}")
print(f"MULTI-SEED SUMMARY ({len(SEEDS)} seeds)")
print(f"{'='*60}")
for r in all_results:
    print(f"  Seed {r['seed']:4d}: epoch {r['best_epoch']:3d} | val ACC = {r['val_acc']:.4f}")
print(f"  Mean ± SD : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  Range     : {min(accs):.4f} – {max(accs):.4f}")

summary = {"seeds": SEEDS, "results": all_results,
           "mean_val_acc": float(np.mean(accs)),
           "std_val_acc":  float(np.std(accs)),
           "min_val_acc":  float(min(accs)),
           "max_val_acc":  float(max(accs))}
with open(metrics_dir/"multiseed_results.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {metrics_dir}/multiseed_results.json")
print("Next: Step 16 (multi-seed test set evaluation)")
