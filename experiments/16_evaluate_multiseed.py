#!/usr/bin/env python3
"""
16_evaluate_multiseed.py
========================
Priority 2 — Multi-seed test set evaluation.

Loads all 15 seed checkpoints from Step 15, runs inference on the
held-out test set (2023-2024) for each seed, and reports ACC as
mean ± SD — the uncertainty-quantified final result for the paper.

Outputs
-------
  outputs/metrics/multiseed_test_results.csv
  outputs/metrics/multiseed_test_summary.json
"""

import os, json, importlib.util
import numpy as np, pandas as pd, xarray as xr
import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from pathlib import Path
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS, SEEDS
from arm_config import resolve_arm, apply_zeroing
ARM = resolve_arm()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
STRIDE     = 8
TRAIN_END  = pd.Timestamp("2021-12-31")
VAL_END    = pd.Timestamp("2022-12-31")
out_dir    = Path(PROC_ROOT)
#metrics_dir = Path("outputs/metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_dir = Path("outputs/metrics") / ARM.tag; metrics_dir.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "model_module", Path(__file__).parent / "10_model.py")
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)
DroughtForecastModel = _mod.DroughtForecastModel

print("Step 16: Multi-seed test set evaluation")



#ds_rzsm   = xr.open_zarr(out_dir / ARM.store, consolidated=True)
#rzsm_arr  = np.nan_to_num(ds_rzsm[ARM.var].values.astype(np.float32), nan=0.0)

ds_rzsm   = xr.open_zarr(out_dir / ARM.store, consolidated=True)
# Capture SMAP validity BEFORE nan_to_num: 15 domain cells over the Great Salt
# Lake, Chesapeake Bay and a Texas coastal lagoon have no SMAP L4 data at any
# timestep. nan_to_num would give them a zero anomaly, which reads as "exactly
# average soil moisture" rather than "no data", while their NIRv targets would
# still be scored. They are excluded from evaluation.
_raw      = ds_rzsm[ARM.var].values.astype(np.float32)
SMAP_VALID = np.isfinite(_raw).all(axis=0)
rzsm_arr  = np.nan_to_num(_raw, nan=0.0)
del _raw


ARM.check_input(rzsm_arr, out_dir)
rzsm_arr  = apply_zeroing(rzsm_arr, ARM)



nirv_arr  = np.nan_to_num(ds_rzsm["nirv_anom"].values.astype(np.float32), nan=0.0)
times     = pd.DatetimeIndex(ds_rzsm.time.values)
ds_master = xr.open_zarr(out_dir/"master_dataset.zarr", consolidated=True)
irr_frac  = ds_master["irr_frac"].values.astype(np.float32)
telecon_df = pd.read_csv(out_dir/"teleconnection_indices_daily.csv",
                          index_col=0, parse_dates=True)
telecon_arr = telecon_df.reindex(times, method="ffill").fillna(0.0)[
    ["Nino34","AMO","PDO"]].values.astype(np.float32)
domain_npz = np.load(out_dir/"domain_masks.npz")
wlc        = domain_npz["water_limited_crop"].astype(bool)
_n_before  = int(wlc.sum())
wlc        = wlc & SMAP_VALID
print(f"  domain: {_n_before} cells, {_n_before - int(wlc.sum())} dropped for "
      f"missing SMAP data -> {int(wlc.sum())} evaluated")

class TestDataset(Dataset):
    def __init__(self):
        self.rzsm=rzsm_arr; self.nirv=nirv_arr
        self.irr=irr_frac[None]; self.telecon=telecon_arr; self.mask=wlc
        self.leads=np.array(LEAD_DAYS); self.max_lead=max(LEAD_DAYS)
        all_t0=np.arange(CONTEXT_LEN, len(times)-self.max_lead, STRIDE)
        self.starts=all_t0[times[all_t0]>VAL_END]
    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        t0=self.starts[idx]
        return {"rzsm":   torch.from_numpy(self.rzsm[t0-CONTEXT_LEN:t0][:,None]),
                "nirv":   torch.from_numpy(self.nirv[t0-CONTEXT_LEN:t0][:,None]),
                "irr":    torch.from_numpy(self.irr),
                "tc":     torch.from_numpy(self.telecon[t0-CONTEXT_LEN:t0]),
                "target": torch.from_numpy(
                    np.stack([self.nirv[t0+lead-1] for lead in self.leads])),
                "mask":   torch.from_numpy(self.mask)}

test_ds     = TestDataset()
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)
print(f"  Test samples: {len(test_ds)}")

def domain_acc(preds, targets, mask, li):
    p=preds[:,li][:,mask].ravel(); t=targets[:,li][:,mask].ravel()
    r,_=pearsonr(p,t); return float(r)




# ── Stratified masks ───────────────────────────────────────────────────────
# The irr_frac bins test the dilution explanation directly. If dilution is why
# dACC is zero, it should grow monotonically with irrigated fraction: pixels at
# >= 0.50 are majority-irrigated and behave like a finer-resolution sample. If
# dACC is flat at zero even there, dilution is NOT the explanation and the
# correction does not matter at any fraction. The test can falsify our own
# proposed mechanism, which is why it is worth running.
_irr = irr_frac / 100.0 if float(np.nanmax(irr_frac)) > 1.5 else irr_frac
MASKS = {
    "all_wlc":       wlc,
    "irrigated":     wlc & (_irr >= 0.15),
    "rainfed":       wlc & (_irr <= 0.05),
    # dose-response on irrigation intensity
    "irr_015_025":   wlc & (_irr >= 0.15) & (_irr < 0.25),
    "irr_025_050":   wlc & (_irr >= 0.25) & (_irr < 0.50),
    "irr_050_plus":  wlc & (_irr >= 0.50),
}
# Regions where irrigation is concentrated, if the masks are on the grid

for _nm, _f in (("Southern_Great_Plains", "mask_Southern_Great_Plains_v2.npy"),
                ("Northern_Great_Plains",  "mask_Northern_Great_Plains_v2.npy")):
    _p = out_dir / _f
    if not _p.exists():
        print(f"  region {_nm}: {_f} not found, skipped")
        continue
    _m = np.load(_p) > 0.5
    if _m.shape != wlc.shape:
        print(f"  region {_nm}: shape {_m.shape} != grid {wlc.shape}, skipped")
        continue
    MASKS[f"reg_{_nm}"] = wlc & _m & (_irr >= 0.15)

print("\nStratified evaluation masks:")
for _k, _m in MASKS.items():
    print(f"  {_k:18s} {int(_m.sum()):5d} px")
strat_rows = []




# ── Evaluate each seed ─────────────────────────────────────────────────────

all_accs = {lead: [] for lead in LEAD_DAYS}
seed_summaries = []
for seed in SEEDS:
    #ckpt_path = Path(f"models/seed_{seed}_convlstm.pt")
    ckpt_path = Path("models") / ARM.tag / f"seed_{seed}_convlstm.pt"    # ← READ
    assert ARM.tag in str(ckpt_path), f"checkpoint path not arm-specific: {ckpt_path}"
    assert ckpt_path.exists(), f"Missing: {ckpt_path} — run Step 15 for arm {ARM.name} first"
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    assert ckpt.get("arm") == ARM.name, \
        f"checkpoint was trained as arm {ckpt.get('arm')!r} but is being evaluated as {ARM.name!r}"
    model = DroughtForecastModel(use_checkpoint=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"\nSeed {seed}: epoch {ckpt['epoch']}, val ACC={ckpt['val_acc']:.4f}")

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
                pred = model(batch["rzsm"].to(DEVICE), batch["nirv"].to(DEVICE),
                             batch["irr"].to(DEVICE), batch["tc"].to(DEVICE)).cpu()
            preds.append(pred.float().numpy())
            targets.append(batch["target"].float().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    del model; torch.cuda.empty_cache()
    for _mk, _mv in MASKS.items():
        if int(_mv.sum()) < 10:
            print(f"  skipping mask {_mk}: only {int(_mv.sum())} px")
            continue
        for _li, _lead in enumerate(LEAD_DAYS):
            strat_rows.append({
                "arm": ARM.name, "seed": seed, "lead": _lead, "mask": _mk,
                "n_pixels": int(_mv.sum()),
                "acc": domain_acc(preds, targets, _mv, _li),
            })


    accs_per_lead = [domain_acc(preds, targets, wlc, li)
                     for li in range(len(LEAD_DAYS))]
    mean_acc = float(np.mean(accs_per_lead))
    for li, lead in enumerate(LEAD_DAYS):
        all_accs[lead].append(accs_per_lead[li])

    seed_summaries.append({"seed":seed, "val_acc":ckpt["val_acc"],
                           "test_acc_mean":mean_acc,
                           **{f"acc_{lead}d":accs_per_lead[i]
                              for i,lead in enumerate(LEAD_DAYS)}})
    print(f"  Test ACC: {mean_acc:.4f} | "
          + " | ".join(f"{l}d={v:.3f}" for l,v in zip(LEAD_DAYS,accs_per_lead)))

# ── Final uncertainty-quantified results ──────────────────────────────────
print(f"\n{'='*65}")
print(f"FINAL RESULTS: mean ± SD across {len(SEEDS)} seeds")
print(f"{'='*65}")
print(f"{'Lead':>6s} {'Mean ACC':>10s} {'SD':>8s} {'Min':>8s} {'Max':>8s}")
print("-"*65)
rows = []
all_mean_accs = [r["test_acc_mean"] for r in seed_summaries]
for li, lead in enumerate(LEAD_DAYS):
    vals = all_accs[lead]
    rows.append({"lead_days":lead, "mean_acc":np.mean(vals),
                 "sd_acc":np.std(vals), "min_acc":min(vals), "max_acc":max(vals)})
    print(f"{lead:>5d}d {np.mean(vals):>10.3f} {np.std(vals):>8.4f} "
          f"{min(vals):>8.3f} {max(vals):>8.3f}")
print("-"*65)
print(f"{'Mean':>6s} {np.mean(all_mean_accs):>10.3f} "
      f"{np.std(all_mean_accs):>8.4f}")
print(f"{'='*65}")
print(f"\nPaper result: domain-averaged ACC = "
      f"{np.mean(all_mean_accs):.3f} ± {np.std(all_mean_accs):.3f}")

pd.DataFrame(rows).to_csv(metrics_dir/"multiseed_test_results.csv", index=False)
pd.DataFrame(strat_rows).to_csv(metrics_dir / "multiseed_stratified.csv", index=False)
print(f"Wrote {metrics_dir/'multiseed_stratified.csv'}  ({len(strat_rows)} rows)")
summary = {"seeds":SEEDS, "per_seed":seed_summaries,
           "final_mean_acc":float(np.mean(all_mean_accs)),
           "final_std_acc": float(np.std(all_mean_accs))}
with open(metrics_dir/"multiseed_test_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {metrics_dir}/multiseed_test_results.csv")
print(f"Saved: {metrics_dir}/multiseed_test_summary.json")
