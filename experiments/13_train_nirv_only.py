#!/usr/bin/env python3
"""
13_train_nirv_only.py
=====================
Priority 1 — NIRv-only ablation baseline.

Trains an identical ConvLSTM architecture using ONLY the NIRv anomaly
sequence as input (C=1, no RZSM, no irrigation fraction, no teleconnections).

Scientific purpose
------------------
This ablation answers the reviewers' question:
  "Does RZSM + irrigation deconvolution + teleconnections add skill beyond
   NIRv autocorrelation alone?"

If full model ACC (0.507) significantly exceeds NIRv-only ACC, the RZSM
and irrigation deconvolution contributions are validated.

The expected result (from Africa paper Adebayo & Nakalembe, preprint):
  NIRv-only ACC ≈ 0.478–0.490 at 40-day lead (domain-averaged)
  Full model advantage ≈ +0.02 to +0.05 ACC at 40-day lead

Architecture change
-------------------
  Full model : input_dim = 26 (RZSM + NIRv + irr(8ch) + telecon(16ch))
  NIRv-only  : input_dim =  1 (NIRv anomaly only)
  All other params identical: hidden=64, kernel=3×3, 2 layers, 5 heads

Outputs
-------
  models/best_nirv_only.pt
  outputs/metrics/training_history_nirv_only.json
"""

import os, sys, json, importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from config import PROC_ROOT, CONTEXT_LEN, LEAD_DAYS, LR, EPOCHS, HIDDEN_DIM, KERNEL_SIZE, NUM_LAYERS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
STRIDE     = 8
PATIENCE   = 15
TRAIN_END  = pd.Timestamp("2021-12-31")
VAL_END    = pd.Timestamp("2022-12-31")

out_dir     = Path(PROC_ROOT)
models_dir  = Path("models");          models_dir.mkdir(exist_ok=True)
metrics_dir = Path("outputs/metrics"); metrics_dir.mkdir(parents=True, exist_ok=True)

print("Step 13: NIRv-only ablation baseline")
print("Device  : {}".format(DEVICE))
print("Purpose : quantify RZSM + irrigation + telecon contribution to full model skill")

# ═══════════════════════════════════════════════════════════════════════════
# 1. NIRv-only model (identical architecture, input_dim=1)
# ═══════════════════════════════════════════════════════════════════════════
from torch.utils.checkpoint import checkpoint as grad_checkpoint

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, padding=pad)
        self.norm = nn.GroupNorm(4, 4 * hidden_dim)

    def forward(self, x, h, c):
        gates = self.norm(self.conv(torch.cat([x, h], dim=1)))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, B, H, W, device):
        z = torch.zeros(B, self.hidden_dim, H, W, device=device)
        return z, z.clone()


class NIRvOnlyModel(nn.Module):
    """
    Ablation model: NIRv sequence only (C=1).
    Identical ConvLSTM encoder and prediction heads as the full model.
    Removing RZSM, irrigation encoder, and teleconnection context.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE,
                 num_layers=NUM_LAYERS, lead_times=LEAD_DAYS,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.n_leads = len(lead_times)

        # Input = NIRv only (1 channel)
        dims = [1] + [hidden_dim] * num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(dims[i], hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),         nn.ReLU(),
                nn.Conv2d(16,  1, 1),
            )
            for _ in range(self.n_leads)
        ])

    def _run_cells(self, inp, *hidden_flat):
        new_flat = []
        for i, cell in enumerate(self.cells):
            h = hidden_flat[2*i]; c = hidden_flat[2*i+1]
            h_new, c_new = cell(inp, h, c)
            new_flat.extend([h_new, c_new])
            inp = h_new
        return tuple(new_flat)

    def forward(self, nirv, **kwargs):
        """
        nirv : (B, T, 1, H, W)   — only input used
        kwargs: ignored (rzsm, irr, tc) — allows same DataLoader as full model
        """
        B, T, _, H, W = nirv.shape
        device = nirv.device
        hidden = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        for t in range(T):
            inp = nirv[:, t]    # (B, 1, H, W)
            hidden_flat = tuple(x for h, c in hidden for x in (h, c))
            if self.use_checkpoint and self.training:
                new_flat = grad_checkpoint(self._run_cells, inp, *hidden_flat,
                                           use_reentrant=False)
            else:
                new_flat = self._run_cells(inp, *hidden_flat)
            hidden = [(new_flat[2*i], new_flat[2*i+1])
                      for i in range(len(self.cells))]

        h_final = hidden[-1][0]
        return torch.cat([head(h_final) for head in self.heads], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Data loading (same as Step 11 — only nirv_arr used for input)
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading data ...")
ds_corr   = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
nirv_arr  = np.nan_to_num(ds_corr["nirv_anom"].values.astype(np.float32), nan=0.0)
times     = pd.DatetimeIndex(ds_corr.time.values)
T, H, W   = nirv_arr.shape

domain_npz = np.load(out_dir / "domain_masks.npz")
wlc_mask   = domain_npz["water_limited_crop"].astype(bool)
print("  WLC pixels : {:,}".format(int(wlc_mask.sum())))


class NIRvOnlyDataset(Dataset):
    """Dataset with only NIRv input — no RZSM, irr, or telecon."""
    def __init__(self, nirv, mask, times, context_len, lead_days,
                 stride, split, train_end, val_end):
        self.nirv = nirv; self.mask = mask
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
            "nirv":   torch.from_numpy(self.nirv[t0-self.ctx:t0][:, None]),
            "target": torch.from_numpy(
                np.stack([self.nirv[t0+lead-1] for lead in self.leads])),
            "mask":   torch.from_numpy(self.mask),
        }


kw = dict(nirv=nirv_arr, mask=wlc_mask, times=times,
          context_len=CONTEXT_LEN, lead_days=LEAD_DAYS, stride=STRIDE,
          train_end=TRAIN_END, val_end=VAL_END)

train_ds = NIRvOnlyDataset(**kw, split="train")
val_ds   = NIRvOnlyDataset(**kw, split="val")
print("  Train/Val : {}/{}".format(len(train_ds), len(val_ds)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Model + training
# ═══════════════════════════════════════════════════════════════════════════
model     = NIRvOnlyModel(use_checkpoint=True).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = GradScaler("cuda" if DEVICE.type=="cuda" else "cpu")

n_params = sum(p.numel() for p in model.parameters())
print("\nNIRv-only model parameters: {:,}".format(n_params))
print("(Full model was 624,333 — difference reflects removed streams)")

def masked_mse(pred, target, mask):
    m = mask.unsqueeze(1).expand_as(pred)
    return ((pred - target)**2 * m).sum() / m.sum().clamp(min=1)

def acc_per_lead(pred, target, mask):
    accs = []; mask2 = mask[0].bool()
    for li in range(pred.shape[1]):
        p = pred[:, li][:, mask2].reshape(-1)
        t = target[:, li][:, mask2].reshape(-1)
        p_c = p - p.mean(); t_c = t - t.mean()
        accs.append(float((p_c*t_c).sum() / (p_c.norm()*t_c.norm()).clamp(min=1e-8)))
    return accs

best_val_acc = -1.0; patience_ctr = 0
history = {"train_loss":[],"val_loss":[],"val_acc_mean":[],"val_acc_per_lead":[],"lr":[]}

print("\nTraining NIRv-only model ({} epochs, patience={}) ...\n".format(EPOCHS, PATIENCE))

for epoch in range(1, EPOCHS+1):
    model.train(); train_losses = []
    for batch in tqdm(train_loader, desc="Ep {:03d} train".format(epoch),
                      leave=False, ncols=72):
        nirv   = batch["nirv"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        mask   = batch["mask"].to(DEVICE)
        optimizer.zero_grad()
        with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
            pred = model(nirv=nirv)
            loss = masked_mse(pred, target, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        train_losses.append(float(loss))

    model.eval(); val_losses = []; val_accs_all = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Ep {:03d} val  ".format(epoch),
                          leave=False, ncols=72):
            nirv   = batch["nirv"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            mask   = batch["mask"].to(DEVICE)
            with autocast("cuda" if DEVICE.type=="cuda" else "cpu"):
                pred = model(nirv=nirv)
                loss = masked_mse(pred, target, mask)
            val_losses.append(float(loss))
            val_accs_all.append(acc_per_lead(pred.cpu(), target.cpu(), mask.cpu()))

    scheduler.step()
    train_loss   = float(np.mean(train_losses))
    val_loss     = float(np.mean(val_losses))
    val_acc_per  = np.mean(val_accs_all, axis=0).tolist()
    val_acc_mean = float(np.mean(val_acc_per))
    lr_now       = float(scheduler.get_last_lr()[0])

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc_mean"].append(val_acc_mean)
    history["val_acc_per_lead"].append(val_acc_per)
    history["lr"].append(lr_now)

    print("Ep {:03d} | trn={:.4f} val={:.4f} | ACC={:.3f} "
          "[8d={:.3f} 40d={:.3f}] | lr={:.1e}".format(
        epoch, train_loss, val_loss, val_acc_mean,
        val_acc_per[0], val_acc_per[-1], lr_now))

    ckpt = {"epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_acc": val_acc_mean, "val_acc_per_lead": val_acc_per}
    torch.save(ckpt, models_dir / "last_nirv_only.pt")
    if val_acc_mean > best_val_acc:
        best_val_acc = val_acc_mean; patience_ctr = 0
        torch.save(ckpt, models_dir / "best_nirv_only.pt")
        print("  ✓ Best saved (ACC={:.4f})".format(best_val_acc))
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping at epoch {}".format(epoch)); break

with open(metrics_dir / "training_history_nirv_only.json", "w") as f:
    json.dump(history, f, indent=2)

best = torch.load(models_dir / "best_nirv_only.pt", map_location="cpu",
                  weights_only=False)
print("\nNIRv-only best epoch {} | ACC={:.4f} | per-lead={}".format(
    best["epoch"], best["val_acc"],
    [round(v,3) for v in best["val_acc_per_lead"]]))
print("\nNext: Step 14 (compare NIRv-only vs full model on test set)")
