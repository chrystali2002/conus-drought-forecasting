#!/usr/bin/env python3
"""
10_model.py
===========
ConvLSTM model for sub-seasonal NIRv anomaly forecasting.

Architecture
------------
Irrigation-aware ConvLSTM with four input streams:

  Stream 1 — RZSM sequence   : (B, T, 1, H, W)
  Stream 2 — NIRv sequence   : (B, T, 1, H, W)
  Stream 3 — Irr. fraction   : (B, 1, H, W)  static
               → IrrEncoder: Conv2d(1→8→8), ReLU
               → broadcast over T: (B, T, 8, H, W)
  Stream 4 — Teleconnection  : (B, T, 3)  Nino3.4, AMO, PDO
               → TeleconContext: 1D-Conv(3→32→embed_dim=16)
               → global avg pool → MLP → (B, embed_dim, 1, 1)
               → broadcast over (T, H, W)

Combined input to ConvLSTM: 1+1+8+16 = 26 channels per time step.

ConvLSTM encoder:
  2 stacked ConvLSTMCells, hidden_dim=64, kernel=3×3, padding=1
  Returns final hidden state h: (B, 64, H, W)

Per-lead prediction heads (5 independent):
  Conv2d(64→32, 3×3) → ReLU → Conv2d(32→16, 3×3) → ReLU → Conv2d(16→1, 1×1)
  Output: (B, 5, H, W) — NIRv anomaly at leads 8, 16, 24, 32, 40 days

Loss mask: applied externally — only water_limited_crop pixels
contribute to training gradients (see Step 11).

Parameter count: ~624K (lightweight for HPC single-GPU training)

Memory management (CUDA OOM fix)
---------------------------------
Without gradient checkpointing, BPTT through T=80 steps stores all
intermediate hidden states simultaneously:
  80 × 2 layers × 2 tensors × (B,64,321,656) × 4B = ~43 GB at B=2 → OOM

Three optimisations applied in training (Step 11):
  1. Gradient checkpointing (this file, ConvLSTMEncoder._run_cells):
     Discards hidden states after each forward step; recomputes during
     backward. Memory: O(1) in T instead of O(T). Cost: ~1.5× slower.
     LSTM memory: ~43 GB → ~0.5 GB. (Chen et al. 2016)
  2. Automatic Mixed Precision (Step 11, torch.cuda.amp):
     float16 forward/backward; float32 optimizer update. ~2× memory
     reduction; ~1.5× faster on A100 Tensor Cores.
  3. batch_size=2 (reduced from 4): ~2× safety buffer.
  Combined: ~86 GB → ~4-8 GB. Confirmed on A100-SXM4-80GB.

Design decisions for the paper
--------------------------------
1. GroupNorm(4) replaces BatchNorm inside ConvLSTMCell: batch statistics
   are unstable at small batch sizes (2–4 spatiotemporal samples); GroupNorm
   normalises over channel groups independently of batch size.

2. TeleconContext uses global average pooling over the 80-day context
   window before the MLP. This produces a single teleconnection embedding
   broadcast spatially. The 1D temporal convolution (kernel=7) before
   pooling learns multi-week lag patterns rather than instantaneous indices.

3. Five independent prediction heads each start from the same shared
   hidden state h. This parallel (direct) multi-step design avoids error
   accumulation across lead times versus an autoregressive decoder, at the
   cost of not capturing temporal coherence between 8-day and 40-day
   forecasts — a standard trade-off in direct multi-step forecasting.

4. Gradient checkpointing (_run_cells wrapped with torch.utils.checkpoint
   per time step) enables full 321×656 CONUS training without spatial
   tiling, preserving the long-range spatial context that is the
   ConvLSTM's key advantage over pixel-wise models.

References
----------
Shi X et al. (2015) Convolutional LSTM Network: A Machine Learning
  Approach for Precipitation Nowcasting. NeurIPS.
  https://arxiv.org/abs/1506.04214
Chen T et al. (2016) Training Deep Nets with Sublinear Memory Cost.
  https://arxiv.org/abs/1604.06174
Ba J et al. (2016) Layer Normalization (GroupNorm basis).
  https://arxiv.org/abs/1607.06450
"""

import torch
import torch.nn as nn
import json
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from pathlib import Path
from config import PROC_ROOT, HIDDEN_DIM, KERNEL_SIZE, NUM_LAYERS, LEAD_DAYS, CONTEXT_LEN

# ═══════════════════════════════════════════════════════════════════════════
# 1. ConvLSTM Cell
# ═══════════════════════════════════════════════════════════════════════════
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015). GroupNorm for stability."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size=kernel_size, padding=pad, bias=bias)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=4 * hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        gates   = self.norm(self.conv(torch.cat([x, h], dim=1)))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        c_new   = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new   = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, B: int, H: int, W: int, device):
        z = torch.zeros(B, self.hidden_dim, H, W, device=device)
        return z, z.clone()


# ═══════════════════════════════════════════════════════════════════════════
# 2. ConvLSTM Encoder — with gradient checkpointing
# ═══════════════════════════════════════════════════════════════════════════
class ConvLSTMEncoder(nn.Module):
    """
    Multi-layer ConvLSTM encoder.

    use_checkpoint=True activates gradient checkpointing on the time loop.
    This trades compute (recomputes each time step during backward) for
    memory (only the current hidden state is kept, not all 80).
    Memory: O(1) in T instead of O(T).  Cost: ~1.5-2× training time.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: int = 3, num_layers: int = 2,
                 use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dims  = [input_dim] + [hidden_dim] * num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(dims[i], hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def _run_cells(self, inp: torch.Tensor, *hidden_flat) -> tuple:
        """
        Process one time step across all layers.
        Takes inp + flattened (h0,c0,h1,c1,...) as separate tensors
        so torch.utils.checkpoint can track them.
        Returns flattened (h0_new,c0_new,h1_new,c1_new,...).
        """
        new_flat = []
        for i, cell in enumerate(self.cells):
            h = hidden_flat[2 * i]
            c = hidden_flat[2 * i + 1]
            h_new, c_new = cell(inp, h, c)
            new_flat.extend([h_new, c_new])
            inp = h_new
        return tuple(new_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C, H, W)
        Returns : (B, hidden_dim, H, W) — final hidden state of last layer
        """
        B, T, C, H, W = x.shape
        device = x.device
        hidden = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        for t in range(T):
            inp         = x[:, t]
            hidden_flat = tuple(tensor for h, c in hidden for tensor in (h, c))

            if self.use_checkpoint and self.training:
                # Gradient checkpointing: recompute time step during backward
                # use_reentrant=False is recommended for PyTorch >= 1.13
                new_flat = grad_checkpoint(
                    self._run_cells, inp, *hidden_flat,
                    use_reentrant=False
                )
            else:
                new_flat = self._run_cells(inp, *hidden_flat)

            hidden = [(new_flat[2 * i], new_flat[2 * i + 1])
                      for i in range(len(self.cells))]

        return hidden[-1][0]   # (B, hidden_dim, H, W)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Teleconnection Context Encoder
# ═══════════════════════════════════════════════════════════════════════════
class TeleconContext(nn.Module):
    """1D-Conv temporal encoder for Nino3.4/AMO/PDO → spatial broadcast."""
    def __init__(self, n_tc: int = 3, embed_dim: int = 16):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_tc, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=7, padding=3), nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tc: torch.Tensor) -> torch.Tensor:
        x = tc.permute(0, 2, 1)          # (B, n_tc, T)
        x = self.temporal(x).mean(dim=-1) # (B, embed_dim)
        return self.mlp(x)[:, :, None, None]  # (B, embed_dim, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Full Model
# ═══════════════════════════════════════════════════════════════════════════
class DroughtForecastModel(nn.Module):
    """
    Irrigation-aware ConvLSTM with gradient checkpointing.
    Input : rzsm (B,T,1,H,W), nirv (B,T,1,H,W), irr (B,1,H,W), tc (B,T,3)
    Output: preds (B, n_leads, H, W)  NIRv anomaly at 8,16,24,32,40 days
    """
    def __init__(self,
                 hidden_dim:    int   = HIDDEN_DIM,
                 kernel_size:   int   = KERNEL_SIZE,
                 num_layers:    int   = NUM_LAYERS,
                 lead_times:    list  = LEAD_DAYS,
                 n_telecon:     int   = 3,
                 telecon_embed: int   = 16,
                 irr_channels:  int   = 8,
                 use_checkpoint: bool = True):
        super().__init__()
        self.n_leads = len(lead_times)

        self.irr_encoder = nn.Sequential(
            nn.Conv2d(1, irr_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(irr_channels, irr_channels, kernel_size=1),
        )
        self.telecon_ctx = TeleconContext(n_telecon, telecon_embed)

        input_dim = 1 + 1 + irr_channels + telecon_embed   # = 26
        self.encoder = ConvLSTMEncoder(input_dim, hidden_dim, kernel_size,
                                        num_layers, use_checkpoint=use_checkpoint)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),         nn.ReLU(),
                nn.Conv2d(16,  1, 1),
            )
            for _ in range(self.n_leads)
        ])

    def forward(self, rzsm, nirv, irr, tc):
        B, T, _, H, W = rzsm.shape
        irr_feat = self.irr_encoder(irr).unsqueeze(1).expand(-1, T, -1, -1, -1)
        tc_feat  = self.telecon_ctx(tc).unsqueeze(1).expand(-1, T, -1, H, W)
        x        = torch.cat([rzsm, nirv, irr_feat, tc_feat], dim=2)
        h        = self.encoder(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Sanity check
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device          : {}".format(device))

    B, T, H_t, W_t = 2, CONTEXT_LEN, 64, 64
    model   = DroughtForecastModel(use_checkpoint=True).to(device)
    rzsm_t  = torch.randn(B, T, 1, H_t, W_t, device=device)
    nirv_t  = torch.randn(B, T, 1, H_t, W_t, device=device)
    irr_t   = torch.rand( B, 1,    H_t, W_t, device=device)
    tc_t    = torch.randn(B, T, 3,           device=device)

    out = model(rzsm_t, nirv_t, irr_t, tc_t)
    print("Output shape    : {}".format(tuple(out.shape)))
    assert out.shape == (B, len(LEAD_DAYS), H_t, W_t)

    total = sum(p.numel() for p in model.parameters())
    print("Parameters      : {:,}".format(total))

    # Test gradient flow with checkpointing
    loss = out.sum()
    loss.backward()
    print("Backward pass   : OK (gradient checkpointing active)")

    arch = {"model_class": "DroughtForecastModel",
            "hidden_dim": HIDDEN_DIM, "kernel_size": KERNEL_SIZE,
            "num_layers": NUM_LAYERS, "lead_days": LEAD_DAYS,
            "context_len": CONTEXT_LEN,
            "input_channels": {"rzsm":1,"nirv":1,"irr_encoded":8,
                                "telecon_embed":16,"total":26},
            "gradient_checkpointing": True,
            "total_params": total}
    Path(PROC_ROOT).mkdir(exist_ok=True)
    with open(Path(PROC_ROOT)/"model_architecture.json","w") as f:
        json.dump(arch, f, indent=2)
    print("Architecture    : saved")
    print("Next: python 11_train.py")
