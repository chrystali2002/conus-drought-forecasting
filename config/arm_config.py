#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_config.py
=============
Resolve which experimental arm a training or evaluation run belongs to, so the
three arms of the irrigation ablation cannot drift apart.

WHY A SHARED MODULE RATHER THAN FORKED SCRIPTS
----------------------------------------------
    The ablation needs three arms that differ in EXACTLY ONE respect - which
    RZSM field reaches the model. Forking 11_train.py into two copies makes that
    guarantee impossible to keep: any later fix to the architecture, the split
    dates or the loss goes into one copy and not the other, and the comparison
    silently stops being an ablation. One resolver, imported everywhere, means
    the arms differ only in what this file says they differ in.

USAGE
-----
    Add near the top of 11_train.py, 12_evaluate.py, 15_train_multiseed.py and
    16_evaluate_multiseed.py:

        from arm_config import resolve_arm
        ARM = resolve_arm()

    Then replace the hard-coded store and variable, e.g. in 15_train_multiseed.py
    lines 57-58:

        ds_corr  = xr.open_zarr(out_dir / "anomalies_corrected.zarr", consolidated=True)
        rzsm_arr = np.nan_to_num(ds_corr["rzsm_anom_rainfed"].values.astype(np.float32), nan=0.0)

    with:

        ds_rzsm  = xr.open_zarr(out_dir / ARM.store, consolidated=True)
        rzsm_arr = np.nan_to_num(ds_rzsm[ARM.var].values.astype(np.float32), nan=0.0)
        ARM.check_input(rzsm_arr, out_dir)

    And suffix every output path so arms cannot overwrite each other:

        ckpt_dir = ckpt_dir / ARM.tag          # or f"{name}{ARM.suffix}"

SELECTING AN ARM
----------------
    Default is "corrected", so an unmodified run reproduces what you already
    have. Select another with an environment variable:

        DROUGHT_ARM=corrected   python 15_train_multiseed.py
        DROUGHT_ARM=uncorrected python 15_train_multiseed.py
        DROUGHT_ARM=nirv_only   python 15_train_multiseed.py

    In a SLURM array, one line covers all three:

        ARMS=(corrected uncorrected nirv_only)
        DROUGHT_ARM=${ARMS[$SLURM_ARRAY_TASK_ID]} python 15_train_multiseed.py

THE ONE FAILURE MODE THIS GUARDS AGAINST
----------------------------------------
    An input swap that silently does not take effect produces two arms with
    identical inputs, a ΔACC of almost exactly zero, and no error. You would
    conclude the correction does not matter when in fact you never tested it.
    check_input() compares the loaded array against the OTHER arm's field and
    aborts if they match, so that mistake fails immediately and loudly rather
    than after two training runs.

Run `python arm_config.py` to print the resolved configuration and self-test.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    name: str
    store: str          # zarr store holding the RZSM field for this arm
    var: str            # variable name within that store
    suffix: str         # append to output filenames
    tag: str            # subdirectory name for checkpoints and predictions
    zero_rzsm: bool     # if True, the model receives no soil moisture at all
    description: str

    def check_input(self, rzsm_arr, out_dir) -> None:
        """Abort if this arm's input is indistinguishable from the other's.

        Cheap insurance against an edit that did not take effect. Compares a
        deterministic subsample rather than the whole 3 GB cube.
        """
        import numpy as np
        if self.zero_rzsm:
            if np.abs(rzsm_arr).max() > 0:
                sys.exit(f"[arm_config] arm {self.name!r} expects zeroed RZSM but "
                         f"the array is non-zero. Apply the zeroing before this check.")
            print(f"[arm_config] {self.name}: RZSM zeroed as expected")
            return

        other = ARMS["uncorrected" if self.name == "corrected" else "corrected"]
        try:
            import xarray as xr
            from pathlib import Path
            ds = xr.open_zarr(Path(out_dir) / other.store, consolidated=True)
            ref = ds[other.var]
            # deterministic thin slice: every 97th timestep, a fixed pixel block
            a = rzsm_arr[::97, 100:140, 200:240]
            b = np.nan_to_num(ref[::97, 100:140, 200:240].values.astype("float32"),
                              nan=0.0)
        except Exception as exc:
            print(f"[arm_config] WARNING: could not verify the input swap ({exc}). "
                  f"Proceeding, but confirm the two arms differ before trusting "
                  f"any delta.")
            return

        if a.shape == b.shape and np.allclose(a, b, atol=1e-7):
            sys.exit(
                f"[arm_config] ABORT: arm {self.name!r} loaded a field identical to "
                f"arm {other.name!r}.\n"
                f"  The input swap did not take effect, so the two arms would train "
                f"on the same data\n"
                f"  and the ablation would return a spurious zero. Check that the "
                f"store and variable\n"
                f"  actually changed:  {self.store}:{self.var}  vs  "
                f"{other.store}:{other.var}")
        diff = float(np.abs(a - b).mean())
        print(f"[arm_config] {self.name}: input differs from {other.name} "
              f"(mean |diff| {diff:.5f} z on the audit slice)")


ARMS = {
    "corrected": Arm(
        name="corrected",
        store="anomalies_corrected.zarr",
        var="rzsm_anom_rainfed",
        suffix="",                     # empty, so existing outputs keep their names
        tag="corrected",
        zero_rzsm=False,
        description="LANID-corrected RZSM. The published configuration."),
    "uncorrected": Arm(
        name="uncorrected",
        store="anomalies.zarr",
        var="rzsm_anom",
        suffix="_uncorrected",
        tag="uncorrected",
        zero_rzsm=False,
        description="Raw RZSM anomaly, no irrigation correction. Isolates the "
                    "contribution of the correction itself - the paper's central "
                    "claim."),
    "nirv_only": Arm(
        name="nirv_only",
        store="anomalies.zarr",
        var="rzsm_anom",
        suffix="_nirv_only",
        tag="nirv_only",
        zero_rzsm=True,
        description="RZSM zeroed after loading. Isolates the contribution of soil "
                    "moisture as such. Already run as Step 13; included so all "
                    "three arms share one code path."),
}

ENV_VAR = "DROUGHT_ARM"


def resolve_arm(default: str = "corrected") -> Arm:
    name = os.environ.get(ENV_VAR, default).strip().lower()
    if name not in ARMS:
        sys.exit(f"[arm_config] unknown arm {name!r}. Choose from "
                 f"{sorted(ARMS)} via {ENV_VAR}.")
    arm = ARMS[name]
    print(f"[arm_config] arm = {arm.name}")
    print(f"[arm_config]   RZSM input : {arm.store}:{arm.var}"
          f"{'  (then zeroed)' if arm.zero_rzsm else ''}")
    print(f"[arm_config]   output tag : {arm.tag!r}, suffix {arm.suffix!r}")
    print(f"[arm_config]   {arm.description}")
    if arm.name == "corrected" and ENV_VAR not in os.environ:
        print(f"[arm_config]   ({ENV_VAR} unset, defaulting to the published arm)")
    return arm


def apply_zeroing(rzsm_arr, arm: Arm):
    """Zero the RZSM channel for the nirv_only arm. Call right after loading."""
    if arm.zero_rzsm:
        rzsm_arr = rzsm_arr * 0.0
        print(f"[arm_config] RZSM channel zeroed for arm {arm.name!r}")
    return rzsm_arr


if __name__ == "__main__":
    print(__doc__.split("Run `python")[0])
    print("=" * 70)
    for name in ARMS:
        os.environ[ENV_VAR] = name
        resolve_arm()
        print("-" * 70)
    print("\nSanity checks")
    ok = True
    c, u = ARMS["corrected"], ARMS["uncorrected"]
    if (c.store, c.var) == (u.store, u.var):
        print("  FAIL corrected and uncorrected point at the same field"); ok = False
    else:
        print("  OK   corrected and uncorrected read different fields")
    tags = [a.tag for a in ARMS.values()]
    sufs = [a.suffix for a in ARMS.values()]
    if len(set(tags)) != len(tags) or len(set(sufs)) != len(sufs):
        print("  FAIL output tags or suffixes collide - arms would overwrite "
              "each other"); ok = False
    else:
        print("  OK   every arm writes to a distinct tag and suffix")
    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
