#!/usr/bin/env python3
"""Print every path each arm will read and write, so you can check them BEFORE
committing GPU hours. Reads nothing, writes nothing."""
import os
from pathlib import Path
from arm_config import ARMS

#SEEDS = (42, 123, 777, 2024, 31337)
# config.py
SEEDS = [42, 123, 777, 2024, 31337,
         7, 99, 512, 1234, 4096, 8191, 20250, 60613, 77777, 90210]
for name, arm in ARMS.items():
    md = Path("models") / arm.tag
    xd = Path("outputs/metrics") / arm.tag
    print(f"\n=== arm: {name} ===")
    print(f"  input      {arm.store}:{arm.var}"
          f"{'  (zeroed)' if arm.zero_rzsm else ''}")
    print(f"  WRITE  11  {md/'best_convlstm_conus.pt'}")
    print(f"  WRITE  11  {md/'last_convlstm_conus.pt'}")
    print(f"  READ   12  {md/'best_convlstm_conus.pt'}")
    print(f"  WRITE  12  {xd/'evaluation_summary.csv'}")
    print(f"  WRITE  12  {xd/'evaluation_by_zone.csv'}")
    for s in SEEDS:
        print(f"  WRITE  15  {md/f'seed_{s}_convlstm.pt'}")
    for s in SEEDS:
        print(f"  READ   16  {md/f'seed_{s}_convlstm.pt'}")
    print(f"  WRITE  16  {xd/'multiseed_test_results.csv'}")

print("\n--- collision check ---")
paths = {}
bad = False
for name, arm in ARMS.items():
    for p in [Path("models")/arm.tag/"best_convlstm_conus.pt",
              Path("outputs/metrics")/arm.tag/"multiseed_test_results.csv"]:
        if p in paths:
            print(f"  COLLISION {p} used by {paths[p]} and {name}"); bad = True
        paths[p] = name
print("  OK - every arm writes to distinct paths" if not bad else "  FIX THE ABOVE")
