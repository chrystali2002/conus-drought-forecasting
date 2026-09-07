#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_ablation_compare.py
======================
Compare the irrigation ablation arms and report the effect of the correction.

WHY PAIRING BY SEED MATTERS MORE THAN THE MASKING
-------------------------------------------------
    Both arms are trained on the SAME seeds. That makes every seed a matched
    pair, so the statistic is the per-seed difference:

        d_s = ACC(corrected, seed s) - ACC(uncorrected, seed s)

    and the question is whether mean(d_s) differs from zero. Comparing two
    independent five-value means instead throws the pairing away: seed-to-seed
    variation in this setup is large relative to the expected effect, and an
    unpaired test would drown a real difference in initialization noise. With
    five seeds the paired form is roughly an order of magnitude more sensitive.

    So do NOT report "corrected 0.51 +/- 0.03 vs uncorrected 0.50 +/- 0.03,
    overlapping, therefore no effect". That comparison is the wrong one.

WHY SCORE ON IRRIGATED PIXELS
-----------------------------
    The correction only alters pixels with irr_frac >= 0.15, which is 554 of the
    3,093 water-limited-crop evaluation pixels (17.9%). Elsewhere the two arms
    receive byte-identical input, so a domain-averaged difference is diluted
    about 5.6x toward zero BY CONSTRUCTION - not because the correction is
    ineffective, but because most pixels cannot respond. The irrigated-pixel
    number is the test of the mechanism; the domain average is context.

    The rainfed subset is the internal control: the arms are identical there, so
    any apparent difference is pure noise and gives you an empirical null band
    to judge the irrigated number against. That comparison is more convincing
    than any p-value.

INPUT
-----
    A long-format CSV per arm, written by 16_evaluate_multiseed.py:

        arm,seed,lead,mask,n_pixels,acc,rmse
        corrected,42,8,all_wlc,3093,0.661,0.72
        corrected,42,8,irrigated,554,0.643,0.75
        corrected,42,8,rainfed,2539,0.667,0.71
        ...

    Default location: outputs/metrics/{arm}/multiseed_stratified.csv

USAGE
-----
    python 17_ablation_compare.py
    python 17_ablation_compare.py --arms corrected uncorrected nirv_only
    python 17_ablation_compare.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")

try:
    from scipy import stats as _st
except ImportError:
    _st = None


def paired_stats(d: np.ndarray, n_boot: int = 20000, seed: int = 0):
    """Mean, CI and p for paired differences.

    With five seeds the sample is small, so report both a t interval and a
    bootstrap interval rather than pretending either is definitive. The
    bootstrap makes no normality assumption; the t interval is what a reviewer
    will expect to see. If they disagree materially, say so.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return dict(n=0, mean=np.nan, t_lo=np.nan, t_hi=np.nan,
                    b_lo=np.nan, b_hi=np.nan, p=np.nan, sd=np.nan)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else np.nan

    t_lo = t_hi = p = np.nan
    if n > 1 and _st is not None and sd > 0:
        se = sd / np.sqrt(n)
        crit = _st.t.ppf(0.975, n - 1)
        t_lo, t_hi = m - crit * se, m + crit * se
        p = float(_st.ttest_1samp(d, 0.0).pvalue)

    rng = np.random.default_rng(seed)
    if n > 1:
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
    else:
        b_lo = b_hi = np.nan
    return dict(n=n, mean=m, sd=sd, t_lo=t_lo, t_hi=t_hi,
                b_lo=float(b_lo), b_hi=float(b_hi), p=p)


def load(arms, root, fname):
    frames = []
    for a in arms:
        p = os.path.join(root, a, fname)
        if not os.path.exists(p):
            print(f"  missing: {p}")
            continue
        df = pd.read_csv(p)
        if "arm" not in df.columns:
            df["arm"] = a
        frames.append(df)
        print(f"  {a}: {len(df)} rows from {p}")
    if not frames:
        sys.exit("No arm CSVs found. Has 16_evaluate_multiseed.py run for each arm, "
                 "and does it write the stratified long-format CSV?")
    return pd.concat(frames, ignore_index=True)


def compare(df, treat, ref, masks, out_csv):
    rows = []
    leads = sorted(df["lead"].unique())
    print(f"\n{'='*74}\nPAIRED COMPARISON: {treat} minus {ref}\n{'='*74}")

    for mask in masks:
        sub = df[df["mask"] == mask]
        if sub.empty:
            print(f"\nmask {mask!r}: absent from the CSVs, skipped")
            continue
        npx = int(sub["n_pixels"].iloc[0]) if "n_pixels" in sub else -1
        print(f"\n--- mask: {mask}  ({npx} px)")
        print(f"{'lead':>6} {'dACC':>9} {'95% t CI':>20} {'95% boot CI':>20} "
              f"{'p':>8} {'seeds':>6}")
        for lead in leads:
            a = sub[(sub.arm == treat) & (sub.lead == lead)].set_index("seed")["acc"]
            b = sub[(sub.arm == ref) & (sub.lead == lead)].set_index("seed")["acc"]
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            if len(common) < max(len(a), len(b)):
                print(f"    note: only {len(common)} seeds common to both arms")
            d = (a.loc[common] - b.loc[common]).values
            s = paired_stats(d)
            rows.append(dict(mask=mask, lead=lead, n_pixels=npx, **s))
            print(f"{lead:>6} {s['mean']:>+9.4f} "
                  f"[{s['t_lo']:>+8.4f},{s['t_hi']:>+8.4f}] "
                  f"[{s['b_lo']:>+8.4f},{s['b_hi']:>+8.4f}] "
                  f"{s['p']:>8.3f} {s['n']:>6}")

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # ---- interpretation, using the rainfed subset as an empirical null
    print(f"\n{'='*74}\nREADING THE RESULT\n{'='*74}")
    irr = res[res["mask"] == "irrigated"]
    rain = res[res["mask"] == "rainfed"]
    if irr.empty:
        print("  No irrigated-pixel rows, so the mechanism was not tested.")
        return res

    if not rain.empty:
        band = float(np.nanmax(np.abs(rain["mean"])))
        print(f"  Rainfed control: |dACC| never exceeds {band:.4f}.")
        print("  The arms are identical on rainfed pixels, so that is your")
        print("  empirical noise floor - a difference this size means nothing.")
    else:
        band = np.nan

    sig = irr[(irr["t_lo"] > 0) | (irr["t_hi"] < 0)]
    big = irr[np.abs(irr["mean"]) > band] if np.isfinite(band) else irr.iloc[:0]

    print(f"\n  Irrigated pixels: dACC ranges "
          f"{irr['mean'].min():+.4f} to {irr['mean'].max():+.4f}")
    print(f"    leads with an interval excluding zero: {len(sig)} of {len(irr)}")
    if np.isfinite(band):
        print(f"    leads exceeding the rainfed noise floor: {len(big)} of {len(irr)}")

    if len(sig) and (irr["mean"] > 0).all():
        print("\n  POSITIVE. The correction improves skill where it acts, "
              "consistently in sign across leads.")
        print("  Report the irrigated-pixel dACC as the headline, the domain")
        print("  average as context, and explain the dilution explicitly.")
    elif len(sig) and not (irr["mean"] > 0).all():
        print("\n  MIXED - significant at some leads but inconsistent in sign.")
        print("  With five seeds that is weak evidence. Report all leads, resist")
        print("  selecting the favourable ones, and treat it as inconclusive.")
    else:
        print("\n  NO DETECTABLE EFFECT at any lead.")
        print("  This is a finding, not a failure. It says that at 9 km over CONUS,")
        print("  irrigation contamination of SMAP L4 RZSM is too small to change")
        print("  sub-seasonal vegetation forecast skill - which is the open question")
        print("  Le et al. (2024) flagged and did not answer.")
        print("  Frame the paper around the forecasting framework, present the")
        print("  correction as a methodological contribution with a measured null,")
        print("  and note the scale dependence: a 9 km pixel at 25% irrigated")
        print("  fraction behaves mostly like rainfed land, so a finer-resolution")
        print("  study could still find an effect.")
        print("  State the minimum effect you could have detected, so the null is")
        print("  quantified rather than merely asserted:")
        for _, r in irr.iterrows():
            half = (r["t_hi"] - r["t_lo"]) / 2
            print(f"    lead {int(r['lead']):>3}d  detectable above ~{half:.4f} ACC")
    return res


def self_test():
    """Synthetic CSVs with a known planted effect, to check the paired logic."""
    print("SELF-TEST\n")
    rng = np.random.default_rng(0)
    seeds = [42, 123, 777, 2024, 31337]
    leads = [8, 16, 24, 32, 40]
    rows = []
    TRUE = 0.03                     # planted effect on irrigated pixels only
    for s in seeds:
        base = rng.normal(0, 0.02)  # large shared seed effect
        for L in leads:
            for mask, npx, eff in (("all_wlc", 3093, TRUE * 0.179),
                                   ("irrigated", 554, TRUE),
                                   ("rainfed", 2539, 0.0)):
                a = 0.6 - L * 0.003 + base + eff + rng.normal(0, 0.002)
                b = 0.6 - L * 0.003 + base + rng.normal(0, 0.002)
                rows.append(dict(arm="corrected", seed=s, lead=L, mask=mask,
                                 n_pixels=npx, acc=a))
                rows.append(dict(arm="uncorrected", seed=s, lead=L, mask=mask,
                                 n_pixels=npx, acc=b))
    df = pd.DataFrame(rows)
    res = compare(df, "corrected", "uncorrected",
                  ["all_wlc", "irrigated", "rainfed"], "/tmp/_selftest.csv")

    irr = res[res["mask"] == "irrigated"]["mean"].mean()
    rain = res[res["mask"] == "rainfed"]["mean"].mean()
    dom = res[res["mask"] == "all_wlc"]["mean"].mean()
    print(f"\n  planted irrigated effect {TRUE:+.4f}, recovered {irr:+.4f}")
    print(f"  planted rainfed effect   +0.0000, recovered {rain:+.4f}")
    print(f"  domain average {dom:+.4f} (dilution {dom/TRUE:.2f}, expected ~0.18)")

    # unpaired comparison, to show what the pairing buys
    sub = df[(df["mask"] == "irrigated") & (df["lead"] == 8)]
    ca = sub[sub.arm == "corrected"]["acc"].values
    ua = sub[sub.arm == "uncorrected"]["acc"].values
    if _st is not None:
        p_un = _st.ttest_ind(ca, ua).pvalue
        p_pa = _st.ttest_rel(ca, ua).pvalue
        print(f"\n  lead 8, unpaired p = {p_un:.3f}   paired p = {p_pa:.4f}")
        print("  The pairing is what makes the effect visible.")

    ok = abs(irr - TRUE) < 0.01 and abs(rain) < 0.005
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/metrics")
    ap.add_argument("--file", default="multiseed_stratified.csv")
    ap.add_argument("--arms", nargs="+", default=["corrected", "uncorrected"])
    ap.add_argument("--treat", default="corrected")
    ap.add_argument("--ref", default="uncorrected")
    ap.add_argument("--masks", nargs="+",
                    default=["all_wlc", "irrigated", "rainfed"])
    ap.add_argument("--out", default="outputs/metrics/ablation_paired.csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    print("Loading arm results")
    df = load(args.arms, args.root, args.file)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    compare(df, args.treat, args.ref, args.masks, args.out)


if __name__ == "__main__":
    main()
