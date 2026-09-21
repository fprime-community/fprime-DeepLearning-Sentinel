#!/usr/bin/env python3
"""Section 69, NG1-NG2 and SD1-SD6: the floor and the band at the flown training length.

68.7 promoted one owed item to blocking: **a floor measured on fits that have converged.**
66.7 measured F = 29.8606% at L = 6,000 and left two readings open -- either residual
noise scales that steeply with training length, or 6,000 timesteps is below the length at
which this architecture fits stably. 68.6 then showed the consequence: the margin derived
from that floor refuses the most favourable arm this project can build, by 10.59 points.

`selftest.healthy_run` returns the longest CONTIGUOUS anomaly-free run, and that length is
**15.0% of the fixture size** -- 45,000 at 300,000, 90,000 at 600,000, 180,000 at
1,200,000, measured. So 64.3's own training length of 18,000 IS reachable: 780,000 gives
117,000 = 6 * 18,000 + 9,000 exactly.

**The margin formula is fixed before anything runs.** m = K * F with K = 2.0330 from 67.3,
imported rather than restated. Only F is measured. There is no dial anywhere in this file.

(!) STOP 44: K, the drift target and the ladder are 67.3's and are imported.
(!) STOP 45: 66's constants are not touched. This measures a DIFFERENT floor at a
different training length and says so; 66.7's F stands as 66.7's.
(!) STOP 38 / 40: no target rate is an input; synthetic.py is not modified.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 69.3's declared constants. Only F is measured; everything else is fixed. -------
FIXTURE_STEPS = 780_000      # -> 117,000 contiguous healthy at the measured 15.0%
FIXTURE_SEED = 0
L = 18_000                   # 64.3's OWN training length, restored
N = 6
HELD_LEN = 9_000             # 64.3's HELD length, unchanged since 66
FLYING_SEG, SHADOW_SEG = 0, 5
RAMP_SEG = 4                 # the plateau geometry 68.2 established
MIN_HEALTHY = N * L + HELD_LEN


def main() -> int:
    s68 = _load("s68_sanity_plateau")
    s67 = _load("s67_sanity_rerun")
    s66 = _load("s66_noise_floor")
    s64 = s67._s64()
    K, target, ladder = s67.K, s67.DRIFT_TARGET, s67.DRIFT_LADDER

    print("== Section 69: the floor and the band at L = 18,000 ==")
    print(f"   K {K:.4f} (67.3, IMPORTED)   drift target {target:.4f}x   "
          f"margin formula m = K * F, F measured below")

    values, names = s64.healthy_run(seed=FIXTURE_SEED, steps=FIXTURE_STEPS)
    n = len(values)
    print(f"\n   fixture n={FIXTURE_STEPS:,} -> healthy {n:,} x {values.shape[1]} "
          f"({100 * n / FIXTURE_STEPS:.1f}%)")
    if n < MIN_HEALTHY:
        print(f"   STOP: {n:,} healthy, below the declared {MIN_HEALTHY:,}. The "
              "segmentation does not fit and is not re-shaped to make it.")
        return 2
    held_lo = n - HELD_LEN
    seg = [values[i * L:(i + 1) * L] for i in range(N)]
    print(f"   segments  {N} x {L:,} over [0:{N * L:,})   HELD [{held_lo:,}:{n:,}], "
          f"settled {HELD_LEN - s64.WARMUP:,}")

    # ---- Part 1: the floor, exactly 66's two arms at the new length ----------------
    print(f"\n-- ARM S: {N} fits on segment 1, seeds 0..{N - 1}")
    arm_s = []
    for s in range(N):
        r = s64.forecast_residual(s67.train_on(s64, seg[0], names, s), values[held_lo:], names)
        arm_s.append(r)
        print(f"     seed {s}   residual {r:.8f}")
    print(f"\n-- ARM G: {N} fits, one per segment, seed 0")
    arm_g = []
    for i in range(N):
        r = s64.forecast_residual(s67.train_on(s64, seg[i], names), values[held_lo:], names)
        arm_g.append(r)
        print(f"     segment {i + 1}   residual {r:.8f}")
    s, g = s66.spreads(np.array(arm_s)), s66.spreads(np.array(arm_g))
    for label, d in (("S (seeds)   ", s), ("G (segments)", g)):
        print(f"   {label}  range {100 * d['range_rel']:.4f}%   std {100 * d['std_rel']:.4f}%")
    F = max(s["range_rel"], g["range_rel"])
    m = K * F
    print(f"\n   (!) F = {100 * F:.4f}%  ->  m = K * F = {100 * m:.4f}%")
    print(f"       66.7 measured F = 29.8606% at L = 6,000.")
    ng1 = "HOLD" if F < 0.298606 else "FAIL"
    ng2 = "HOLD" if s["range_rel"] < g["range_rel"] else "FAIL"
    print(f"   NG1 (F falls at the longer training length)  -> {ng1}")
    print(f"   NG2 (seed spread < segment spread, as 66's NF1 predicted and lost) -> {ng2}")

    # ---- Part 2: the band, at the m just derived -----------------------------------
    flying = s67.train_on(s64, seg[FLYING_SEG], names)
    clean = s64.forecast_residual(flying, values[held_lo:], names)
    ramp_from, plateau_from = RAMP_SEG * L, SHADOW_SEG * L
    print(f"\n-- DRIFT SIZING (flying only, no fit)   clean {clean:.8f}")
    print(f"   ramp [{ramp_from:,}:{plateau_from:,}), FLAT to {n:,}")
    chosen = None
    for gain in ladder:
        d = s68.apply_plateau_drift(values, ramp_from, plateau_from, gain)
        r = s64.forecast_residual(flying, d[held_lo:], names)
        mark = ""
        if chosen is None and r >= target * clean:
            chosen, mark = gain, "   <- TAKEN"
        print(f"     1.000 -> {1 + gain:.3f}   residual {r:.8f}   {r / clean:.4f}x{mark}")
    sd5 = "HOLD" if chosen is not None else "FAIL"
    if chosen is None:
        print("   (!) LADDER EXHAUSTED.")
        chosen = ladder[-1]
    drifted = s68.apply_plateau_drift(values, ramp_from, plateau_from, chosen)

    sha_lo, sha_hi = SHADOW_SEG * L, (SHADOW_SEG + 1) * L

    def max_gain(lo, hi):
        raw = values[lo:hi]
        safe = np.where(np.abs(raw) < 1e-12, np.nan, raw)
        return float(np.nanmax(drifted[lo:hi] / safe))

    seen, tested = max_gain(sha_lo, sha_hi), max_gain(held_lo, None)
    sd6 = "HOLD" if abs(seen - tested) <= 1e-4 else "FAIL"
    print(f"\n-- SD6: shadow trains at {seen:.6f}, HELD scored at {tested:.6f}  -> {sd6}")

    saved = s67.MARGIN
    s67.MARGIN = m       # the derived margin, used by run_arm; 67.3's value is not read
    try:
        arm_a = s67.run_arm(s64, "A (stationary)", flying,
                            s67.train_on(s64, seg[SHADOW_SEG], names),
                            seg[FLYING_SEG], seg[SHADOW_SEG], values[held_lo:], names)
        print(f"\n   ARM B drift: ALL {values.shape[1]} channels, flat gain {1 + chosen:.3f}")
        arm_b = s67.run_arm(s64, "B (learnable drift)", flying,
                            s67.train_on(s64, drifted[sha_lo:sha_hi], names),
                            seg[FLYING_SEG], drifted[sha_lo:sha_hi],
                            drifted[held_lo:], names)
    finally:
        s67.MARGIN = saved

    sd1 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        sd2 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        sd2 = "NO VERDICT"
    else:
        sd2 = "FAIL"
    if arm_a["inert"] or arm_b["inert"]:
        sd4 = "NO VERDICT"
    else:
        sd4 = "HOLD" if ((arm_a["part_i"] == arm_a["part_ii"])
                         and (arm_b["part_i"] == arm_b["part_ii"])) else "NO VERDICT"
    print(f"\n   NG1 (F falls at L = 18,000)                       -> {ng1}")
    print(f"   NG2 (seed spread < segment spread)                -> {ng2}")
    print(f"   SD1 (Arm A NOT certified)                         -> {sd1}")
    print(f"   SD2 (Arm B certified)                             -> {sd2}")
    print("   SD3 (no target rate is an input)                  -> HOLD")
    print(f"   SD4 (part (ii) not inert, and the parts agree)    -> {sd4}")
    print(f"   SD5 (the ladder reached its target)               -> {sd5}")
    print(f"   SD6 (the shadow trains on what HELD is scored at) -> {sd6}")
    return 0 if (sd1 == "HOLD" and sd2 == "HOLD" and sd6 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
