#!/usr/bin/env python3
"""Section 74, WS1-WS5: the sanity band with the shadow WARM-STARTED.

`docs/DECISIONS.md` D76 ruled on 2026-09-21 that the flown retrainer initialises
the shadow from the flying model's weights and fine-tunes -- "there is no random
initialisation on the flight path" -- and D76 consequence 7 says what that did
and did not buy: it removed the blocker D74.4 named, and **it did not make the
gate enforceable, and the claim is not made**. What is owed is an arm in which
the shadow is ACTUALLY warm-started. Every arm in 67, 68 and 69 was a cold fit.

This is that arm.

(!) THE MARGIN IS DECLARED HERE, BEFORE ANYTHING RUNS, AND IT IS NOT MEASURED.
K = 2.0330 is 67.3's and is imported. F = 6.5687% is 69.7's Arm G -- six disjoint
segments of 18,000 at one seed -- which D76 consequence 2 made the operative
floor. m = K * F = 13.3542%. **Nothing in this file chooses a number.**

(!) STOP 44: the margin and the drift multiple are declared before this runs, not
after an arm is seen. (!) STOP 45: N, L, HELD and F's definition are 69's and do
not move; this measures a different QUANTITY at the same geometry.
(!) STOP 38 / 40: no target rate is an input; synthetic.py is not modified.
(!) D76 c.5: 69's SD2 is NOT amended and no arm there is retroactively certified.

Runner:  PYTHONPATH=src .venv/bin/python scripts/s74_warm_start.py
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


# --- 74.2's declared constants. NOTHING here is measured by this file. -------------
FIXTURE_STEPS = 780_000      # 69's fixture -> 117,000 contiguous healthy
FIXTURE_SEED = 0
L = 18_000                   # 64.3's own training length, as 69 restored it
N = 6
HELD_LEN = 9_000
FLYING_SEG, SHADOW_SEG = 0, 5
RAMP_SEG = 4                 # 68.2's plateau geometry
MIN_HEALTHY = N * L + HELD_LEN

#: 69.7's Arm G, made the operative floor by D76 consequence 2. NOT re-measured.
FLOOR_G = 0.065687
#: 67.3's factor, imported. The margin follows arithmetically (D76 consequence 3).
K = 2.0330
MARGIN = K * FLOOR_G         # 0.133542


def train_warm(s64, segment, names, init_weights, seed: int = 0):
    """A fit that STARTS FROM `init_weights` instead of from a distribution.

    Identical to `s67.train_on` in every other respect -- same hyper, same
    `reuse_weights=False`, same context -- so the only difference between a warm
    arm and a cold one is where the optimiser began.
    """
    from sentinel_eval.detector import Context
    from sentinel_models import telemanom
    from sentinel_models.detectors import GRUForecastDetector

    hyper = s64._hyper(s64.FLOWN["window"], s64.FLOWN["hidden"],
                       s64.FLOWN["n_predictions"], s64.FLOWN["max_epochs"], seed)
    det = GRUForecastDetector(hyper=hyper, reuse_weights=False,
                              config=telemanom.Config(error_window=s64.FLIGHT_ERROR_WINDOW),
                              init_weights=init_weights)
    ctx = Context(mission="s74", channels=list(names), groups=(1,) * segment.shape[1],
                  period_seconds=1, fold=0, window=(0, len(segment)))
    det.fit(segment, np.ones(len(segment), dtype=bool), ctx)
    return det


def main() -> int:
    s68 = _load("s68_sanity_plateau")
    s67 = _load("s67_sanity_rerun")
    s66 = _load("s66_noise_floor")
    s64 = s67._s64()
    target, ladder = s67.DRIFT_TARGET, s67.DRIFT_LADDER

    print("== Section 74: the sanity band with the shadow warm-started ==")
    print(f"   K {K:.4f} (67.3, IMPORTED)   F {100 * FLOOR_G:.4f}% "
          f"(69.7 Arm G, D76 c.2)   m = K * F = {100 * MARGIN:.4f}%")
    print(f"   drift target {target:.4f}x (67.3, IMPORTED)")
    print("   (!) EVERY ONE OF THOSE IS DECLARED. This file measures arms, not constants.")

    values, names = s64.healthy_run(seed=FIXTURE_SEED, steps=FIXTURE_STEPS)
    n = len(values)
    print(f"\n   fixture n={FIXTURE_STEPS:,} -> healthy {n:,} x {values.shape[1]} "
          f"({100 * n / FIXTURE_STEPS:.1f}%)")
    if n < MIN_HEALTHY:
        print(f"   STOP: {n:,} healthy, below the declared {MIN_HEALTHY:,}.")
        return 2
    held_lo = n - HELD_LEN
    seg = [values[i * L:(i + 1) * L] for i in range(N)]
    print(f"   segments  {N} x {L:,}   HELD [{held_lo:,}:{n:,}], "
          f"settled {HELD_LEN - s64.WARMUP:,}")

    # ---- the flying model: a COLD fit, because that is what a flying model is ------
    print("\n-- the flying model: cold fit, segment 1, seed 0 (69's, unchanged)")
    flying = s67.train_on(s64, seg[FLYING_SEG], names)
    flying_weights = flying._weights
    clean = s64.forecast_residual(flying, values[held_lo:], names)
    print(f"   clean residual {clean:.8f}")

    # ---- ARM W: how far two WARM starts diverge -----------------------------------
    print(f"\n-- ARM W: {N} WARM-STARTED fits, one per segment, all from the flying weights")
    print("   D76 c.6 predicted this is SMALLER than Arm G's cold-fit floor, before")
    print("   the arm existed. 69.7's Arm G measured 6.5687% on the same segments.")
    arm_w = []
    for i in range(N):
        det = train_warm(s64, seg[i], names, flying_weights)
        r = s64.forecast_residual(det, values[held_lo:], names)
        arm_w.append(r)
        print(f"     segment {i + 1}   residual {r:.8f}")
    w = s66.spreads(np.array(arm_w))
    print(f"   W (warm segments)  range {100 * w['range_rel']:.4f}%   "
          f"std {100 * w['std_rel']:.4f}%")
    print(f"   G (cold segments)  range {100 * FLOOR_G:.4f}%   (69.7, for comparison)")
    ws3 = "HOLD" if w["range_rel"] < FLOOR_G else "FAIL"

    # ---- the drift, sized by the flying model alone, no fit involved ---------------
    ramp_from, plateau_from = RAMP_SEG * L, SHADOW_SEG * L
    print(f"\n-- DRIFT SIZING (flying only, no fit).  ramp [{ramp_from:,}:{plateau_from:,}),"
          f" FLAT to {n:,}")
    chosen = None
    for gain in ladder:
        d = s68.apply_plateau_drift(values, ramp_from, plateau_from, gain)
        r = s64.forecast_residual(flying, d[held_lo:], names)
        mark = ""
        if chosen is None and r >= target * clean:
            chosen, mark = gain, "   <- TAKEN"
        print(f"     1.000 -> {1 + gain:.3f}   residual {r:.8f}   {r / clean:.4f}x{mark}")
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
    ws5 = "HOLD" if abs(seen - tested) <= 1e-4 else "FAIL"
    print(f"\n-- WS5: shadow trains at {seen:.6f}, HELD scored at {tested:.6f}  -> {ws5}")

    # ---- the two arms, both WARM-STARTED ------------------------------------------
    saved = s67.MARGIN
    s67.MARGIN = MARGIN
    try:
        print("\n   ARM A: warm-started, later healthy segment, NO drift")
        arm_a = s67.run_arm(s64, "A (stationary, warm)", flying,
                            train_warm(s64, seg[SHADOW_SEG], names, flying_weights),
                            seg[FLYING_SEG], seg[SHADOW_SEG], values[held_lo:], names)
        print(f"\n   ARM B: warm-started, drift-affected segment, "
              f"ALL {values.shape[1]} channels, flat gain {1 + chosen:.3f}")
        arm_b = s67.run_arm(s64, "B (learnable drift, warm)", flying,
                            train_warm(s64, drifted[sha_lo:sha_hi], names, flying_weights),
                            seg[FLYING_SEG], drifted[sha_lo:sha_hi],
                            drifted[held_lo:], names)
    finally:
        s67.MARGIN = saved

    ws1 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        ws2 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        ws2 = "NO VERDICT"
    else:
        ws2 = "FAIL"

    print(f"\n   WS1 (Arm A, warm and stationary, NOT certified)   -> {ws1}")
    print(f"   WS2 (Arm B, warm and drifted, IS certified)       -> {ws2}")
    print(f"   WS3 (warm spread < Arm G's cold floor, D76 c.6)   -> {ws3}")
    print("   WS4 (no target rate is an input)                  -> HOLD")
    print(f"   WS5 (the shadow trains on what HELD is scored at) -> {ws5}")
    print(f"\n   (!) 69's SD2 is NOT amended. Its arms were cold fits at a common seed "
          f"and stay as recorded (D76 c.5).")
    return 0 if (ws1 == "HOLD" and ws2 == "HOLD" and ws3 == "HOLD"
                 and ws5 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
