#!/usr/bin/env python3
"""Section 68, SC1-SC6: the sanity arm with a drift the shadow can actually learn.

67.8 found Arm B failing part (i) with a drift that quintupled the flying model's
residual, and found the reason in the arm's own geometry: the ramp ran from the start of
the shadow's segment through the end of HELD, so the shadow trained on gain factors
1.0000 -> 1.7999 while HELD spanned 1.8001 -> 3.0000. **HELD reached a gain the shadow
never saw**, and a forecaster windowed on 250 steps cannot extrapolate a slope.

That geometry is inherited from `scripts/s64_sanity_band.py:44-56` and is wrong for
testing a shadow model independently of the drift's magnitude.

**The fix changes the geometry and nothing else.** The ramp completes BEFORE the shadow's
segment begins, and the gain is then FLAT across the shadow's segment and all of HELD, so
the shadow trains on exactly the distribution it is scored on.

(!) STOP 44: the margin `m` is NOT moved. It is 67.3's, derived from 66.7's floor, and it
is imported from the 67 runner rather than restated so it cannot drift.
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


# --- 68.3's declared constants. Everything numeric is 67's, unchanged. --------------
RAMP_FROM = 24_000          # the ramp occupies segment 5, BEFORE the shadow's segment
PLATEAU_FROM = 30_000       # flat from here to the end: shadow's segment AND all of HELD


def apply_plateau_drift(values: np.ndarray, ramp_from: int, plateau_from: int,
                        gain: float) -> np.ndarray:
    """Ramp 1.000 -> 1.000+gain over [ramp_from:plateau_from), then FLAT to the end.

    The whole point: `values[plateau_from:]` is one distribution, and it contains both
    the shadow's training segment and HELD.
    """
    out = values.astype(np.float32).copy()
    ramp = np.linspace(0.0, gain, plateau_from - ramp_from, dtype=np.float32)
    out[ramp_from:plateau_from] = out[ramp_from:plateau_from] * (1.0 + ramp)[:, None]
    out[plateau_from:] = out[plateau_from:] * np.float32(1.0 + gain)
    return out


def main() -> int:
    s67 = _load("s67_sanity_rerun")
    s64 = s67._s64()
    m, target, ladder = s67.MARGIN, s67.DRIFT_TARGET, s67.DRIFT_LADDER
    L, HELD_LEN = s67.L, s67.HELD_LEN

    print("== Section 68, SC1-SC6: the sanity arm with a learnable drift ==")
    print(f"   margin {100 * m:.4f}% (67.3, UNCHANGED)   drift target {target:.4f}x")

    values, names = s64.healthy_run(seed=s67.FIXTURE_SEED, steps=s67.FIXTURE_STEPS)
    n = len(values)
    held_lo = n - HELD_LEN
    fly_lo, fly_hi = s67.FLYING_SEG * L, (s67.FLYING_SEG + 1) * L
    sha_lo, sha_hi = s67.SHADOW_SEG * L, (s67.SHADOW_SEG + 1) * L
    print(f"   FLYING [{fly_lo:,}:{fly_hi:,}]  SHADOW [{sha_lo:,}:{sha_hi:,}]  "
          f"HELD [{held_lo:,}:{n:,}]")
    print(f"   ramp [{RAMP_FROM:,}:{PLATEAU_FROM:,}), then FLAT to {n:,}")

    flying = s67.train_on(s64, values[fly_lo:fly_hi], names)
    clean = s64.forecast_residual(flying, values[held_lo:], names)
    print(f"\n-- DRIFT SIZING (flying model only, no fit)   clean {clean:.8f}")
    chosen = None
    for gain in ladder:
        drifted = apply_plateau_drift(values, RAMP_FROM, PLATEAU_FROM, gain)
        r = s64.forecast_residual(flying, drifted[held_lo:], names)
        mark = ""
        if chosen is None and r >= target * clean:
            chosen, mark = gain, "   <- TAKEN"
        print(f"     1.000 -> {1 + gain:.3f}   residual {r:.8f}   {r / clean:.4f}x{mark}")
    sc5 = "HOLD" if chosen is not None else "FAIL"
    if chosen is None:
        print("   (!) LADDER EXHAUSTED -- no declared drift reaches the target.")
        chosen = ladder[-1]

    drifted = apply_plateau_drift(values, RAMP_FROM, PLATEAU_FROM, chosen)

    # SC6: the defect 67.8 found, checked mechanically rather than assumed fixed.
    def max_gain(lo: int, hi: int | None) -> float:
        raw = values[lo:hi]
        got = drifted[lo:hi]
        safe = np.where(np.abs(raw) < 1e-12, np.nan, raw)
        return float(np.nanmax(got / safe))

    seen = max_gain(sha_lo, sha_hi)
    tested = max_gain(held_lo, None)
    sc6 = "HOLD" if abs(seen - tested) <= 1e-4 else "FAIL"
    print(f"\n-- SC6: gain the shadow TRAINS on {seen:.6f}; gain HELD is SCORED at "
          f"{tested:.6f}")
    print(f"   |difference| {abs(seen - tested):.2e}   need <= 1e-4   -> {sc6}")

    arm_a = s67.run_arm(s64, "A (stationary)", flying,
                        s67.train_on(s64, values[sha_lo:sha_hi], names),
                        values[fly_lo:fly_hi], values[sha_lo:sha_hi],
                        values[held_lo:], names)
    print(f"\n   ARM B drift: ALL {values.shape[1]} channels, flat gain "
          f"{1 + chosen:.3f} from index {PLATEAU_FROM:,}")
    arm_b = s67.run_arm(s64, "B (learnable drift)", flying,
                        s67.train_on(s64, drifted[sha_lo:sha_hi], names),
                        values[fly_lo:fly_hi], drifted[sha_lo:sha_hi],
                        drifted[held_lo:], names)

    sc1 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        sc2 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        sc2 = "NO VERDICT"
    else:
        sc2 = "FAIL"
    if arm_a["inert"] or arm_b["inert"]:
        sc4 = "NO VERDICT"
    else:
        sc4 = "HOLD" if ((arm_a["part_i"] == arm_a["part_ii"])
                         and (arm_b["part_i"] == arm_b["part_ii"])) else "NO VERDICT"

    print(f"\n   SC1 (Arm A NOT certified)                         -> {sc1}")
    print(f"   SC2 (Arm B certified)                             -> {sc2}")
    print("   SC3 (no target rate is an input)                  -> HOLD")
    print(f"   SC4 (part (ii) not inert, and the parts agree)    -> {sc4}")
    print(f"   SC5 (the ladder reached its target)               -> {sc5}")
    print(f"   SC6 (the shadow trains on what HELD is scored at) -> {sc6}")
    return 0 if (sc1 == "HOLD" and sc2 == "HOLD" and sc6 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
