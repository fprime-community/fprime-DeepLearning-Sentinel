#!/usr/bin/env python3
"""Section 76, CT1-CT6: part (i) against a CONTROL SHADOW, per D78.

`docs/MODELS.md` 74.7 measured that every warm-started shadow beats the flying
model -- including one retrained on the flying model's OWN training segment, where
there is no drift and no new data. **Part (i) was measuring the retraining.**

D78 replaces it with a comparison that cancels the training advantage by
construction: the candidate against a CONTROL that differs from it in exactly one
thing, the data it was retrained on.

    control    C  warm from the flying weights, retrained on the FLYING MODEL'S
                  OWN segment, same step budget
    candidate  S  warm from the same weights, retrained on the new segment
    floor      F  the control-vs-control spread -- N controls on the flying
                  model's segment differing only in SEED -- MEASURED FIRST
    margin     m  = K * F, K = 2.0330 from 67.3, imported
    part (i)      certifies iff (res_held(C) - res_held(S)) / res_held(C) >= m

(!) STOP 46: the floor is measured before the margin is named. Only F is measured
here; K and the formula are 67.3's and are imported.
(!) STOP 44: the drift multiple is 67.3's ladder and target, imported, not chosen
after an arm is seen.
(!) STOP 45: N, L, HELD and the segmentation are 69's and do not move.
(!) D78 c.2: Arm G's 6.5687% and Arm W's 27.3251% are NOT the floor here. They
measure spreads of a comparison this criterion no longer makes, and they are
retired as floors, not as measurements.
(!) D78 c.1: part (ii) is D74.4's, UNCHANGED -- candidate against flying. The
candidate-against-control form is printed REPORTED, NEVER TARGETED, so the next
decision has the number without this section taking it.

Runner:  PYTHONPATH=src .venv/bin/python scripts/s76_control_shadow.py
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


# --- 76.2's declared constants. Only F is measured. -------------------------------
FIXTURE_STEPS = 780_000
FIXTURE_SEED = 0
L = 18_000
N = 6
HELD_LEN = 9_000
FLYING_SEG, SHADOW_SEG = 0, 5
RAMP_SEG = 4
MIN_HEALTHY = N * L + HELD_LEN
K = 2.0330                   # 67.3, IMPORTED


def main() -> int:
    s74 = _load("s74_warm_start")
    s68 = _load("s68_sanity_plateau")
    s67 = _load("s67_sanity_rerun")
    s66 = _load("s66_noise_floor")
    s64 = s67._s64()
    target, ladder = s67.DRIFT_TARGET, s67.DRIFT_LADDER

    print("== Section 76: part (i) against a control shadow (D78) ==")
    print(f"   K {K:.4f} (67.3, IMPORTED). F is measured below, m = K * F follows.")
    print("   (!) The floor is measured BEFORE the margin is named. Stop 46.")

    values, names = s64.healthy_run(seed=FIXTURE_SEED, steps=FIXTURE_STEPS)
    n = len(values)
    print(f"\n   fixture n={FIXTURE_STEPS:,} -> healthy {n:,} x {values.shape[1]}")
    if n < MIN_HEALTHY:
        print(f"   STOP: {n:,} healthy, below the declared {MIN_HEALTHY:,}.")
        return 2
    held_lo = n - HELD_LEN
    seg = [values[i * L:(i + 1) * L] for i in range(N)]
    held = values[held_lo:]

    print("\n-- the flying model: cold fit, segment 1, seed 0 (69's, unchanged)")
    flying = s67.train_on(s64, seg[FLYING_SEG], names)
    fw = flying._weights
    print(f"   clean held-out residual {s64.forecast_residual(flying, held, names):.8f}")

    # ---- ARM C: the floor. N controls, one segment, seeds differ, NOTHING else. ----
    print(f"\n-- ARM C (the floor): {N} CONTROLS on the flying model's own segment,")
    print("   warm-started from the flying weights, seeds 0..5. They differ in the")
    print("   stochastic order of training and in nothing else.")
    controls, res_c = [], []
    for s in range(N):
        det = s74.train_warm(s64, seg[FLYING_SEG], names, fw, seed=s)
        r = s64.forecast_residual(det, held, names)
        controls.append(det)
        res_c.append(r)
        print(f"     seed {s}   residual {r:.8f}")
    c_spread = s66.spreads(np.array(res_c))
    F = c_spread["range_rel"]
    m = K * F
    print(f"   C (controls)  range {100 * F:.4f}%   std {100 * c_spread['std_rel']:.4f}%")
    print(f"\n   (!) F = {100 * F:.4f}%  ->  m = K * F = {100 * m:.4f}%")
    print(f"       For comparison and NOT as the floor (D78 c.2):")
    print(f"       Arm G 6.5687% (69.7, cold segments)   Arm W 27.3251% (74.7, warm segments)")
    print(f"       MATCHED comparison: Arm S 31.9283% (69.7, COLD, one segment, seeds)")
    # CT1's comparison is the MATCHED one: 69.7's Arm S is the same experiment
    # done COLD -- one segment, seeds differing, nothing else -- so the only
    # difference is where the fit started. 31.9283%.
    ct1 = "HOLD" if F < 0.319283 else "FAIL"

    control = controls[0]                      # seed 0, the control of record
    c_res = res_c[0]

    # ---- the drift, sized by the flying model alone, no fit (67.3's ladder) --------
    clean = s64.forecast_residual(flying, held, names)
    ramp_from, plateau_from = RAMP_SEG * L, SHADOW_SEG * L
    print(f"\n-- DRIFT SIZING (flying only, no fit). ramp [{ramp_from:,}:{plateau_from:,}), FLAT")
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
    ct5 = "HOLD" if abs(seen - tested) <= 1e-4 else "FAIL"
    print(f"\n-- CT5: shadow trains at {seen:.6f}, HELD scored at {tested:.6f}  -> {ct5}")

    def arm(label, candidate, cand_fit, scored_held, ctrl, ctrl_res):
        """Part (i) against the CONTROL (D78). Part (ii) is D74.4's, unchanged."""
        print(f"\n-- ARM {label}")
        s_res = s64.forecast_residual(candidate, scored_held, names)
        c_r = s64.forecast_residual(ctrl, scored_held, names)
        improvement = (c_r - s_res) / c_r
        part_i = improvement >= m
        print(f"   (i)  held-out residual   CONTROL {c_r:.8f}   candidate {s_res:.8f}")
        print(f"        improvement over CONTROL {100 * improvement:+.4f}%   "
              f"need >= {100 * m:.4f}%   -> {'PASS' if part_i else 'FAIL'}")

        f_res = s64.forecast_residual(flying, scored_held, names)
        print(f"        (the OLD part (i), against the FLYING model, for contrast only: "
              f"{100 * (1 - s_res / f_res):+.4f}%)")

        f_gap, f_fit, f_held = s67.gap(s64, flying, seg[FLYING_SEG], scored_held, names)
        s_gap, s_fit, s_held = s67.gap(s64, candidate, cand_fit, scored_held, names)
        part_ii = s_gap <= f_gap
        print(f"   (ii) generalisation gap  flying {f_gap:.6f} (fit {f_fit:.8f} -> held "
              f"{f_held:.8f})")
        print(f"                            candidate {s_gap:.6f} (fit {s_fit:.8f} -> held "
              f"{s_held:.8f})")
        print(f"        D74.4, UNCHANGED: need candidate <= flying      -> "
              f"{'PASS' if part_ii else 'FAIL'}")
        c_gap, _, _ = s67.gap(s64, ctrl, seg[FLYING_SEG], scored_held, names)
        print("   REPORTED, NEVER TARGETED -- part (ii) against the CONTROL instead:")
        print(f"        control {c_gap:.6f}   candidate {s_gap:.6f}   "
              f"would be {'PASS' if s_gap <= c_gap else 'FAIL'}")
        certified = part_i and part_ii
        print(f"   CERTIFIED: {certified}")
        return dict(part_i=part_i, part_ii=part_ii, certified=certified,
                    improvement=improvement)

    print("\n   ARM A: candidate on a later HEALTHY segment. No drift. Must NOT certify.")
    arm_a = arm("A (stationary)", s74.train_warm(s64, seg[SHADOW_SEG], names, fw),
                seg[SHADOW_SEG], held, control, c_res)

    print(f"\n   ARM B: candidate on the DRIFTED segment, all {values.shape[1]} channels,"
          f" flat gain {1 + chosen:.3f}. Must certify.")
    drift_control = s74.train_warm(s64, seg[FLYING_SEG], names, fw)
    arm_b = arm("B (learnable drift)",
                s74.train_warm(s64, drifted[sha_lo:sha_hi], names, fw),
                drifted[sha_lo:sha_hi], drifted[held_lo:], drift_control, None)

    ct2 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        ct3 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        ct3 = "NO VERDICT"
    else:
        ct3 = "FAIL"
    ct6 = "HOLD" if abs(arm_a["improvement"]) < 0.525689 else "FAIL"

    print(f"\n   CT1 (warm seed spread < 69.7 Arm S's cold 31.9283%)  -> {ct1}")
    print(f"   CT2 (Arm A, stationary, NOT certified)             -> {ct2}")
    print(f"   CT3 (Arm B, drifted, IS certified)                 -> {ct3}")
    print("   CT4 (no target rate is an input)                   -> HOLD")
    print(f"   CT5 (shadow trains on what HELD is scored at)      -> {ct5}")
    print(f"   CT6 (the control cancels the training advantage:   -> {ct6}")
    print(f"        |Arm A improvement| {100 * abs(arm_a['improvement']):.4f}% < 52.5689%,")
    print("        74.7's SMALLEST improvement over the flying model)")
    return 0 if (ct2 == "HOLD" and ct3 == "HOLD" and ct5 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
