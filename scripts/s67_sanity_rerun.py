#!/usr/bin/env python3
"""Section 67, SB1-SB5: the sanity band re-run, with a margin derived from 66's floor.

64.7 recorded the criterion certifying a shadow on stationary data, for two reasons: a
2.0% margin that was declared rather than derived, and a second part that could not tell
two models apart because `src/sentinel_toolkit/statistic.py:65` fuses
`max(z_residual, z_derivative)` and the derivative set the maximum in 7 of 7 steps at the
top quantile. Both are replaced here. **64.3's constants are not touched, read or moved**
-- stop 39 binds section 64 and is honoured by leaving it alone.

(!) STOP 38: no target, budgeted or desired alarm rate is an input to anything here.
(!) STOP 40: `src/sentinel_eval/synthetic.py` is not modified.
(!) STOP 44: every constant below is declared in 67.3, BEFORE these arms run, and derived
from 66.7's committed floor. None moves afterwards.
(!) STOP 45: 66's N, L, HELD and F are used exactly as 66 measured them.

The forecaster, the fit, the residual and the calibration are 64's, loaded from
`scripts/s64_sanity_band.py` rather than copied, so nothing can drift from what 64.7
measured.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _s64():
    path = ROOT / "scripts" / "s64_sanity_band.py"
    spec = importlib.util.spec_from_file_location("s64_sanity_band", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 67.3's declared constants, derived from 66.7. Stop 44 forbids moving any. ------
FIXTURE_STEPS, FIXTURE_SEED = 300_000, 0
HELD_LEN, L, N = 9_000, 6_000, 6         # 66.3's segmentation, unchanged
FLYING_SEG, SHADOW_SEG = 0, 5            # oldest, and the one immediately before HELD
SEED = 0

FLOOR = 0.298606                         # 66.7's F, as committed
D2_N6, Z99 = 2.534, 2.5758               # see 67.3 for the derivation of K
K = (2 * Z99) / D2_N6                    # 2.0330
MARGIN = K * FLOOR                       # 0.6071 -- required relative improvement
DRIFT_HEADROOM = 1.2
DRIFT_TARGET = DRIFT_HEADROOM / (1 - MARGIN)          # 3.0539
DRIFT_LADDER = (0.060, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00)


def apply_drift(values: np.ndarray, start: int, gain: float) -> np.ndarray:
    """A linear gain ramp 1.000 -> 1.000+gain from `start` to the end, ALL channels.

    64.3 drifted 2 of 7 channels and 64.7 recorded the cost: "a 6% terminal gain on 2 of
    7 channels is averaged across all seven", moving the held-out residual by 0.03%. All
    seven here, for that reason, and the reason is 64.7's own.
    """
    out = values.astype(np.float32).copy()
    ramp = np.linspace(0.0, gain, len(out) - start, dtype=np.float32)
    out[start:] = out[start:] * (1.0 + ramp)[:, None]
    return out


def residual_only_ratio(s64, det, values: np.ndarray, names) -> float:
    """ROUTE (a), REPORTED AND NEVER GATING: calibration on the residual channel alone.

    The same two-half calibration `fit.py:110-121` runs, with `zstat(smoothed_error)`
    in place of `max(z_residual, z_derivative)`. It is computed because it costs one
    call on data already in hand, and reported because 64.7's finding deserves a number
    beside it -- not because anything here reads it.
    """
    from sentinel_eval.detector import Context
    from sentinel_toolkit import statistic
    from sentinel_toolkit.calibrate import calibrate
    ctx = Context(mission="s67", channels=list(names), groups=(1,) * values.shape[1],
                  period_seconds=1, fold=0, window=(0, len(values)))
    smoothed = np.asarray(det._smoothed_errors(values, ctx))
    span = s64.FLIGHT_ERROR_WINDOW
    per_channel = np.empty(smoothed.shape, dtype=np.float64)
    for c in range(smoothed.shape[1]):
        per_channel[:, c] = statistic.zstat(smoothed[:, c], span)
    settled = statistic.reduce_across_channels(per_channel)[s64.WARMUP:]
    half = len(settled) // 2
    cal = calibrate(settled[:half], settled[half:], span=span, quantile=s64.QUANTILE)
    return abs(1.0 - cal.ratio)


def gap(s64, det, fit_segment: np.ndarray, held: np.ndarray, names) -> tuple[float, float, float]:
    """PART (ii), ADOPTED: the generalisation gap on the forecast residual.

    `|res_held - res_fit| / res_fit`. Model-dependent by construction, which is exactly
    what 64.7's part (ii) was not, and it tests a different failure from part (i):
    part (i) asks whether the model predicts well, this asks whether it predicts as well
    on data it never saw as on data it fitted.
    """
    res_fit = s64.forecast_residual(det, fit_segment, names)
    res_held = s64.forecast_residual(det, held, names)
    return abs(res_held - res_fit) / res_fit, res_fit, res_held


def train_on(s64, segment: np.ndarray, names, seed: int = SEED):
    from sentinel_eval.detector import Context
    from sentinel_models import telemanom
    from sentinel_models.detectors import GRUForecastDetector
    hyper = s64._hyper(s64.FLOWN["window"], s64.FLOWN["hidden"],
                       s64.FLOWN["n_predictions"], s64.FLOWN["max_epochs"], seed)
    det = GRUForecastDetector(hyper=hyper, reuse_weights=False,
                              config=telemanom.Config(error_window=s64.FLIGHT_ERROR_WINDOW))
    ctx = Context(mission="s67", channels=list(names), groups=(1,) * segment.shape[1],
                  period_seconds=1, fold=0, window=(0, len(segment)))
    det.fit(segment, np.ones(len(segment), dtype=bool), ctx)
    return det


def run_arm(s64, label, flying, shadow, fly_fit, sha_fit, held, names):
    print(f"\n-- ARM {label}")
    f_res = s64.forecast_residual(flying, held, names)
    s_res = s64.forecast_residual(shadow, held, names)
    part_i = s_res <= (1.0 - MARGIN) * f_res

    f_gap, f_fit_r, f_held_r = gap(s64, flying, fly_fit, held, names)
    s_gap, s_fit_r, s_held_r = gap(s64, shadow, sha_fit, held, names)
    part_ii = s_gap <= f_gap

    print(f"   (i)  held-out residual   flying {f_res:.8f}   shadow {s_res:.8f}")
    print(f"        improvement {100 * (1 - s_res / f_res):+.4f}%   "
          f"need >= {100 * MARGIN:.4f}%   -> {'PASS' if part_i else 'FAIL'}")
    print(f"   (ii) generalisation gap  flying {f_gap:.6f} (fit {f_fit_r:.8f} -> held "
          f"{f_held_r:.8f})")
    print(f"                            shadow {s_gap:.6f} (fit {s_fit_r:.8f} -> held "
          f"{s_held_r:.8f})")
    print(f"        need shadow <= flying                          -> "
          f"{'PASS' if part_ii else 'FAIL'}")
    inert = abs(f_gap - s_gap) < 1e-6
    print(f"        the two models differ on part (ii): {'NO -- INERT' if inert else 'yes'}")

    f_a = residual_only_ratio(s64, flying, held, names)
    s_a = residual_only_ratio(s64, shadow, held, names)
    print("   REPORTED, NEVER TARGETED -- route (a), |1 - holdout/fit| on the residual")
    print(f"        channel alone: flying {f_a:.6f}   shadow {s_a:.6f}")

    certified = part_i and part_ii
    print(f"   CERTIFIED: {certified}")
    return dict(part_i=part_i, part_ii=part_ii, certified=certified, inert=inert,
                f_res=f_res, s_res=s_res, f_gap=f_gap, s_gap=s_gap)


def main() -> int:
    s64 = _s64()
    print("== Section 67, SB1-SB5: the sanity band re-run ==")
    print(f"   floor F {100 * FLOOR:.4f}% (66.7)   K {K:.4f}   "
          f"margin {100 * MARGIN:.4f}%   drift target {DRIFT_TARGET:.4f}x")

    values, names = s64.healthy_run(seed=FIXTURE_SEED, steps=FIXTURE_STEPS)
    n = len(values)
    held_lo = n - HELD_LEN
    fly_lo, fly_hi = FLYING_SEG * L, (FLYING_SEG + 1) * L
    sha_lo, sha_hi = SHADOW_SEG * L, (SHADOW_SEG + 1) * L
    print(f"   healthy {n:,} x {values.shape[1]}   FLYING [{fly_lo:,}:{fly_hi:,}]   "
          f"SHADOW [{sha_lo:,}:{sha_hi:,}]   HELD [{held_lo:,}:{n:,}]")

    flying = train_on(s64, values[fly_lo:fly_hi], names)

    print(f"\n-- DRIFT SIZING: smallest rung moving the FLYING model's held-out residual")
    print(f"   to >= {DRIFT_TARGET:.4f}x its clean value. No fit is involved, so this")
    print("   cannot tune the comparison. Ladder and target declared at 67.3.")
    clean = s64.forecast_residual(flying, values[held_lo:], names)
    print(f"     clean flying held-out residual {clean:.8f}")
    chosen = None
    for gain in DRIFT_LADDER:
        drifted = apply_drift(values, sha_lo, gain)
        r = s64.forecast_residual(flying, drifted[held_lo:], names)
        mark = ""
        if chosen is None and r >= DRIFT_TARGET * clean:
            chosen, mark = gain, "   <- TAKEN"
        print(f"     gain 1.000 -> {1 + gain:.3f}   residual {r:.8f}   "
              f"{r / clean:.4f}x{mark}")
    sb5 = "HOLD" if chosen is not None else "FAIL"
    if chosen is None:
        print("   (!) THE LADDER IS EXHAUSTED. No declared drift moves the residual far")
        print("       enough for the criterion to be satisfiable, and that is the finding.")
        chosen = DRIFT_LADDER[-1]

    arm_a = run_arm(s64, "A (stationary)", flying, train_on(s64, values[sha_lo:sha_hi], names),
                    values[fly_lo:fly_hi], values[sha_lo:sha_hi], values[held_lo:], names)

    drifted = apply_drift(values, sha_lo, chosen)
    print(f"\n   ARM B drift: ALL {values.shape[1]} channels, gain 1.000 -> {1 + chosen:.3f}")
    print(f"   linearly from index {sha_lo:,} to {n:,}; the flying model's segment untouched")
    arm_b = run_arm(s64, "B (declared drift)", flying,
                    train_on(s64, drifted[sha_lo:sha_hi], names),
                    values[fly_lo:fly_hi], drifted[sha_lo:sha_hi], drifted[held_lo:], names)

    sb1 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        sb2 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        sb2 = "NO VERDICT"
    else:
        sb2 = "FAIL"
    if arm_a["inert"] or arm_b["inert"]:
        sb4 = "NO VERDICT"
    else:
        sb4 = "HOLD" if ((arm_a["part_i"] == arm_a["part_ii"])
                         and (arm_b["part_i"] == arm_b["part_ii"])) else "NO VERDICT"

    print(f"\n   SB1 (Arm A NOT certified)            -> {sb1}")
    print(f"   SB2 (Arm B certified)                -> {sb2}")
    print("   SB3 (no target rate is an input)     -> HOLD")
    print(f"   SB4 (part (ii) is not inert, and the parts agree) -> {sb4}")
    print(f"   SB5 (the drift ladder reached its target)         -> {sb5}")
    return 0 if (sb1 == "HOLD" and sb2 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
