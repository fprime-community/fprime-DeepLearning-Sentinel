#!/usr/bin/env python3
"""Section 64, SR1-SR4: the pre-launch sanity band, in two arms.

D74.2 gives `Objective.md:1066`'s "measurably better" a two-part operational reading, and
`docs/MODELS.md` 64.3 runs it in two arms because **a criterion that certifies a shadow on
stationary data is broken** and only both arms test that.

(!) STOP 38: no target, budgeted or desired alarm rate is an input to anything here. The
held-out rate and the sensitivity curve are REPORTED and steer nothing.
(!) STOP 39: the 2.0% margin and the 0.060 drift scalar are 64.3's and do not move.
(!) STOP 40: `src/sentinel_eval/synthetic.py` is not modified; 64.3 records why.

Nothing here trains, forecasts or calibrates anything of its own: the forecaster is
`GRUForecastDetector`, the statistic is `sentinel_toolkit.statistic` and the calibration is
`sentinel_toolkit.calibrate`, all exactly as `src/sentinel_toolkit/fit.py:100-121` drives
them.
"""
from __future__ import annotations

import sys

import numpy as np

from sentinel_eval.detector import Context
from sentinel_models import telemanom
from sentinel_models.detectors import GRUForecastDetector
from sentinel_toolkit import statistic
from sentinel_toolkit.calibrate import DEFAULT_QUANTILE, calibrate
from sentinel_toolkit.fit import FLOWN, _hyper
from sentinel_toolkit.limits import FLIGHT_ERROR_WINDOW
from sentinel_toolkit.selftest import healthy_run

# --- 64.3's declared constants. Stop 39 forbids moving any of them. ------------------
FIXTURE_STEPS = 300_000
SEED = 0
EARLY_FRAC, LATE_FRAC = 0.40, 0.40
DRIFT_SCALAR = 0.060            # terminal multiplicative gain
DRIFT_CHANNELS = (0, 1)
MARGIN = 0.98                   # shadow residual must be <= 0.98x flying's
QUANTILE = DEFAULT_QUANTILE
WARMUP = FLOWN["window"] + FLIGHT_ERROR_WINDOW


def apply_drift(values: np.ndarray, start: int) -> np.ndarray:
    """64.3's Arm B: a linear gain ramp 1.000 -> 1.000+DRIFT_SCALAR from `start` onward.

    Applied to the extracted healthy run, NOT to the generator (stop 40). A channel-wise
    multiplicative gain is what a degrading sensor produces, and it is the same one-scalar
    discipline 42's testbed uses for a fault.
    """
    out = values.astype(np.float32).copy()
    n = len(out) - start
    ramp = np.linspace(0.0, DRIFT_SCALAR, n, dtype=np.float32)
    for c in DRIFT_CHANNELS:
        out[start:, c] = out[start:, c] * (1.0 + ramp)
    return out


def train(values: np.ndarray, names, label: str):
    """`fit.py:102-107`'s detector, fitted on `values` and nothing else."""
    hyper = _hyper(FLOWN["window"], FLOWN["hidden"], FLOWN["n_predictions"],
                   FLOWN["max_epochs"], SEED)
    det = GRUForecastDetector(hyper=hyper, reuse_weights=False,
                              config=telemanom.Config(error_window=FLIGHT_ERROR_WINDOW))
    ctx = Context(mission="s64", channels=list(names), groups=(1,) * values.shape[1],
                  period_seconds=1, fold=0, window=(0, len(values)))
    det.fit(values, np.ones(len(values), dtype=bool), ctx)
    return det


def forecast_residual(det, values: np.ndarray, names) -> float:
    """Criterion (i): mean absolute ONE-STEP-AHEAD forecast error, over the settled region.

    `detectors.py:466-467` is `errors = np.abs(filled[lo:hi] - forecast)`, then an EWMA.
    **The EWMA is the statistic's, not the forecaster's**, so it is deliberately omitted:
    this measures how well the model predicts, which is the claim retraining exists to make.
    The settled region is the same `[WARMUP:]` the calibration uses, so both parts of the
    criterion are measured on identical timesteps.
    """
    filled = det._filled(values)
    steps = filled.shape[0]
    block = det.chunks * det.chunk_steps
    err = np.empty_like(filled)
    for lo in range(0, steps, block):
        hi = min(lo + block, steps)
        err[lo:hi] = np.abs(filled[lo:hi] - det._forecast(filled, lo, hi))
    return float(err[WARMUP:].mean())


def calibrate_on(det, values: np.ndarray, names):
    """`fit.py:110-121` exactly: smoothed -> fused -> settled -> two halves -> own cut."""
    ctx = Context(mission="s64", channels=list(names), groups=(1,) * values.shape[1],
                  period_seconds=1, fold=0, window=(0, len(values)))
    smoothed = np.asarray(det._smoothed_errors(values, ctx))
    fused = statistic.reduce_across_channels(
        statistic.fused_per_channel(smoothed, values, FLIGHT_ERROR_WINDOW))
    settled = fused[WARMUP:]
    half = len(settled) // 2
    return calibrate(settled[:half], settled[half:],
                     span=FLIGHT_ERROR_WINDOW, quantile=QUANTILE)


def run_arm(name: str, early, late, held, names):
    print(f"\n-- ARM {name}")
    flying = train(early, names, "flying")
    shadow = train(late, names, "shadow")

    f_res = forecast_residual(flying, held, names)
    s_res = forecast_residual(shadow, held, names)
    f_cal = calibrate_on(flying, held, names)
    s_cal = calibrate_on(shadow, held, names)

    f_dev, s_dev = abs(1.0 - f_cal.ratio), abs(1.0 - s_cal.ratio)
    part_i = s_res <= MARGIN * f_res
    part_ii = s_dev <= f_dev
    certified = part_i and part_ii

    print(f"   held-out timesteps {len(held):,}; settled {len(held) - WARMUP:,} "
          f"(warm-up {WARMUP:,} = window {FLOWN['window']} + span {FLIGHT_ERROR_WINDOW:,})")
    print(f"   (i)  forecast residual   flying {f_res:.8f}   shadow {s_res:.8f}   "
          f"ratio {s_res / f_res:.4f}   need <= {MARGIN:.2f}  -> {'PASS' if part_i else 'FAIL'}")
    print(f"   (ii) |1 - holdout/fit|   flying {f_dev:.4f}       shadow {s_dev:.4f}       "
          f"need shadow <= flying                -> {'PASS' if part_ii else 'FAIL'}")
    print("   REPORTED, NEVER TARGETED -- these steer nothing:")
    print(f"        flying  cut {f_cal.cut:.6f}  fit {100*f_cal.fit_rate:.4f}%  "
          f"held-out {100*f_cal.holdout_rate:.4f}%  ratio {f_cal.ratio:.2f}")
    print(f"        shadow  cut {s_cal.cut:.6f}  fit {100*s_cal.fit_rate:.4f}%  "
          f"held-out {100*s_cal.holdout_rate:.4f}%  ratio {s_cal.ratio:.2f}")
    print(f"   CERTIFIED: {certified}")
    return dict(part_i=part_i, part_ii=part_ii, certified=certified,
                f_res=f_res, s_res=s_res, f_dev=f_dev, s_dev=s_dev,
                f_cal=f_cal, s_cal=s_cal)


def main() -> int:
    print("== Section 64, SR1-SR4: the pre-launch sanity band ==")
    values, names = healthy_run(seed=SEED, steps=FIXTURE_STEPS)
    n = len(values)
    a, b = int(n * EARLY_FRAC), int(n * (EARLY_FRAC + LATE_FRAC))
    print(f"   fixture seed {SEED}, n={FIXTURE_STEPS:,} -> healthy {n:,} x {values.shape[1]} channels")
    print(f"   EARLY [0:{a:,}]  LATE [{a:,}:{b:,}]  HELD [{b:,}:{n:,}]  (contiguous, in time order)")
    print(f"   criterion D74.2: (i) residual <= {MARGIN:.2f}x  AND  (ii) |1-ratio| no worse")

    arm_a = run_arm("A (stationary)", values[:a], values[a:b], values[b:], names)

    drifted = apply_drift(values, a)
    print(f"\n   ARM B drift: channels {DRIFT_CHANNELS}, gain 1.000 -> {1+DRIFT_SCALAR:.3f} "
          f"linearly from index {a:,} to {n:,} (LATE and HELD); EARLY untouched")
    arm_b = run_arm("B (declared drift)", values[:a], drifted[a:b], drifted[b:], names)

    sr1 = "HOLD" if not arm_a["certified"] else "FAIL"
    if arm_b["certified"]:
        sr2 = "HOLD"
    elif arm_b["part_i"] or arm_b["part_ii"]:
        sr2 = "NO VERDICT"
    else:
        sr2 = "FAIL"
    agree = ((arm_a["part_i"] == arm_a["part_ii"]) and (arm_b["part_i"] == arm_b["part_ii"]))
    sr4 = "HOLD" if agree else "NO VERDICT"

    print(f"\n   SR1 (Arm A not certified) -> {sr1}")
    print(f"   SR2 (Arm B certified)     -> {sr2}")
    print(f"   SR4 (the two parts agree) -> {sr4}")
    return 0 if (sr1 == "HOLD" and sr2 == "HOLD") else 1


if __name__ == "__main__":
    sys.exit(main())
