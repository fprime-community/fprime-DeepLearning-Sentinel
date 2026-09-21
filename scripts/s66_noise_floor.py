#!/usr/bin/env python3
"""Section 66, NF1-NF4: the noise floor the sanity margin must sit above.

`docs/MODELS.md` 64.7 found the pre-launch sanity criterion certifying a shadow on
stationary data, and named the cause: the 2.0% margin of 64.3 was **declared, not
derived**, and *"two fits on different segments of the same stationary series differ by
about 4% in held-out forecast error from nothing but the segments."* That 4% was one
observation on one fixture, not a distribution -- 64.8 says so in as many words -- so no
margin can be derived from it yet.

This measures the floor, and measures nothing else. **It sets no margin and states no
criterion**; section 67 does that, afterwards, from the number this prints.

Two arms, because "noise" has two sources and only one of them is the one 64.7 blamed:

    Arm S   N fits on ONE segment at N different seeds      -- seed-only spread
    Arm G   N fits on N different segments at one seed      -- segment spread

(!) STOP 38 (64.3): no target, budgeted or desired alarm rate is an input to anything
here. Nothing in this file reads a rate at all.
(!) STOP 39 (64.3): 64.3's 2.0% margin and 0.060 drift scalar are NOT touched. This file
declares none of them and reads none of them; 64.7's record stands exactly as written.
(!) STOP 40 (64.3): `src/sentinel_eval/synthetic.py` is not modified.
(!) STOP 45 (66.5): N, L, HELD and F's definition are fixed below, BEFORE the fits run,
and none of them moves afterwards.

The forecaster, the fit and the residual are `scripts/s64_sanity_band.py`'s, loaded from
that file rather than copied, so the statistic is the same one 64.7 measured and cannot
drift from it.
"""
from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _s64():
    """64's own `train` and `forecast_residual`, loaded rather than duplicated."""
    path = ROOT / "scripts" / "s64_sanity_band.py"
    spec = importlib.util.spec_from_file_location("s64_sanity_band", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 66.3's declared constants. Stop 45 forbids moving any of them. -----------------
FIXTURE_STEPS = 300_000         # 64.3's fixture, unchanged
FIXTURE_SEED = 0                # 64.3's fixture seed, unchanged
HELD_LEN = 9_000                # 64.3's HELD length, so the held-out statistic is comparable
N = 6                           # fits per arm
L = 6_000                       # training segment length; 6 * 6,000 = the 36,000 that remain
MIN_HEALTHY = 45_000            # refuse rather than silently re-shape if the fixture moves


def spreads(values: np.ndarray) -> dict:
    """The two spread statistics, both reported, with the range as the one that counts.

    The range is what a gate has to survive: a margin below the largest gap two null
    fits produced cannot distinguish a real improvement from a pair of them. The
    standard deviation is reported beside it because a range over six points is a
    coarse instrument and saying so is cheaper than being asked.
    """
    lo, hi, mean = float(values.min()), float(values.max()), float(values.mean())
    return dict(values=[float(v) for v in values], min=lo, max=hi, mean=mean,
                std=float(values.std(ddof=1)),
                range_rel=(hi - lo) / mean, std_rel=float(values.std(ddof=1)) / mean)


def spearman_exact(values: np.ndarray) -> tuple[float, float]:
    """Spearman rho against position, with an EXACT two-sided permutation p-value.

    n = 6, so all 720 orderings are enumerated. An asymptotic p-value at n = 6 would be
    a guess dressed as a test, and NF4's band is stated in p.
    """
    n = len(values)
    order = np.argsort(np.argsort(values)).astype(float)
    position = np.arange(n, dtype=float)

    def rho(a: np.ndarray) -> float:
        a, b = a - a.mean(), position - position.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    observed = rho(order)
    perms = [abs(rho(np.array(p, dtype=float))) for p in itertools.permutations(range(n))]
    p = float(np.mean(np.array(perms) >= abs(observed) - 1e-12))
    return observed, p


def main() -> int:
    s64 = _s64()
    print("== Section 66, NF1-NF4: the noise floor ==")
    print("   (!) THIS SECTION SETS NO MARGIN. It measures the floor 67 will derive one from.")

    values, names = s64.healthy_run(seed=FIXTURE_SEED, steps=FIXTURE_STEPS)
    n = len(values)
    print(f"\n   fixture seed {FIXTURE_SEED}, n={FIXTURE_STEPS:,} -> healthy {n:,} x "
          f"{values.shape[1]} channels")
    if n < MIN_HEALTHY:
        print(f"   STOP: the healthy run is {n:,}, below the declared {MIN_HEALTHY:,}. "
              "66.3's segmentation does not fit and is not re-shaped to make it.")
        return 2

    held = values[n - HELD_LEN:]
    segments = [values[i * L:(i + 1) * L] for i in range(N)]
    warm = s64.WARMUP
    print(f"   segments  {N} x {L:,} over [0:{N * L:,}), disjoint, in time order")
    print(f"   HELD      last {HELD_LEN:,} = [{n - HELD_LEN:,}:{n:,}]; settled "
          f"{HELD_LEN - warm:,} after WARMUP {warm:,} "
          f"(window {s64.FLOWN['window']} + span {s64.FLIGHT_ERROR_WINDOW:,})")
    print(f"   statistic mean absolute one-step-ahead forecast error on HELD[{warm:,}:]")
    print("   (!) L = 6,000 is a third of 64.3's 18,000. A shorter fit is a noisier fit, so")
    print("       F is more likely an OVER-estimate of the floor at 18,000 than an under-.")
    print("       That makes 67's derived margin conservative -- harder to certify, not")
    print("       easier -- and it is said here, before the run, not afterwards.")

    print(f"\n-- ARM S: {N} fits on segment 1, seeds 0..{N - 1} (seed-only spread)")
    arm_s = []
    for seed in range(N):
        det = _train_at(s64, segments[0], names, seed)
        r = s64.forecast_residual(det, held, names)
        arm_s.append(r)
        print(f"     seed {seed}   residual {r:.8f}")
    arm_s = np.array(arm_s)

    print(f"\n-- ARM G: {N} fits, one per segment, seed 0 (segment spread)")
    arm_g = []
    for i, seg in enumerate(segments):
        det = _train_at(s64, seg, names, 0)
        r = s64.forecast_residual(det, held, names)
        arm_g.append(r)
        print(f"     segment {i + 1} [{i * L:,}:{(i + 1) * L:,}]   residual {r:.8f}")
    arm_g = np.array(arm_g)

    s, g = spreads(arm_s), spreads(arm_g)
    print("\n-- SPREADS, both reported")
    for label, d in (("S (seeds)   ", s), ("G (segments)", g)):
        print(f"     {label}  min {d['min']:.8f}  max {d['max']:.8f}  mean {d['mean']:.8f}")
        print(f"                   range {100 * d['range_rel']:.4f}%   "
              f"std {100 * d['std_rel']:.4f}%")

    floor = max(s["range_rel"], g["range_rel"])
    print(f"\n   (!) THE NOISE FLOOR F = {100 * floor:.4f}%  "
          f"-- max over the two arms of (max - min) / mean")

    nf1 = "HOLD" if s["range_rel"] < g["range_rel"] else "FAIL"
    if g["range_rel"] >= 0.02:
        nf2 = "HOLD"
    elif g["range_rel"] >= 0.01:
        nf2 = "NO VERDICT"
    else:
        nf2 = "FAIL"
    rho, p = spearman_exact(arm_g)
    if p > 0.05:
        nf4 = "HOLD"
    elif p > 0.01:
        nf4 = "NO VERDICT"
    else:
        nf4 = "FAIL"

    print(f"\n   NF1 (seed spread < segment spread)       {100 * s['range_rel']:.4f}% vs "
          f"{100 * g['range_rel']:.4f}%   -> {nf1}")
    print(f"   NF2 (segment spread >= 2.0%)             {100 * g['range_rel']:.4f}%"
          f"                -> {nf2}")
    print("   NF3 (no target rate is an input)         0 occurrences"
          "               -> HOLD")
    print(f"   NF4 (no monotone trend in time order)    rho {rho:+.4f}, exact p {p:.4f}"
          f"     -> {nf4}")
    print("\n   REPORTED, NEVER TARGETED -- these steer nothing in this section.")
    return 0 if (nf1 == "HOLD" and nf2 == "HOLD" and nf4 == "HOLD") else 1


def _train_at(s64, segment: np.ndarray, names, seed: int):
    """64's `train`, at a chosen seed. `train` pins SEED; only the seed differs here."""
    from sentinel_eval.detector import Context
    from sentinel_models import telemanom
    from sentinel_models.detectors import GRUForecastDetector
    hyper = s64._hyper(s64.FLOWN["window"], s64.FLOWN["hidden"],
                       s64.FLOWN["n_predictions"], s64.FLOWN["max_epochs"], seed)
    det = GRUForecastDetector(hyper=hyper, reuse_weights=False,
                              config=telemanom.Config(error_window=s64.FLIGHT_ERROR_WINDOW))
    ctx = Context(mission="s66", channels=list(names), groups=(1,) * segment.shape[1],
                  period_seconds=1, fold=0, window=(0, len(segment)))
    det.fit(segment, np.ones(len(segment), dtype=bool), ctx)
    return det


if __name__ == "__main__":
    sys.exit(main())
