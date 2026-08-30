#!/usr/bin/env python3
"""Measure what a real run will cost, before spending an operation on it.

Objective.md 12 makes Phase 1 the phase where changing your mind is supposed to
be cheap. That only holds if we know what an experiment costs in wall-clock, and
the honest way to find out is to measure the real shapes rather than to estimate
them: `m1-g8.9.10` is 14,728,316 timesteps of twelve channels, and the harness
scores the whole training window as well as the test fold, so a run covers 33.1M
timesteps per channel set.

**Zero R2 operations, nothing on disk.** The arrays are generated locally at the
exact shapes the real folds have. That measures compute, which is the thing in
question; it says nothing about the data, which is not.

    python scripts/bench_forecaster.py            # folds 0 and 2, real config
    python scripts/bench_forecaster.py --quick    # a tenth of the sequences
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_eval.detector import Context                      # noqa: E402
from sentinel_models import telemanom                           # noqa: E402
from sentinel_models.detectors import ForecastDetector, clear_caches  # noqa: E402
from sentinel_models.lstm import Hyper                          # noqa: E402

#: The real geometry, from `runs/m1-g8.9.10/rstd/`: 14,728,316 steps of 30s,
#: forward chaining with a 25% seed and three folds.
GRID = 14_728_316
FOLDS = [(0, 3_682_079), (0, 7_364_158), (0, 11_046_237)]
TEST = 3_682_079
CHANNELS = 12


def synthetic_series(steps: int, channels: int, seed: int = 0) -> np.ndarray:
    """Correlated channels at ESA-ADB's amplitude, so the model has real work."""
    rng = np.random.default_rng(seed)
    driver = np.cumsum(rng.standard_normal(steps)) * 0.001
    lags = rng.integers(1, 40, size=channels)
    out = np.empty((steps, channels), dtype=np.float32)
    for c in range(channels):
        out[:, c] = np.roll(driver, int(lags[c])) * (0.5 + c / channels)
        out[:, c] += 0.02 * rng.standard_normal(steps)
    return np.clip(out - out.min(), 0.0, None) / max(1e-9, float(np.ptp(out)))


def bench_fold(index: int, quick: bool) -> dict:
    train_lo, train_hi = FOLDS[index]
    hyper = Hyper(sequence_budget_divisor=1800 if quick else 180)
    detector = ForecastDetector(hyper=hyper)

    values = synthetic_series(train_hi + TEST, CHANNELS, seed=index)
    usable = np.ones(values.shape[0], dtype=bool)
    context = Context(mission="mission1", channels=tuple(f"channel_{41 + i}" for i in range(CHANNELS)),
                      groups=(8, 9, 10), period_seconds=30, fold=index,
                      window=(train_lo, train_hi))

    print(f"\n  fold {index}: train {train_hi - train_lo:,} steps, "
          f"{hyper.sequences_per_epoch(train_hi - train_lo):,} sequences/epoch, "
          f"up to {hyper.max_epochs} epochs")

    started = time.perf_counter()
    detector.fit(values[train_lo:train_hi], usable[train_lo:train_hi], context)
    fit_seconds = time.perf_counter() - started
    epochs = detector.report["epochs_run"]
    print(f"    fit                 {fit_seconds / 60:7.2f} min   "
          f"({epochs} epochs, {fit_seconds / max(epochs, 1):.1f}s each)")

    # The harness scores the training window as well, to pick an operating point.
    sample = min(TEST, 1_000_000)
    clear_caches()
    started = time.perf_counter()
    detector._smoothed_errors(values[:sample], context)
    forecast_seconds = time.perf_counter() - started
    per_step = forecast_seconds / sample

    smoothed = np.abs(np.random.default_rng(0).standard_normal((200_000, CHANNELS))
                      ).astype(np.float32) * 0.05
    started = time.perf_counter()
    telemanom.combine(smoothed, detector.config)
    ndt_per_step = (time.perf_counter() - started) / 200_000

    scored = (train_hi - train_lo) + TEST
    print(f"    forecast + smooth   {per_step * 1e6:7.2f} us/step  "
          f"-> {per_step * scored / 60:6.2f} min for this fold's {scored:,} steps")
    print(f"    dynamic threshold   {ndt_per_step * 1e6:7.2f} us/step  "
          f"-> {ndt_per_step * scored / 60:6.2f} min")
    clear_caches()
    return {"fit": fit_seconds, "per_step": per_step, "ndt_per_step": ndt_per_step,
            "epochs": epochs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="a tenth of the sequence budget")
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 2])
    args = parser.parse_args()

    print(f"  BENCHMARK -- generated arrays at the real fold shapes, zero R2 operations")
    print(f"  grid {GRID:,} steps of 30s, {CHANNELS} channels"
          f"{'   [QUICK: a tenth of the sequences]' if args.quick else ''}")

    results = {i: bench_fold(i, args.quick) for i in args.folds}

    # Extrapolate the whole paired run: three folds on twelve channels, then the
    # same three folds on the six-channel subset. The fits must be scaled by each
    # fold's sequence budget -- that scaling is the entire point of the budget, so
    # reusing one fold's time for all three would understate the run badly.
    hyper = Hyper(sequence_budget_divisor=1800 if args.quick else 180)
    usable = {i: hi - lo for i, (lo, hi) in enumerate(FOLDS)}
    budgets = {i: hyper.sequences_per_epoch(n) for i, n in usable.items()}
    per_sequence = np.mean([r["fit"] / (r["epochs"] * budgets[i])
                            for i, r in results.items()])
    epochs = int(np.mean([r["epochs"] for r in results.values()]))

    fits = {i: (results[i]["fit"] if i in results
                else per_sequence * epochs * budgets[i]) for i in range(3)}
    per_step = np.mean([r["per_step"] for r in results.values()])
    ndt = np.mean([r["ndt_per_step"] for r in results.values()])
    scored_steps = sum(hi - lo for lo, hi in FOLDS) + 3 * TEST

    print(f"\n  PROJECTED, one invocation over both paired sets")
    print(f"    per sequence {per_sequence * 1000:.2f} ms, early stop at ~{epochs} epochs")
    for i in range(3):
        measured = "measured" if i in results else "projected"
        print(f"      fold {i}: {budgets[i]:>6,} sequences/epoch   "
              f"fit {fits[i] / 60:6.2f} min   [{measured}]")
    # The six-channel set is barely cheaper: cost per timestep is 320*(C+240)
    # multiply-accumulates, so halving C moves it by about 2%.
    narrow = 0.98
    twelve_fit, six_fit = sum(fits.values()), sum(fits.values()) * narrow
    scoring = (per_step + ndt) * scored_steps
    print(f"    12-channel set: fits {twelve_fit / 60:6.2f} min"
          f"   scoring {scoring / 60:6.2f} min")
    print(f"     6-channel set: fits {six_fit / 60:6.2f} min"
          f"   scoring {scoring * narrow / 60:6.2f} min")
    print(f"    lstm-quantile reuses every fold's weights and skips the dynamic "
          f"threshold: +{per_step * scored_steps * (1 + narrow) / 60:.1f} min")

    projected = (twelve_fit + six_fit + scoring * (1 + narrow)
                 + per_step * scored_steps * (1 + narrow)) / 60
    print(f"\n    TOTAL  ~{projected:.0f} min"
          f"{'  (a quick run understates the fit; scale by ten)' if args.quick else ''}")
    print("    folds not benchmarked are projected from the measured "
          "per-sequence cost and that fold's budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
