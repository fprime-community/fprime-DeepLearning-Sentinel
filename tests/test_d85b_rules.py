"""D85b / docs/MODELS.md 81: the runner's registered rules, each shown in both directions.

- T = 1 + max(1.5 x (R - 1), 0.02), including its floor.
- The rate-normal check is two-sided at +/-5% on |dx|'s 99.9th percentile and maximum: a
  channel whose steps grow 6% fails, one whose steps shrink 6% fails, and 4% passes. A
  healthy statistic of zero passes only if the aged one is zero too.
- A candidate is presented to the gate only when its need window lies after the baseline
  and both HELD windows fit the run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d85b_arms", ROOT / "scripts" / "d85b_arms.py")
arms = importlib.util.module_from_spec(_spec)
sys.modules["d85b_arms"] = arms
_spec.loader.exec_module(arms)


def test_the_threshold_rule_and_its_floor() -> None:
    assert abs(arms.threshold(1.04) - 1.06) < 1e-12
    assert arms.threshold(1.001) == 1.02            # the floor
    assert arms.threshold(0.99) == 1.02             # R below 1 cannot lower it


def _trace(scale: float = 1.0, n: int = 2_000) -> np.ndarray:
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.uniform(-1, 1, size=(n, 8)), axis=0)
    base[:, 6] = 0.0                                 # a channel that never moves
    return base * scale


def test_rate_normal_is_two_sided_at_five_percent() -> None:
    h = _trace()
    ok = lambda a: all(r["ok"] for r in arms.rate_normal(a, h, 100, 2_000))
    assert ok(_trace(1.04)) and ok(_trace(0.96)) and ok(h)
    assert not ok(_trace(1.06)), "steps 6% larger must fail"
    assert not ok(_trace(0.94)), "steps 6% smaller must fail (two-sided)"


def test_a_channel_that_starts_moving_is_not_rate_normal() -> None:
    h = _trace()
    a = h.copy()
    a[500:, 6] = np.arange(1_500) * 1e-3            # the still channel now drifts
    rows = arms.rate_normal(a, h, 100, 2_000)
    assert not rows[6]["ok"] and all(r["ok"] for i, r in enumerate(rows) if i != 6)


def test_the_gateable_geometry() -> None:
    n = arms.ARM_TICKS
    first_ok = 2 * arms.NEED_SPAN - 1                 # last training tick: need window starts at the baseline's end
    assert arms.gateable({"last_data_tick": first_ok}, n)
    assert not arms.gateable({"last_data_tick": first_ok - 1}, n)
    last_ok = n - 2 * arms.HELD_SPAN - 1
    assert arms.gateable({"last_data_tick": last_ok}, n)
    assert not arms.gateable({"last_data_tick": last_ok + 1}, n)


def test_the_constants_are_81s() -> None:
    assert (arms.NEED_SPAN, arms.HELD_SPAN) == (23_950, 13_150)
    assert arms.LADDER == (0.02, 0.04, 0.06) and arms.C_SEEDS == (4, 5, 6)
    assert (arms.OCV_START, arms.OCV_TAU, arms.B_ONSET) == (30_000, 20_000.0, 150_000)
    assert not set(arms.C_SEEDS) & set(arms.CAL_SEEDS), "a calibration seed evaluates C1"
