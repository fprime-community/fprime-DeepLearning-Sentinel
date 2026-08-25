"""Trivial detectors -- the floor, the ceiling, and the one that must lose.

**The one that must lose.** Wu & Keogh (IEEE TKDE 2023) showed that a moving
average with a standard-deviation threshold achieves state-of-the-art on
SMAP/MSL, which is a large part of why Objective.md 9.2 demoted that benchmark:
if a two-line script wins, the benchmark is not measuring sophistication. So
:class:`MovingAverage` is scored first here, and it is scored with a prediction
already written down (docs/HARNESS.md):

* **headline-cell / contextual recall must be near-floor.** It looks at each
  channel alone, so it is blind to cross-channel structure by construction. This
  is the number that matters, and a strong score here means something is wrong
  with the benchmark, the loader or the harness -- stop and report.
* **point recall may legitimately be decent.** Point anomalies are single-channel
  spikes; that is a moving average's home turf. A good score is the right answer,
  not a red flag.
* **rare-event false alarms should be bad.** A commanded manoeuvre looks like a
  spike, and nothing here can tell the two apart.

**The floor and the ceiling.** :class:`AlwaysQuiet` and :class:`Oracle` exist to
prove the harness's own arithmetic: a detector that never fires must score zero
recall, and one handed the answers must reach 1.0. A referee that has never been
checked against its own extremes is untested machinery.
"""
from __future__ import annotations

import numpy as np

from sentinel_eval.detector import Detector

EPSILON = 1e-12


def _rolling(values: np.ndarray, window: int, want: str) -> np.ndarray:
    """NaN-aware rolling mean or standard deviation, trailing, in O(T).

    Trailing rather than centred: a detector that peeks at future samples is not
    something that can fly, and the harness should not measure one that does.
    """
    filled = np.nan_to_num(values, nan=0.0)
    present = np.isfinite(values).astype(np.float64)

    def trailing_sum(a: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
        upper = cumulative[1:]
        lower = np.concatenate([np.zeros((min(window, a.shape[0]),) + a.shape[1:]),
                                cumulative[1:max(1, a.shape[0] - window + 1)]])
        return upper - lower[: a.shape[0]]

    count = np.maximum(trailing_sum(present), 1.0)
    mean = trailing_sum(filled) / count
    if want == "mean":
        return mean
    second = trailing_sum(filled * filled) / count
    return np.sqrt(np.maximum(second - mean * mean, 0.0))


class MovingAverage(Detector):
    """|x - trailing mean(x)|, per channel. The Wu & Keogh one-liner."""

    name = "mavg"

    def __init__(self, window: int = 120) -> None:
        super().__init__(window=window)
        self.window = int(window)
        self._scale: np.ndarray | None = None

    @property
    def warmup_steps(self) -> int:
        return self.window

    def fit(self, values, usable, context) -> None:
        """Learn each channel's normal residual size, from nominal data only."""
        residual = np.abs(values - _rolling(values, self.window, "mean"))
        rows = np.asarray(usable, dtype=bool)
        sample = residual[rows] if rows.any() else residual
        with np.errstate(invalid="ignore"):
            self._scale = np.nanstd(sample, axis=0)

    def score(self, values, valid, context) -> np.ndarray:
        residual = np.abs(values - _rolling(values, self.window, "mean"))
        scale = self._scale if self._scale is not None else np.nanstd(residual, axis=0)
        return residual / np.maximum(scale, EPSILON)


class RollingStd(Detector):
    """Trailing standard deviation per channel. The other Wu & Keogh one-liner."""

    name = "rstd"

    def __init__(self, window: int = 120) -> None:
        super().__init__(window=window)
        self.window = int(window)
        self._scale: np.ndarray | None = None

    @property
    def warmup_steps(self) -> int:
        return self.window

    def fit(self, values, usable, context) -> None:
        spread = _rolling(values, self.window, "std")
        rows = np.asarray(usable, dtype=bool)
        sample = spread[rows] if rows.any() else spread
        with np.errstate(invalid="ignore"):
            self._scale = np.nanstd(sample, axis=0)

    def score(self, values, valid, context) -> np.ndarray:
        spread = _rolling(values, self.window, "std")
        scale = self._scale if self._scale is not None else np.nanstd(spread, axis=0)
        return spread / np.maximum(scale, EPSILON)


class AlwaysQuiet(Detector):
    """Never fires. Recall must be 0/n and false alarms 0/n -- the floor."""

    name = "quiet"

    def score(self, values, valid, context) -> np.ndarray:
        return np.zeros(values.shape[0], dtype=np.float64)

    def threshold_from(self, train_scores):
        return 0.5          # above every score it will ever emit


class RandomScore(Detector):
    """Seeded noise. Should land near the base rate, and reproduce exactly."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.seed = int(seed)

    def score(self, values, valid, context) -> np.ndarray:
        rng = np.random.default_rng(self.seed + context.fold)
        return rng.random(values.shape[0])


class Oracle(Detector):
    """Handed the answers. Exists only to prove the metrics can reach 1.0.

    Never a result. If this does not score a perfect recall, the harness is
    broken and every other number it has produced is suspect.
    """

    name = "oracle"

    def __init__(self, anomaly: np.ndarray) -> None:
        super().__init__(cheats=True)
        self._anomaly = np.asarray(anomaly, dtype=bool)

    def score(self, values, valid, context) -> np.ndarray:
        lo, hi = context.window
        return self._anomaly[lo:hi].astype(np.float64)

    def threshold_from(self, train_scores):
        return 0.5
