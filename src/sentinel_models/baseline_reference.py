"""The Level 1 statistical baseline, stated as the flight component computes it.

This module is to `flight/src/Baseline.cpp` what `reference.py` is to the GRU
core: the NumPy statement of the arithmetic, transcribed into C++ and pinned
against it by golden vectors. It is **not** a detector, has no registry entry,
and nothing in `sentinel_eval` imports it.

WHY IT EXISTS RATHER THAN REUSING `baselines.RollingStd`
--------------------------------------------------------
`baselines._rolling` does not compute the rule it states. It builds prefix sums
with ``np.cumsum`` over a float32 array and promotes to float64 only afterwards
(`baselines.py:44`), then differences two large float32 prefix sums to recover a
small second moment. That is catastrophic cancellation, and the variance floor at
`baselines.py:55` clamps the negative result to zero. Measured on N(1000, 3),
float32, T = 8,000, post-warm-up: this module and ``numpy.nanstd`` agree to
3.5e-10, and `_rolling` disagrees with both by 7.66 on a true sigma of 3.0.

So the flight baseline transcribes **the rule**, not the implementation, and the
divergence is deliberate: see `docs/DECISIONS.md` D37 and `docs/MODELS.md` 20.6.
`baselines.py` is not changed here; that repair is scoped as `docs/MODELS.md` 21.

(!) CORRECTED 2026-09-10. The two paragraphs above describe `baselines._rolling`
as it stood when this module was written, and are kept because they are why this
module exists. **The repair has since landed**: `baselines.py` now promotes to
float64 *before* accumulating (`baselines.py:51`), and `tests/test_rolling_precision.py`
pins it. Re-measured today on N(1000, 3), float32, T = 8,000, W = 120,
post-warm-up: `_rolling` agrees with ``numpy.nanstd`` to **3.5e-07** -- float32
input resolution -- with **zero** spurious exact zeros, against the 7.66 of error
and 3,975 zeros recorded above. **The reason for a separate reference module is
unchanged**: it states the rule the flight core transcribes, independently of any
harness implementation, which is what makes the golden vectors evidence rather
than a copy.

THE RULE
--------
Per channel, over a trailing, right-inclusive window of ``WINDOW`` samples that
includes the current one:

    n      = max(count of finite samples in the window, 1)
    S1     = sum of the finite samples
    S2     = sum of their squares
    spread = sqrt(max(S2 / n - (S1 / n)ptwo, 0))
    score  = spread / max(scale[channel], EPSILON)

then the maximum across channels, and one alarm when that maximum is ``>=`` the
threshold. The reduction and the comparison are `sentinel_eval.detector`'s and
`harness.py:169-172`'s, and both are float64.

Sources, quoted by line so the transcription can be checked: the rule
`src/sentinel_models/baselines.py:34-55` and `:86-110`; ``window = 120``
`baselines.py:91`; ``EPSILON = 1e-12`` `baselines.py:31`; the max across channels
`src/sentinel_eval/detector.py:143-155`; the 99.9th-percentile threshold recipe
`detector.py:127-132`; the ``>=`` comparison `src/sentinel_eval/harness.py:169-172`.

TWO DELIBERATE DEPARTURES, BOTH UNREACHABLE FROM THE HARNESS'S OWN DATA
-----------------------------------------------------------------------
1. **Accumulation is float64**, which is the point of this module.
2. **A non-finite sample is excluded from the sums, not merely from the count.**
   `_rolling` sums ``np.nan_to_num(values)`` without masking, so a NaN
   contributes 0 -- harmless -- but a +/-inf contributes +/-3.4e38 while not
   incrementing the count, which poisons the window. Bundle values are float32
   and NaN where unobserved (`bundle.py:39`), never infinite, so the two agree on
   every input the harness can produce. `test_baseline_reference.py` pins that
   agreement and pins the infinite case as a known, intended difference.
"""
from __future__ import annotations

import numpy as np

#: `baselines.py:91`. The only value any run has ever used.
WINDOW = 120

#: `baselines.py:31`. The floor under the scale divisor.
EPSILON = 1e-12


def spread(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Trailing population standard deviation per channel, float64, exact.

    Computed by recomputing each window rather than by differencing prefix sums,
    which is both what a fixed-memory ring buffer does in flight and what keeps
    the cancellation out.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be (T, C), got shape {values.shape}")
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0).astype(np.float64)
    present = finite.astype(np.float64)

    steps, channels = values.shape
    out = np.empty((steps, channels), dtype=np.float64)
    for t in range(steps):
        lo = max(0, t - window + 1)
        window_filled = filled[lo:t + 1]
        window_present = present[lo:t + 1]
        count = np.maximum(window_present.sum(axis=0), 1.0)
        mean = window_filled.sum(axis=0) / count
        second = (window_filled * window_filled).sum(axis=0) / count
        out[t] = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return out


def score(values: np.ndarray, scale: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Per-channel score: the spread divided by its calibrated per-channel scale.

    `scale` is a flight parameter (`BASELINE_SCALE`, D34), not a fitted quantity
    this module derives -- Level 1 must work when no model file can be read.
    """
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (values.shape[1],):
        raise ValueError(f"scale must be ({values.shape[1]},), got {scale.shape}")
    return spread(values, window) / np.maximum(scale, EPSILON)


def reduce_scores(per_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum across channels, and the channel that took it.

    `detector.py:143-155` takes the max; the argmax is what the warning event
    names. Ties keep the **lowest** index, which is what a strict ``>`` scan does
    and what `Detector.cpp:100-106` already does on the model path.
    """
    per_channel = np.asarray(per_channel, dtype=np.float64)
    return per_channel.max(axis=1), per_channel.argmax(axis=1).astype(np.int64)


class StreamingBaseline:
    """One tick at a time, with the fixed memory the flight component has.

    This is the authoritative statement of the flight arithmetic: the golden
    vectors are generated from it, and `flight/src/Baseline.cpp` is held to it.
    `spread()` above is the vectorised cross-check; the two agree to 1.3e-15 and
    not bit-exactly, because this class sums over the ring in slot order while
    `spread()` sums a chronological slice, and floating-point addition is not
    associative. `test_baseline_reference.py` pins that bound.
    """

    def __init__(self, n_channels: int, scale: np.ndarray, threshold: float,
                 window: int = WINDOW) -> None:
        self.n_channels = int(n_channels)
        self.window = int(window)
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        if self.scale.shape != (self.n_channels,):
            raise ValueError(f"scale must be ({self.n_channels},), got {self.scale.shape}")
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        """Clear the ring and restart the warm-up. What a restart looks like."""
        self._ring = np.zeros((self.window, self.n_channels), dtype=np.float64)
        self._present = np.zeros((self.window, self.n_channels), dtype=bool)
        self.steps = 0
        self.channel_scores = np.zeros(self.n_channels, dtype=np.float64)
        self.score = -np.inf
        self.peak_channel = 0
        self.crossing = False
        self.emitted = False

    @property
    def warmed(self) -> bool:
        """`baselines.py:96-98`: warmup_steps is the window, 120, not 2,350.

        Strictly greater, and that is the harness's convention rather than an
        off-by-one. `harness.py:160-167` prefixes the scored window with
        ``warmup_steps`` samples of history and then drops exactly that many
        outputs, so the first surviving score is the one produced by call
        ``window + 1``. `flight/src/Detector.cpp:113` says the same thing a
        different way: it tests the count of *previous* steps, because its
        increment comes after the check. This class increments first, so the
        equivalent test is ``>`` rather than ``>=``. The two halves of the core
        must not disagree about when they are allowed to speak.
        """
        return self.steps > self.window

    def step(self, values: np.ndarray, valid: bool = True) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.n_channels,):
            raise ValueError(f"values must be ({self.n_channels},), got {values.shape}")

        finite = np.isfinite(values)
        slot = self.steps % self.window
        self._ring[slot] = np.where(finite, values, 0.0).astype(np.float64)
        self._present[slot] = finite
        self.steps += 1

        filled = np.where(self._present, self._ring, 0.0)
        used = self._present.astype(np.float64)
        if self.steps < self.window:            # partial window, as the batch form has
            filled = filled[:self.steps]
            used = used[:self.steps]

        count = np.maximum(used.sum(axis=0), 1.0)
        mean = filled.sum(axis=0) / count
        second = (filled * filled).sum(axis=0) / count
        spread_now = np.sqrt(np.maximum(second - mean * mean, 0.0))
        self.channel_scores = spread_now / np.maximum(self.scale, EPSILON)

        if valid:
            self.peak_channel = int(self.channel_scores.argmax())
            self.score = float(self.channel_scores[self.peak_channel])
        else:
            # Nothing measured is never an alarm (`detectors.py:394-399`), the
            # same rule the model path applies at `Detector.cpp:95-101`.
            self.peak_channel = 0
            self.score = float("-inf")

        self.crossing = bool(self.score >= self.threshold)
        self.emitted = bool(self.crossing and self.warmed)
