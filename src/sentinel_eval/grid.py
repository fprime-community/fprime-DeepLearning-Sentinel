"""The common time grid, and the zero-order hold that puts channels on it.

Every channel in ESA-ADB has its own irregular sampling times, so cross-channel
work needs one shared grid first. ESA prescribes **zero-order hold, never linear
and never Fourier** -- holding the last observed value, because interpolating
telemetry invents readings that the spacecraft never reported.

**The gap guard is the part that matters.** A plain forward fill holds the last
value across a three-day communication outage and produces a perfectly flat line
that no detector can distinguish from a healthy, quiet subsystem. The model then
learns from it and is scored on it. So a grid point is only filled if the sample
behind it is younger than ``max_hold``; beyond that the point is marked invalid
and excluded from every metric, exactly like an annotated gap. Mission3 alone
carries 397 communication gaps, so this is not a corner case.

Timestamps here are anonymised mission time (``t_anon``), never UTC. Positions on
this grid are **timesteps**. No duration derived from it is a real duration, so
nothing downstream may express one in hours or minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import CoverageError
from .read import Series

#: How stale a held sample may be before its grid point is called invalid,
#: as a multiple of the grid period. Ten steps of silence is already a gap.
DEFAULT_MAX_HOLD_STEPS = 10


@dataclass(frozen=True)
class Grid:
    """A regular lattice of anonymised mission time. Positions are timesteps."""

    start: np.datetime64
    period: np.timedelta64
    n: int

    def __len__(self) -> int:
        return self.n

    @classmethod
    def spanning(cls, start, end, period) -> "Grid":
        start = np.datetime64(start, "ns")
        end = np.datetime64(end, "ns")
        period = np.timedelta64(period, "ns")
        if end <= start:
            raise CoverageError(f"empty grid: {start} -> {end}")
        n = int((end - start) // period)
        if n < 1:
            raise CoverageError(f"grid period {period} is longer than the span {start}..{end}")
        return cls(start=start, period=period, n=n)

    @property
    def end(self) -> np.datetime64:
        """Exclusive upper bound: the grid covers [start, end)."""
        return self.start + self.period * self.n

    @property
    def times(self) -> np.ndarray:
        """The full time axis. 8 bytes a step -- build it only when needed."""
        return self.start + np.arange(self.n, dtype="int64") * self.period

    def position_of(self, when) -> int:
        """First timestep at or after ``when``, clamped to the grid."""
        delta = np.datetime64(when, "ns") - self.start
        if delta < np.timedelta64(0, "ns"):
            return 0
        return int(min(-(-int(delta) // int(self.period)), self.n))

    def slice_of(self, start, end) -> tuple[int, int]:
        """Half-open ``[lo, hi)`` of timesteps ``t`` with ``start <= t < end``.

        A zero-length event still claims exactly one timestep. Rasterising it to
        nothing would make it unscorable by construction and quietly depress
        recall -- and ten of the eleven point anomalies in ESA-ADB are shorter
        than one grid period.
        """
        lo = self.position_of(start)
        hi = self.position_of(end)
        if hi <= lo:
            hi = min(lo + 1, self.n)
        return lo, hi

    def subgrid(self, lo: int, hi: int) -> "Grid":
        return Grid(start=self.start + self.period * lo, period=self.period, n=hi - lo)

    def describe(self) -> str:
        return (f"{self.n:,} timesteps of {int(self.period / np.timedelta64(1, 's'))}s "
                f"from {self.start}")


def resample_zoh(series: Series, grid: Grid, *, max_hold: np.timedelta64 | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Put one channel on the grid by zero-order hold, with a staleness guard.

    Returns ``(values, valid)``. ``valid`` is False wherever no sample precedes
    the timestep, or where the sample behind it is older than ``max_hold`` --
    that is, wherever a held value would be fiction rather than measurement.
    """
    if max_hold is None:
        max_hold = grid.period * DEFAULT_MAX_HOLD_STEPS

    times = grid.times
    # index of the last sample at or before each timestep; -1 where none exists
    behind = np.searchsorted(series.t_anon, times, side="right") - 1
    valid = behind >= 0
    safe = np.where(valid, behind, 0)

    age = times - series.t_anon[safe]
    valid &= age <= max_hold

    values = np.asarray(series.value)[safe].astype(np.float32, copy=True)
    values[~valid] = np.nan
    return values, valid


def difference(values: np.ndarray) -> np.ndarray:
    """First difference for monotonic counters, ESA's prescribed treatment.

    The first timestep has no predecessor and becomes NaN rather than a
    fabricated zero, so the caller's validity mask carries the truth.
    """
    out = np.empty_like(values)
    out[0] = np.nan
    out[1:] = np.diff(values)
    return out
