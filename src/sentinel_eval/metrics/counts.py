"""k/n, never a bare rate.

Every recall, precision and false-alarm figure the harness produces is a count
over a denominator. Printing only the rate loses the one fact a reader needs to
judge it: over eleven point anomalies, recall takes twelve possible values, so
"0.64 against 0.55" is a difference of one event and not evidence of anything.

So :class:`Count` is the return type, it renders its own denominator and the
resolution that denominator implies, and it says so out loud when the denominator
is too small to separate detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Below this many events, a difference between detectors is not measurable.
UNDERPOWERED_BELOW = 20


@dataclass(frozen=True)
class Count:
    """``k`` of ``n``. The rate is derived, never stored alone."""

    k: int
    n: int

    def __post_init__(self) -> None:
        if self.k < 0 or self.n < 0 or self.k > self.n:
            raise ValueError(f"nonsensical count {self.k}/{self.n}")

    @property
    def rate(self) -> float | None:
        """``None`` when there is nothing to measure -- never a silent 0.0."""
        return None if self.n == 0 else self.k / self.n

    @property
    def resolution(self) -> float | None:
        """The smallest difference this denominator can express: ``1/n``."""
        return None if self.n == 0 else 1.0 / self.n

    @property
    def undefined(self) -> bool:
        return self.n == 0

    @property
    def underpowered(self) -> bool:
        return 0 < self.n < UNDERPOWERED_BELOW

    def render(self, width: int = 7) -> str:
        if self.undefined:
            return f"{'-/0':>{width}}  (undefined -- nothing of this kind in the window)"
        text = f"{f'{self.k}/{self.n}':>{width}}  ({self.rate:.3f})   resolution {self.resolution:.3f}"
        if self.underpowered:
            text += f"   [n<{UNDERPOWERED_BELOW}: UNDERPOWERED]"
        return text

    def brief(self) -> str:
        """Compact ``k/n (rate)`` for use inside a sentence."""
        return "-/0" if self.undefined else f"{self.k}/{self.n} ({self.rate:.3f})"

    def as_dict(self) -> dict:
        return {"k": self.k, "n": self.n, "rate": self.rate,
                "resolution": self.resolution, "underpowered": self.underpowered}

    def __str__(self) -> str:
        return self.render()

    def __add__(self, other: "Count") -> "Count":
        """Pool disjoint sets -- e.g. folds, where each event is scored once."""
        return Count(self.k + other.k, self.n + other.n)


def f_beta(precision: Count, recall: Count, beta: float = 0.5) -> float | None:
    """F-beta from two counts. ``beta=0.5`` weights precision twice recall.

    Objective.md 9.5 adopts event-wise F0.5: on a real spacecraft a false alarm
    costs more than a late detection, because an ignored detector is worse than
    no detector at all.
    """
    p, r = precision.rate, recall.rate
    if p is None or r is None:
        return None
    b2 = beta * beta
    denominator = b2 * p + r
    return None if denominator == 0 else (1 + b2) * p * r / denominator
