"""VUS-PR: threshold-free, and tolerant of being right a little early or late.

Objective.md 9.5 adopts VUS-PR alongside event-wise F0.5, and the two answer
different complaints. Event-wise scores depend on a threshold, which a detector
may have chosen well or badly. VUS integrates over thresholds *and* over a buffer
width, so a detection that fires ten timesteps before an event begins -- which is
the whole point of an early-warning system -- is not punished as a false alarm
followed by a miss.

**What is computed here, stated precisely.** For each buffer width ``l`` the true
anomaly ranges are widened by ``l`` timesteps on each side, with the label
decaying linearly from 1 inside the event to 0 at the edge of the buffer.
Average precision is then computed against those soft labels, and the results are
averaged over the buffer widths -- the volume under the precision-recall surface.

Exact numerical agreement with the reference VUS implementation is **not**
claimed. It could not be checked: TimeEval and its dependencies do not install on
this project's Python. That is consistent with the Phase 1 gate, which compares
architectures on this harness and puts external comparability out of scope. What
is guaranteed is stated in the tests -- a perfect detector scores 1.0, an
inverted one scores near the base rate, and widening the buffer never lowers the
score for a detection that is merely early.

No sklearn: average precision is a sort and a cumulative sum.
"""
from __future__ import annotations

import numpy as np

from .ranges import Range

#: The narrowest sweep that still integrates a volume. Below three distinct
#: widths, VUS is AUC-PR wearing a different name.
MIN_BUFFER_STEPS = 8


def soft_labels(truth: list[Range], n: int, buffer: int) -> np.ndarray:
    """1.0 inside each event, decaying linearly to 0 across ``buffer`` steps."""
    label = np.zeros(n, dtype=np.float64)
    for lo, hi in truth:
        lo, hi = max(0, lo), min(n, hi)
        if hi <= lo:
            continue
        label[lo:hi] = 1.0
        if buffer <= 0:
            continue
        ramp = np.linspace(1.0, 0.0, buffer + 2)[1:-1]      # excludes 1.0 and 0.0
        before_lo = max(0, lo - buffer)
        if lo > before_lo:
            head = ramp[::-1][-(lo - before_lo):]
            np.maximum.at(label, np.arange(before_lo, lo), head)
        after_hi = min(n, hi + buffer)
        if after_hi > hi:
            np.maximum.at(label, np.arange(hi, after_hi), ramp[: after_hi - hi])
    return label


def tie_groups(sorted_scores: np.ndarray) -> np.ndarray:
    """Index of the last element of each run of equal scores.

    Ties must be resolved as one operating point, not walked through one index
    at a time. Without this, a detector emitting a *constant* score is ranked by
    array order, and inherits whatever accidental merit that order has: on one
    synthetic fold, a detector that never fires scored 0.45 against 0.26 for one
    that actually discriminated, purely because the anomalies sat early in the
    window. A constant score must earn the base rate and nothing more.
    """
    if sorted_scores.size == 0:
        return np.empty(0, dtype=np.intp)
    # Compared directly rather than via np.diff: masked timesteps carry -inf,
    # and -inf minus -inf is NaN, which would make every one its own group.
    changed = np.concatenate([sorted_scores[:-1] != sorted_scores[1:], [True]])
    return np.flatnonzero(changed)


def average_precision(scores: np.ndarray, label: np.ndarray, *,
                      total: float | None = None, order=None, last=None) -> float | None:
    """Area under the precision-recall curve for soft labels in [0, 1].

    ``total`` is the recall denominator, and it is deliberately **not**
    ``label.sum()`` when a buffer is in play. The buffer exists to *forgive* a
    detection that is a few timesteps early, not to *demand* that the detector
    also cover the forgiving region: scored the other way, a detector matching
    the ground truth exactly cannot reach 1.0 and the top of the scale becomes
    unattainable. Passing the unbuffered event mass makes buffer coverage
    partial credit that substitutes for hard coverage, and recall is clipped at
    1.0 so covering both cannot exceed it.

    ``order`` lets the caller sort once and reuse it across buffer widths; the
    sort dominates the cost and only the labels change.
    """
    total = float(label.sum()) if total is None else float(total)
    if total <= 0:
        return None
    if order is None:
        order = np.argsort(-scores, kind="stable")
    if last is None:
        last = tie_groups(scores[order])

    tp = np.cumsum(label[order])[last]
    precision = tp / (last + 1)
    recall = np.minimum(tp / total, 1.0)
    gain = np.diff(recall, prepend=0.0)
    return float(np.sum(gain * precision))


def default_buffers(truth: list[Range], n: int, count: int = 5) -> tuple[int, ...]:
    """Buffer widths scaled to how long the events actually are.

    Derived from the **75th percentile**, not the median. Event footprints are
    extremely skewed: on `m1-ss5` the median is 1 timestep while the mean is
    2,909 and the maximum 81,717, because 18 of 38 headline-cell events register
    as sub-grid-cell spikes on those particular channels. A median-derived sweep
    collapsed to ``[0, 1]`` there, so VUS quietly stopped integrating any volume
    and became plain AUC-PR -- on one channel set and not the other, which is
    precisely the kind of silent substitution that surfaces at a gate as an
    inexplicable result.

    :data:`MIN_BUFFER_STEPS` floors the sweep so a degenerate distribution cannot
    collapse it again, and the cap keeps the widest buffer to 1% of the window,
    since a buffer approaching the window length would forgive everything.
    """
    widths = [hi - lo for lo, hi in truth if hi > lo]
    if not widths:
        return (0,)
    typical = int(np.percentile(widths, 75))
    ceiling = max(MIN_BUFFER_STEPS, n // 100)
    top = max(1, min(max(typical, MIN_BUFFER_STEPS), ceiling))
    return tuple(sorted({int(round(top * i / (count - 1))) for i in range(count)}))


def vus_pr(scores: np.ndarray, truth: list[Range], n: int, *,
           buffers: tuple[int, ...] | None = None) -> tuple[float | None, dict]:
    """Volume under the PR surface, plus the per-buffer curve it averaged."""
    if buffers is None:
        buffers = default_buffers(truth, n)

    clean = np.asarray(scores, dtype=np.float64).copy()
    clean[~np.isfinite(clean)] = -np.inf        # a NaN score never alarms
    order = np.argsort(-clean, kind="stable")   # sorted once, reused per buffer
    last = tie_groups(clean[order])             # tie structure is buffer-independent

    hard_total = float(soft_labels(truth, n, 0).sum())   # the recall denominator
    per_buffer: dict[int, float] = {}
    for buffer in buffers:
        value = average_precision(clean, soft_labels(truth, n, buffer),
                                  total=hard_total, order=order, last=last)
        if value is not None:
            per_buffer[int(buffer)] = value

    if not per_buffer:
        return None, {"buffers": list(buffers), "per_buffer": {}}
    volume = float(np.mean(list(per_buffer.values())))
    return volume, {"buffers": list(buffers), "per_buffer": per_buffer}
