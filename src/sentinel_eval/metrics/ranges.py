"""Interval algebra over a timestep grid. Half-open ``[lo, hi)`` throughout.

Everything above this module -- event-wise scoring, false alarms, VUS buffers --
reduces to overlaps between contiguous runs, so the conversions live here once
and are tested against hand-computed values rather than reimplemented per metric.
"""
from __future__ import annotations

import numpy as np

Range = tuple[int, int]


def mask_to_ranges(mask: np.ndarray) -> list[Range]:
    """Contiguous ``True`` runs as half-open ranges."""
    if mask.size == 0:
        return []
    flat = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def ranges_to_mask(ranges: list[Range], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for lo, hi in ranges:
        mask[max(0, lo):max(0, hi)] = True
    return mask


def merge(ranges: list[Range]) -> list[Range]:
    """Union of possibly overlapping ranges, in order."""
    out: list[Range] = []
    for lo, hi in sorted(ranges):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def overlap(a: Range, b: Range) -> int:
    """Number of timesteps in common."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlaps(a: Range, b: Range) -> bool:
    return overlap(a, b) > 0


def dilate(ranges: list[Range], buffer: int, n: int) -> list[Range]:
    """Widen each range by ``buffer`` timesteps on both sides, clipped to ``n``.

    VUS integrates over a range of buffer widths, which is what makes it
    insensitive to a detection that is right but a few timesteps early or late.
    """
    return merge([(max(0, lo - buffer), min(n, hi + buffer)) for lo, hi in ranges])
