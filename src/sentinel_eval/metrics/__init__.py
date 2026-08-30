"""Metrics for the Sentinel harness -- every count reported as k/n.

A recall of 0.36 over eleven events can take twelve values, and the gap between
two detectors can be a single event. So nothing in this package returns a bare
float: results are :class:`~sentinel_eval.metrics.counts.Count` objects carrying
their own denominator and the resolution ``1/n`` it implies, and any denominator
below :data:`~sentinel_eval.metrics.counts.UNDERPOWERED_BELOW` is stamped
UNDERPOWERED wherever it is rendered.

Point-adjusted F1 lives in :mod:`.diagnostics`, quarantined and opt-in. It is
known to inflate results and never appears in a headline table.
"""
from __future__ import annotations

from .counts import Count, UNDERPOWERED_BELOW, f_beta
from .ranges import dilate, mask_to_ranges, merge, overlap, overlaps, ranges_to_mask

__all__ = [
    "Count", "UNDERPOWERED_BELOW", "f_beta",
    "mask_to_ranges", "ranges_to_mask", "merge", "overlap", "overlaps", "dilate",
]
