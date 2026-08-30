"""Quarantine. Numbers computed only to show how misleading they are.

Objective.md 9.5: *avoid point-adjusted F1 -- known to inflate results.* Under
point adjustment, detecting a single timestep anywhere inside a true anomaly
promotes the **entire** segment to detected. On data whose anomalies run for
days, one lucky sample buys thousands of true positives, and near-perfect scores
fall out of detectors that found almost nothing. It is the metric behind a good
deal of the anomaly-detection literature that Wu & Keogh (IEEE TKDE 2023) took
apart, and the same paper's finding is why SMAP/MSL was demoted here.

It is implemented anyway, for one reason: quantifying the inflation on our own
results is a stronger argument than asserting it. Nothing in this module may
appear in a headline table or in `docs/RESULTS.md`; the CLI computes it only
behind ``--diagnostics`` and renders it under a banner saying what it is.
"""
from __future__ import annotations

import numpy as np

from .counts import Count
from .ranges import Range

BANNER = (
    "  DIAGNOSTIC -- NOT A HEADLINE NUMBER\n"
    "  Point adjustment promotes a whole anomaly to 'detected' on one lucky sample.\n"
    "  Shown only to quantify the inflation against the event-wise figures above.\n"
    "  Never report this. See Wu & Keogh, IEEE TKDE 2023."
)


def point_wise(predicted: np.ndarray, anomaly: np.ndarray, scorable: np.ndarray) -> dict:
    """Plain per-timestep precision and recall. Honest, but not event-shaped."""
    fired = np.asarray(predicted, dtype=bool) & scorable
    truth = np.asarray(anomaly, dtype=bool) & scorable
    tp = int((fired & truth).sum())
    return {
        "precision": Count(tp, int(fired.sum())).as_dict(),
        "recall": Count(tp, int(truth.sum())).as_dict(),
    }


def point_adjusted(predicted: np.ndarray, anomaly: np.ndarray, scorable: np.ndarray,
                   truth_ranges: list[Range]) -> dict:
    """Point-adjusted precision/recall/F1. Inflated by construction."""
    fired = np.asarray(predicted, dtype=bool) & scorable
    adjusted = fired.copy()
    for lo, hi in truth_ranges:
        if fired[lo:hi].any():
            adjusted[lo:hi] = True        # one sample promotes the whole segment

    truth = np.asarray(anomaly, dtype=bool) & scorable
    tp = int((adjusted & truth).sum())
    precision = Count(tp, int(adjusted.sum()))
    recall = Count(tp, int(truth.sum()))
    p, r = precision.rate, recall.rate
    f1 = None if not p or not r else 2 * p * r / (p + r)
    return {
        "warning": "point-adjusted: inflated, never report",
        "precision": precision.as_dict(),
        "recall": recall.as_dict(),
        "f1": f1,
    }
