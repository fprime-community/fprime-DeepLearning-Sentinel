"""Event-wise scoring: did we catch it, and how much of what we said was real.

Two families, reported side by side because they answer different questions.

**Event-wise (existence).** An event counts as detected when the prediction fires
anywhere inside it. This is what an operator means by "did we catch it", it is
the form ESA-ADB reports, and it is the only form in which a k/n denominator is
meaningful -- 31 headline-cell events, not 47,000 anomalous timesteps. Precision
is counted over predicted *ranges*: a contiguous alarm is one claim, whether it
lasts four timesteps or four hundred, because that is how an operator receives it.

**Range-based (Tatbul et al., NeurIPS 2018).** The published definition, with
existence reward ``alpha``, a cardinality factor penalising one true event
answered by many fragmented alarms, and a positional bias. Reported alongside so
our numbers are comparable to work built on that definition. Only the ``flat``
bias is implemented: one bias done correctly beats four done approximately, and
flat is the default everywhere it matters.

**Rare events are not positives.** A predicted range overlapping only a rare
nominal event -- a manoeuvre, a reset, a calibration -- is a false alarm, and is
counted as one. That is the whole adoption argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..labels import ANOMALY, HEADLINE_CELL, Event
from .counts import Count, f_beta
from .ranges import Range, mask_to_ranges, merge, overlap, overlaps

BETA = 0.5


@dataclass(frozen=True)
class EventScore:
    """Every headline number, each carrying its own denominator."""

    recall: Count
    precision: Count
    f_beta: float | None
    contextual: Count
    headline_cell: Count
    point: Count
    by_cell: dict[str, Count] = field(default_factory=dict)
    range_recall: float | None = None
    range_precision: float | None = None
    range_f_beta: float | None = None
    predicted_ranges: int = 0

    def as_dict(self) -> dict:
        return {
            "event_recall": self.recall.as_dict(),
            "event_precision": self.precision.as_dict(),
            "event_f0.5": self.f_beta,
            "contextual_recall": self.contextual.as_dict(),
            "headline_cell_recall": self.headline_cell.as_dict(),
            "point_recall": self.point.as_dict(),
            "recall_by_cell": {k: v.as_dict() for k, v in self.by_cell.items()},
            "range_recall": self.range_recall,
            "range_precision": self.range_precision,
            "range_f0.5": self.range_f_beta,
            "predicted_ranges": self.predicted_ranges,
        }


def detected(span: Range, predicted: np.ndarray) -> bool:
    """One event is caught if the prediction fires anywhere inside it."""
    lo, hi = span
    return bool(predicted[lo:hi].any())


def score(
    events: list[Event],
    spans: dict[str, Range],
    predicted: np.ndarray,
    scorable: np.ndarray,
    *,
    beta: float = BETA,
) -> EventScore:
    """Score one prediction against the events visible in this window.

    ``predicted`` is masked to ``scorable`` first: an alarm raised inside a
    communication gap is neither right nor wrong, because there was no
    measurement there to react to.
    """
    fired = np.asarray(predicted, dtype=bool) & np.asarray(scorable, dtype=bool)
    anomalies = [e for e in events if e.category == ANOMALY and e.event_id in spans]
    truth_ranges = merge([spans[e.event_id] for e in anomalies])
    pred_ranges = mask_to_ranges(fired)

    caught = [e for e in anomalies if detected(spans[e.event_id], fired)]
    hit_ids = {e.event_id for e in caught}

    real_alarms = sum(1 for r in pred_ranges if any(overlaps(r, t) for t in truth_ranges))

    by_cell: dict[str, Count] = {}
    for cell in sorted({e.cell for e in anomalies}):
        members = [e for e in anomalies if e.cell == cell]
        by_cell[cell] = Count(sum(1 for e in members if e.event_id in hit_ids), len(members))

    def subset(predicate) -> Count:
        members = [e for e in anomalies if predicate(e)]
        return Count(sum(1 for e in members if e.event_id in hit_ids), len(members))

    recall = Count(len(caught), len(anomalies))
    precision = Count(real_alarms, len(pred_ranges))

    return EventScore(
        recall=recall,
        precision=precision,
        f_beta=f_beta(precision, recall, beta),
        contextual=subset(lambda e: e.contextual),
        headline_cell=subset(lambda e: e.cell == HEADLINE_CELL),
        point=subset(lambda e: e.is_point),
        by_cell=by_cell,
        range_recall=range_recall(truth_ranges, pred_ranges),
        range_precision=range_precision(truth_ranges, pred_ranges),
        range_f_beta=_range_f(truth_ranges, pred_ranges, beta),
        predicted_ranges=len(pred_ranges),
    )


# --------------------------------------------------------------------------
# Tatbul et al. (NeurIPS 2018), flat positional bias
# --------------------------------------------------------------------------
def _omega(base: Range, other: Range) -> float:
    """Flat-bias overlap: the fraction of ``base`` that ``other`` covers."""
    width = base[1] - base[0]
    return 0.0 if width <= 0 else overlap(base, other) / width


def _cardinality(base: Range, others: list[Range], mode: str) -> float:
    """Penalise one true event answered by many fragmented alarms."""
    hits = sum(1 for o in others if overlaps(base, o))
    if hits <= 1:
        return 1.0
    if mode == "reciprocal":
        return 1.0 / hits
    if mode == "one":
        return 1.0
    raise ValueError(f"unknown cardinality mode {mode!r}")


def _side(bases: list[Range], others: list[Range], *, alpha: float, cardinality: str,
          bias: str) -> float | None:
    if bias != "flat":
        raise ValueError(
            f"positional bias {bias!r} is not implemented; only 'flat' is, deliberately"
        )
    if not bases:
        return None
    total = 0.0
    for base in bases:
        existence = 1.0 if any(overlaps(base, o) for o in others) else 0.0
        covered = _cardinality(base, others, cardinality) * sum(
            _omega(base, o) for o in others if overlaps(base, o)
        )
        total += alpha * existence + (1.0 - alpha) * covered
    return total / len(bases)


def range_recall(truth: list[Range], predicted: list[Range], *, alpha: float = 0.0,
                 cardinality: str = "reciprocal", bias: str = "flat") -> float | None:
    """How much of each true event the prediction covered."""
    return _side(truth, predicted, alpha=alpha, cardinality=cardinality, bias=bias)


def range_precision(truth: list[Range], predicted: list[Range], *,
                    cardinality: str = "reciprocal", bias: str = "flat") -> float | None:
    """How much of each alarm was real. No existence term, by definition."""
    return _side(predicted, truth, alpha=0.0, cardinality=cardinality, bias=bias)


def _range_f(truth: list[Range], predicted: list[Range], beta: float) -> float | None:
    p = range_precision(truth, predicted)
    r = range_recall(truth, predicted)
    if p is None or r is None:
        return None
    b2 = beta * beta
    denominator = b2 * p + r
    return None if denominator == 0 else (1 + b2) * p * r / denominator
