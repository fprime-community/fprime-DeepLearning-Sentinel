"""Event-wise and Tatbul range-based scoring, against hand-computed values."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.labels import ANOMALY, Event, Segment
from sentinel_eval.metrics.eventwise import (range_precision, range_recall, score)


def _event(event_id, cell=("Multivariate", "Global", "Subsequence"), category=ANOMALY):
    return Event(mission="m", event_id=event_id, category=category,
                 start=np.datetime64("2000-01-01"), end=np.datetime64("2000-01-02"),
                 dimensionality=cell[0], locality=cell[1], length=cell[2],
                 esa_class="c", subclass="s",
                 segments=(Segment("channel_1", np.datetime64("2000-01-01"),
                                   np.datetime64("2000-01-02")),))


def test_flat_bias_overlap_fraction():
    # one true event of 10 steps, one alarm covering 4 of them
    assert range_recall([(10, 20)], [(12, 16)]) == pytest.approx(0.4)
    assert range_precision([(10, 20)], [(12, 16)]) == pytest.approx(1.0)


def test_fragmented_alarms_are_penalised():
    # same 8 steps covered, but split in two: 0.8 * reciprocal cardinality 1/2
    assert range_recall([(10, 20)], [(10, 14), (16, 20)]) == pytest.approx(0.4)
    assert range_recall([(10, 20)], [(10, 18)]) == pytest.approx(0.8)


def test_perfect_and_missed():
    assert range_recall([(0, 10)], [(0, 10)]) == pytest.approx(1.0)
    assert range_recall([(0, 10)], [(50, 60)]) == pytest.approx(0.0)
    assert range_precision([(0, 10)], []) is None


def test_existence_reward():
    assert range_recall([(0, 100)], [(0, 1)], alpha=1.0) == pytest.approx(1.0)


def test_unimplemented_bias_is_refused():
    with pytest.raises(ValueError, match="deliberately"):
        range_recall([(0, 10)], [(0, 5)], bias="front")


def test_event_recall_is_existence_based():
    events = [_event("a"), _event("b")]
    spans = {"a": (0, 10), "b": (50, 60)}
    predicted = np.zeros(100, dtype=bool)
    predicted[5] = True                       # one timestep inside 'a' is enough
    result = score(events, spans, predicted, np.ones(100, dtype=bool))
    assert (result.recall.k, result.recall.n) == (1, 2)


def test_alarms_inside_gaps_are_neither_right_nor_wrong():
    events = [_event("a")]
    spans = {"a": (0, 10)}
    predicted = np.zeros(100, dtype=bool)
    predicted[80:90] = True
    scorable = np.ones(100, dtype=bool)
    scorable[70:100] = False                  # that whole alarm sits in a gap
    result = score(events, spans, predicted, scorable)
    assert result.predicted_ranges == 0
    assert result.precision.undefined


def test_recall_is_broken_out_by_taxonomy_cell():
    events = [_event("a"), _event("b", ("Univariate", "Local", "Point"))]
    spans = {"a": (0, 10), "b": (50, 51)}
    predicted = np.zeros(100, dtype=bool)
    predicted[0:10] = True
    result = score(events, spans, predicted, np.ones(100, dtype=bool))
    assert result.headline_cell.k == 1 and result.headline_cell.n == 1
    assert result.point.k == 0 and result.point.n == 1
    assert result.contextual.n == 1           # only the multivariate one counts
