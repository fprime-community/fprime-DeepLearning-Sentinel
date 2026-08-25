"""Lead time, against values computed by hand.

The metric that answers *how early*, which nothing in Phase 1 had measured
despite early warning being the project's headline claim (Objective.md 4). It is
also the discriminator the architecture gate may end up needing: 46 anomalies
cannot separate an LSTM from a GRU on recall, because one event moves the rate by
0.022.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.labels import Event, Segment
from sentinel_eval.metrics import leadtime


def _event(event_id: str, cell="Multivariate/Global/Subsequence", category="Anomaly"):
    dimensionality, locality, length = cell.split("/")
    when = np.datetime64("2000-01-01", "ns")
    return Event(mission="missionX", event_id=event_id, category=category,
                 start=when, end=when, dimensionality=dimensionality,
                 locality=locality, length=length, esa_class="", subclass="",
                 segments=(Segment("channel_1", when, when),))


def _mask(n: int, *ranges) -> np.ndarray:
    fired = np.zeros(n, dtype=bool)
    for lo, hi in ranges:
        fired[lo:hi] = True
    return fired


SCORABLE = np.ones(1000, dtype=bool)


# -- the definition, by hand ------------------------------------------------
def test_an_alarm_raised_before_the_event_has_positive_lead():
    """Alarm opens at 100, event starts at 180 -> 80 timesteps of warning."""
    result = leadtime.score([_event("A")], {"A": (180, 300)},
                            _mask(1000, (100, 250)), SCORABLE)
    assert result.per_event["A"] == (80, 150)          # lead 80, alarm 150 wide


def test_an_alarm_raised_inside_the_event_has_negative_lead():
    """Detected late is a real answer and must not be clipped to zero."""
    result = leadtime.score([_event("A")], {"A": (100, 400)},
                            _mask(1000, (250, 300)), SCORABLE)
    assert result.per_event["A"][0] == -150
    assert result.summary()["negative"] == 1


def test_an_alarm_exactly_on_the_boundary_is_zero_lead():
    result = leadtime.score([_event("A")], {"A": (200, 300)},
                            _mask(1000, (200, 260)), SCORABLE)
    assert result.per_event["A"][0] == 0


def test_the_earliest_overlapping_alarm_is_the_one_credited():
    result = leadtime.score([_event("A")], {"A": (300, 400)},
                            _mask(1000, (280, 310), (350, 360)), SCORABLE)
    assert result.per_event["A"] == (20, 30)           # the 280 alarm, not the 350


def test_an_alarm_that_does_not_overlap_credits_nothing():
    """Near-misses are not detections; recall already says the event was missed."""
    result = leadtime.score([_event("A")], {"A": (500, 600)},
                            _mask(1000, (100, 200)), SCORABLE)
    assert result.per_event == {}


def test_a_missed_event_has_no_lead_time_rather_than_a_zero():
    """A detector that catches nothing must not report an excellent median."""
    result = leadtime.score([_event("A")], {"A": (500, 600)},
                            _mask(1000), SCORABLE)
    assert result.summary()["n"] == 0
    assert result.summary()["median"] is None


def test_alarms_inside_a_gap_do_not_count():
    """Unscorable means there was no measurement to react to."""
    scorable = SCORABLE.copy()
    scorable[100:250] = False
    result = leadtime.score([_event("A")], {"A": (180, 300)},
                            _mask(1000, (100, 250)), scorable)
    assert result.per_event == {}


def test_only_anomalies_are_measured_not_rare_events():
    events = [_event("A"), _event("R", category="Rare Event")]
    spans = {"A": (200, 300), "R": (400, 500)}
    result = leadtime.score(events, spans, _mask(1000, (150, 450)), SCORABLE)
    assert set(result.per_event) == {"A"}


# -- summary and pooling ----------------------------------------------------
def test_quantiles_are_computed_over_caught_events_only():
    events = [_event(name) for name in "ABC"]
    spans = {"A": (200, 250), "B": (400, 450), "C": (800, 850)}
    fired = _mask(1000, (100, 210), (390, 410))        # C is missed entirely
    summary = leadtime.score(events, spans, fired, SCORABLE).summary()
    assert summary["n"] == 2
    assert summary["median"] == pytest.approx((100 + 10) / 2)


def test_leads_are_split_by_taxonomy_cell():
    events = [_event("A", "Multivariate/Global/Subsequence"),
              _event("B", "Univariate/Local/Point")]
    spans = {"A": (300, 400), "B": (600, 610)}
    result = leadtime.score(events, spans, _mask(1000, (200, 350), (500, 605)), SCORABLE)
    assert result.by_cell["Multivariate/Global/Subsequence"] == [100]
    assert result.by_cell["Univariate/Local/Point"] == [100]


def test_pooling_folds_is_a_union_because_each_event_is_scored_once():
    a = leadtime.score([_event("A")], {"A": (200, 300)}, _mask(1000, (150, 250)), SCORABLE)
    b = leadtime.score([_event("B")], {"B": (600, 700)}, _mask(1000, (500, 650)), SCORABLE)
    pooled = leadtime.pool([a, b])
    assert sorted(pooled.per_event) == ["A", "B"]
    assert pooled.summary()["n"] == 2


# -- the honesty conditions -------------------------------------------------
def test_the_credited_alarm_width_is_reported_beside_the_lead():
    """A wide alarm inflates lead; the reader must be able to see that it did."""
    result = leadtime.score([_event("A")], {"A": (900, 950)},
                            _mask(1000, (10, 990)), SCORABLE)
    assert result.per_event["A"] == (890, 980)
    assert result.as_dict()["credited_alarm_width"]["median"] == 980


def test_the_units_travel_with_the_numbers():
    """ESA-ADB time is anonymised and scaled; hours would be a lie."""
    payload = leadtime.score([_event("A")], {"A": (200, 300)},
                             _mask(1000, (100, 250)), SCORABLE).as_dict()
    assert payload["units"] == "timesteps"
    assert "hour" not in payload["definition"].lower()


def test_it_is_absent_from_a_scorecard_rather_than_null(loaded, split):
    """Additive: a fold that computed no lead time is byte-identical to before."""
    from sentinel_eval.scorecard import FoldResult
    from sentinel_eval.metrics.falsealarm import FalseAlarmScore
    from sentinel_eval.metrics.counts import Count

    fold = FoldResult(fold=0, window=(0, 10), threshold=1.0, events=None,
                      false_alarms=FalseAlarmScore(Count(0, 0), Count(0, 0), None, 0),
                      vus_pr=None, vus_detail={}, oracle_f_beta=None,
                      oracle_threshold=None)
    assert "lead_time" not in fold.as_dict()
