"""Categories, taxonomy, and what must never be scored as nominal."""
from __future__ import annotations

from sentinel_eval.labels import (ANOMALY, COMMUNICATION_GAP, HEADLINE_CELL,
                                  INVALID_SEGMENT, RARE_EVENT, UNSCORABLE)


def test_gaps_and_invalid_segments_are_neither_positive_nor_negative(loaded):
    truth = loaded.truth
    assert truth.unscorable.any()
    assert not (truth.unscorable & truth.anomaly).any()
    assert truth.scorable.sum() + truth.unscorable.sum() == truth.anomaly.shape[0]


def test_rare_events_are_nominal_not_anomalous(loaded):
    truth = loaded.truth
    assert truth.rare_event.any()
    assert not (truth.rare_event & truth.anomaly).any()
    for event in truth.of(RARE_EVENT):
        assert event.category not in (ANOMALY,) + UNSCORABLE


def test_every_annotated_anomaly_lands_on_the_grid(loaded):
    for event in loaded.truth.anomalies:
        lo, hi = loaded.truth.spans[event.event_id]
        assert hi > lo


def test_contextual_means_multivariate(loaded):
    for event in loaded.truth.anomalies:
        assert event.contextual == (event.dimensionality == "Multivariate")


def test_the_headline_cell_is_present_and_named_consistently(loaded, labels):
    census = labels.census(list(loaded.truth.events))
    assert census["headline_cell"] == census["by_cell"].get(HEADLINE_CELL, 0)
    assert census["headline_cell"] > 0


def test_the_census_counts_events_not_timesteps(loaded, labels):
    census = labels.census(list(loaded.truth.events))
    assert census["anomalies"] == len(loaded.truth.anomalies)
    assert census["anomalies"] < loaded.truth.anomaly.sum()
    assert census["contextual"] <= census["anomalies"]
    assert census["point"] <= census["anomalies"]


def test_all_four_categories_are_present_in_the_fixture(loaded):
    categories = {e.category for e in loaded.truth.events}
    assert categories == {ANOMALY, RARE_EVENT, COMMUNICATION_GAP, INVALID_SEGMENT}


def test_per_channel_attribution_matches_the_selection(loaded):
    assert loaded.truth.per_channel.shape == (len(loaded.grid), len(loaded.channels))
    assert loaded.truth.per_channel.any()
