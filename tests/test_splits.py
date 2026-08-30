"""Chronology, contamination, and the refusal to score a split that is too thin."""
from __future__ import annotations

import pytest

from sentinel_eval import splits
from sentinel_eval.errors import SplitTooThin


def test_training_never_follows_the_data_it_is_scored_on():
    for fold in splits.forward_chaining(1000, seed_fraction=0.25, folds=3).folds:
        assert fold.train[1] <= fold.test[0]


def test_folds_tile_the_test_side_without_overlapping():
    folds = splits.forward_chaining(1000, seed_fraction=0.25, folds=3).folds
    assert folds[0].test[0] == 250 and folds[-1].test[1] == 1000
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test[1] == later.test[0]


def test_each_fold_trains_on_more_history_than_the_last():
    sizes = [f.train_steps for f in splits.forward_chaining(1000).folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_moving_the_boundary_earlier_buys_test_side_events(loaded):
    early = splits.coverage(splits.chronological(len(loaded.grid), fraction=0.25),
                            loaded.truth)[0]
    late = splits.coverage(splits.chronological(len(loaded.grid), fraction=0.60),
                           loaded.truth)[0]
    assert early.test_anomalies > late.test_anomalies


def test_annotated_anomalies_are_removed_from_training(loaded):
    fold = splits.chronological(len(loaded.grid)).folds[0]
    usable = splits.train_mask(fold, loaded.truth)
    assert not (usable & loaded.truth.anomaly).any()
    assert not (usable & loaded.truth.unscorable).any()
    assert usable[fold.train[1]:].sum() == 0        # nothing beyond the boundary


def test_contamination_is_reported_rather_than_hidden(loaded):
    row = splits.coverage(splits.chronological(len(loaded.grid)), loaded.truth)[0]
    assert row.masked_steps == row.train_steps - row.train_usable_steps
    assert row.train_anomaly_steps > 0


def test_a_split_too_thin_to_compare_is_refused(loaded):
    with pytest.raises(SplitTooThin, match="floor"):
        splits.check(splits.chronological(len(loaded.grid), fraction=0.25),
                     loaded.truth, minimum=10_000)


def test_every_event_is_scored_in_exactly_one_fold(loaded):
    rows = splits.coverage(splits.forward_chaining(len(loaded.grid)), loaded.truth)
    single = splits.coverage(splits.chronological(len(loaded.grid), fraction=0.25),
                             loaded.truth)[0]
    assert sum(r.test_anomalies for r in rows) == single.test_anomalies
