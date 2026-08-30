"""Several detectors, one bundle load. The reason a gate run stays affordable.

A cold run costs the same operations whether it scores one detector or five,
because `bundle.load` is called once and every detector and every fold reads the
resident arrays. Scoring LSTM, GRU and TCN separately would triple the bill and
approach the thousand-operation tripwire over a month of work.
"""
from __future__ import annotations

from sentinel_eval import harness, tasks
from sentinel_models import baselines


class CountingSource:
    """Wraps the fixture and counts every object handed over."""

    def __init__(self, bucket):
        self.bucket, self.gets = bucket, 0

    def get(self, key):
        self.gets += 1
        return self.bucket.get(key)


def test_scoring_more_detectors_does_not_cost_more_reads(bucket, catalog, labels,
                                                         channel_ids, split):
    from sentinel_eval import bundle as bundle_mod

    task = tasks.get("synthetic")
    counting = CountingSource(bucket)
    loaded = bundle_mod.load(counting, catalog, labels, mission="missionX",
                             channel_ids=channel_ids)
    after_load = counting.gets

    harness.evaluate(loaded, split,
                     [baselines.MovingAverage(60), baselines.RollingStd(60),
                      baselines.AlwaysQuiet(), baselines.RandomScore(3)],
                     task, sweep=False)
    assert counting.gets == after_load          # scoring reads nothing further


def test_exactly_one_read_per_manifest_object(bucket, catalog, labels, channel_ids):
    """Sharded channels cost one read per shard, unsharded one. Nothing re-read."""
    from sentinel_eval import bundle as bundle_mod

    counting = CountingSource(bucket)
    bundle_mod.load(counting, catalog, labels, mission="missionX",
                    channel_ids=channel_ids)
    expected = sum(len(catalog.channel("missionX", c).objects) for c in channel_ids)
    assert counting.gets == expected


def test_every_detector_gets_the_same_folds(loaded, split):
    task = tasks.get("synthetic")
    record = harness.evaluate(loaded, split,
                              [baselines.MovingAverage(60), baselines.AlwaysQuiet()],
                              task, sweep=False)
    windows = [[f.window for f in card.folds] for card in record.scorecards]
    assert windows[0] == windows[1]


def test_denominators_are_identical_across_detectors(loaded, split):
    task = tasks.get("synthetic")
    record = harness.evaluate(loaded, split,
                              [baselines.MovingAverage(60), baselines.RollingStd(60)],
                              task, sweep=False)
    a, b = (c.pooled["contextual_recall"] for c in record.scorecards)
    assert a.n == b.n                            # the referee treats them identically
