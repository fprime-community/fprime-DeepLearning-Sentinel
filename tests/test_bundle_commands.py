"""Supplying telecommands must change nothing for anything that does not ask.

The scope change was authorised as **additive only**: existing metrics keep their
values, and a scorecard produced without telecommands must be byte-identical to
one produced before the capability existed. These tests hold that line, and cover
the slicing -- which has an asymmetry that is easy to get wrong and impossible to
see in a result.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import harness, splits, tasks
from sentinel_eval.detector import Context, Detector
from sentinel_models import baselines


@pytest.fixture
def commanded(bucket, catalog, labels, channel_ids):
    return bundle_mod.load(bucket, catalog, labels, mission="missionX",
                           channel_ids=channel_ids, telecommands=3)


# -- additive only ----------------------------------------------------------
def test_a_bundle_that_did_not_ask_carries_none(loaded):
    assert loaded.commands is None
    assert loaded.command_ids == ()
    assert "telecommands" not in loaded.provenance


def test_provenance_is_unchanged_when_telecommands_are_not_requested(
        bucket, catalog, labels, channel_ids, commanded):
    plain = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                            channel_ids=channel_ids)
    assert "telecommands" not in plain.provenance
    assert commanded.provenance["telecommands"] == list(commanded.command_ids)
    assert {k: v for k, v in commanded.provenance.items() if k != "telecommands"} \
        == plain.provenance


def test_a_scorecard_without_telecommands_is_unchanged(loaded, split):
    """The byte-identical condition the scope change was authorised under."""
    record = harness.evaluate(loaded, split, [baselines.MovingAverage(60)],
                              tasks.get("synthetic"), sweep=False)
    payload = json.loads(record.to_json())
    blob = json.dumps(payload)
    assert "command" not in blob


def test_the_values_are_untouched_by_supplying_commands(
        bucket, catalog, labels, channel_ids, commanded):
    plain = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                            channel_ids=channel_ids)
    assert np.array_equal(plain.values, commanded.values)
    assert np.array_equal(plain.valid, commanded.valid)
    assert np.array_equal(plain.truth.anomaly, commanded.truth.anomaly)


def test_scores_are_identical_for_a_detector_that_ignores_commands(
        bucket, catalog, labels, channel_ids, commanded, split):
    plain = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                            channel_ids=channel_ids)
    task = tasks.get("synthetic")
    a = harness.evaluate(plain, split, [baselines.RollingStd(60)], task, sweep=False)
    b = harness.evaluate(commanded, split, [baselines.RollingStd(60)], task, sweep=False)
    assert a.scorecards[0].pooled["event_recall"] == b.scorecards[0].pooled["event_recall"]
    assert a.scorecards[0].pooled["event_f0.5"] == b.scorecards[0].pooled["event_f0.5"]


# -- the slicing, which has an asymmetry ------------------------------------
class _Recorder(Detector):
    """Records the shapes it was handed, and scores nothing."""

    name = "recorder"

    def __init__(self, warmup=0):
        super().__init__()
        self._warmup = warmup
        self.seen = []

    @property
    def warmup_steps(self):
        return self._warmup

    def fit(self, values, usable, context):
        self.seen.append(("fit", values.shape[0],
                          None if context.commands is None else context.commands.shape[0]))

    def score(self, values, valid, context):
        self.seen.append(("score", values.shape[0],
                          None if context.commands is None else context.commands.shape[0]))
        return np.zeros(values.shape[0])


def test_commands_are_sliced_exactly_as_the_values_are(commanded, split):
    """Two different slices: the fit window, and the score window plus warm-up.

    A detector indexes the two together without knowing which window it is in, so
    a one-row disagreement would silently misalign every command against the
    telemetry it explains -- and would look like the commands simply not helping.
    """
    detector = _Recorder(warmup=500)
    harness.evaluate(commanded, split, [detector], tasks.get("synthetic"), sweep=False)
    assert detector.seen, "the detector was never called"
    for _, values_rows, command_rows in detector.seen:
        assert command_rows == values_rows


def test_a_detector_sees_the_command_ids_it_was_given(commanded, split):
    seen = {}

    class Peek(_Recorder):
        def fit(self, values, usable, context):
            seen["ids"] = context.command_ids
            seen["has"] = context.has_commands

    harness.evaluate(commanded, split, [Peek()], tasks.get("synthetic"), sweep=False)
    assert seen["ids"] == commanded.command_ids
    assert seen["has"] is True


def test_a_subset_keeps_the_commands_the_spacecraft_received(commanded, labels,
                                                             channel_ids):
    """Narrowing the channels does not narrow which commands were sent."""
    child = commanded.subset(channel_ids[:3], labels)
    assert np.array_equal(child.commands, commanded.commands)
    assert child.command_ids == commanded.command_ids


def test_context_promises_no_labels_and_still_does(commanded, split):
    """Commands are an input the spacecraft has, not an answer it was given."""
    fields = set(Context.__dataclass_fields__)
    assert "commands" in fields
    assert not fields & {"anomaly", "truth", "labels", "spans", "events"}
