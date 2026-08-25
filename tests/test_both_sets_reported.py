"""Both channel sets travel together, permanently.

We changed the evaluation set after seeing a result we did not like. However
sound the reasoning, that is externally indistinguishable from cherry-picking.
Reporting both forever is what converts a suspicious edit into a stated scope
decision -- you cannot cherry-pick if you never discard anything. Barring partial
artifacts from RESULTS.md makes single-set publication structurally impossible
rather than merely discouraged.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import tasks
from sentinel_eval.errors import TaskError
from sentinel_eval.scorecard import GateRecord, RunRecord


def test_the_demoted_set_is_retained_not_deleted():
    assert "m1-ss5" in tasks.TASKS
    assert "DEMOTED" in tasks.get("m1-ss5").note
    assert "never discarded" in tasks.get("m1-ss5").note


def test_the_promotion_is_recorded_on_the_task_itself():
    note = tasks.get("m1-g8.9.10").note
    assert "PROMOTED" in note and "post-hoc" in note
    assert tasks.get("m1-g8.9.10").headline.startswith("GATE")


def test_each_paired_member_pulls_in_the_other():
    for member in ("m1-ss5", "m1-g8.9.10"):
        assert set(tasks.paired_with(member)) == {"m1-ss5", "m1-g8.9.10"}


def test_the_adoption_set_is_not_paired_with_a_recall_set():
    assert tasks.paired_with("m2-ss1") == ("m2-ss1",)


def test_a_partial_artifact_is_marked_for_exclusion():
    partial = GateRecord(records=[RunRecord(task={"id": "m1-ss5"}, bundle={}, coverage=[])],
                         partial=True)
    assert partial.as_dict()["partial"] is True
    assert "barred from docs/RESULTS.md" in partial.render()

    complete = GateRecord(records=[
        RunRecord(task={"id": "m1-g8.9.10"}, bundle={}, coverage=[]),
        RunRecord(task={"id": "m1-ss5"}, bundle={}, coverage=[])], partial=False)
    assert complete.as_dict()["sets"] == ["m1-g8.9.10", "m1-ss5"]
    assert "barred" not in complete.render()


def test_a_subset_costs_no_reads_and_matches_a_pinned_direct_load(bucket, catalog,
                                                                  labels, channel_ids):
    class Counting:
        def __init__(self, inner):
            self.inner, self.gets = inner, 0

        def get(self, key):
            self.gets += 1
            return self.inner.get(key)

    source = Counting(bucket)
    wide = bundle_mod.load(source, catalog, labels, mission="missionX",
                           channel_ids=channel_ids)
    after_load = source.gets

    narrow_ids = channel_ids[:3]
    view = wide.subset(narrow_ids, labels)
    assert source.gets == after_load                     # zero further reads

    direct = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=narrow_ids, start=wide.grid.start,
                             end=wide.grid.end)
    assert view.grid == direct.grid
    assert np.array_equal(view.values, direct.values, equal_nan=True)
    assert np.array_equal(view.truth.anomaly, direct.truth.anomaly)
    assert view.truth.spans == direct.truth.spans


def test_a_subset_must_actually_be_a_subset(loaded, labels):
    with pytest.raises(TaskError, match="not in this bundle"):
        loaded.subset(["channel_999"], labels)


def test_the_subset_records_what_it_came_from(loaded, labels, channel_ids):
    view = loaded.subset(channel_ids[:2], labels)
    assert view.provenance["derived_from"] == list(loaded.channel_ids)
    assert view.provenance["channels"] == channel_ids[:2]
