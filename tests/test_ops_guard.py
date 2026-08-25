"""The budget enforcement that did not previously exist.

`OPS_TRIPWIRE` and `OPS_CEILING` had two consumers before this: a projection made
before a run, and a percentage printed after it. Nothing raised.
"""
from __future__ import annotations

import pytest

from sentinel_data import config as C
from sentinel_data import r2
from sentinel_eval.errors import OpsCeilingExceeded, OpsTripwire
from sentinel_eval.ops import Budget, Ledger, commit


def test_a_runaway_run_is_stopped_at_the_tripwire():
    budget = Budget()
    for _ in range(C.OPS_TRIPWIRE - 1):
        budget.record("GetObject")
    with pytest.raises(OpsTripwire, match="shaped"):
        budget.record("GetObject")


def test_a_human_may_acknowledge_the_tripwire():
    budget = Budget(acknowledge_tripwire=True)
    for _ in range(C.OPS_TRIPWIRE + 5):
        budget.record("GetObject")
    assert budget.tripwire_hit and budget.class_b == C.OPS_TRIPWIRE + 5


def test_the_monthly_ceiling_has_no_override():
    budget = Budget(prior_class_b=C.OPS_CEILING["class_b"], acknowledge_tripwire=True)
    with pytest.raises(OpsCeilingExceeded, match="no override"):
        budget.record("GetObject")


def test_the_ceiling_counts_the_whole_month_not_the_run():
    budget = Budget(prior_class_a=C.OPS_CEILING["class_a"] - 1, acknowledge_tripwire=True)
    budget.record("PutObject")                       # exactly at the ceiling
    with pytest.raises(OpsCeilingExceeded):
        budget.record("PutObject")


def test_list_operations_remain_banned_outright():
    for operation in ("ListObjectsV2", "ListObjects", "ListBuckets"):
        with pytest.raises(RuntimeError, match="LIST"):
            Budget().record(operation)


def test_operations_are_classified_the_way_they_are_billed():
    budget = Budget()
    budget.record("PutObject")
    budget.record("GetObject")
    budget.record("HeadObject")
    budget.record("DeleteObject")                    # free
    assert (budget.class_a, budget.class_b) == (1, 2)


def test_the_ledger_is_read_modify_write_never_reset(monkeypatch):
    """`publish_ledger` used `new_ledger()`, wiping the month on every run."""
    existing = r2.roll_and_add(r2.new_ledger(), 200, 300)
    written = {}

    def fake_put(client, bucket, key, blob, md5, metadata=None):
        written["blob"] = blob

    monkeypatch.setattr(r2, "put_bytes", fake_put)
    ledger = Ledger(document=existing, month_class_a=200, month_class_b=300, existed=True)
    budget = Budget(prior_class_a=200, prior_class_b=300)
    budget.record("GetObject")

    document = commit(None, "bucket", ledger, budget, log=lambda *a: None)
    month = document["months"][r2.current_month()]
    assert month["class_a"] == 201                   # 200 prior + this run's own PUT
    assert month["class_b"] == 301                   # 300 prior + one GET
    assert written["blob"]


def test_a_missing_ledger_starts_fresh_but_a_broken_one_does_not(monkeypatch):
    """A transient failure must not report the month's spend as zero."""
    class Boom:
        class exceptions:
            class NoSuchKey(Exception):
                pass

    def explode(client, bucket, key):
        raise ValueError("network")

    monkeypatch.setattr(r2, "get_bytes", explode)
    with pytest.raises(ValueError):
        r2.fetch_ledger(Boom(), "bucket")
