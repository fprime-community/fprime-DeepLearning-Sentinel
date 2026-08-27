"""One run has one provenance.

`RunRecord.git_commit` was a `default_factory` over a plain function, so it
re-ran for every record a run constructed -- one per channel set. A paired run
takes long enough for the repository to move underneath it, and one did:
`runs/m1-g8.9.10/lstm-telemanom/2026-08-26T212610Z-1f8b6fd6.json` stamps
`1bf6710` on `m1-g8.9.10` and `5bab55e` on `m1-ss5`. Same weights, same bundle,
same invocation, two answers to *which code produced this*.

No metric moved and none can: the field is provenance, not a measurement. What
it cost was the ability to trust a stamp, which is the thing provenance is for.
"""
from __future__ import annotations

import sentinel_eval.scorecard as sc


def _fresh():
    """A cache cleared between tests, so ordering cannot make one of them pass."""
    sc.git_commit.cache_clear()


def test_the_commit_is_resolved_once_per_process(monkeypatch):
    _fresh()
    calls = []

    class Result:
        def __init__(self, out):
            self.stdout = out

    def fake_run(*args, **kwargs):
        calls.append(args)
        return Result(f"commit{len(calls)}\n")

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    assert sc.git_commit() == "commit1"
    assert sc.git_commit() == "commit1"
    assert sc.git_commit() == "commit1"
    assert len(calls) == 1, "git was asked more than once for one process's commit"
    _fresh()


def test_two_records_in_one_run_agree_even_if_the_repository_moves(monkeypatch):
    """The defect, reproduced: HEAD changes between the two records of a pair."""
    _fresh()
    moving = iter(["aaaaaaa\n", "bbbbbbb\n"])

    class Result:
        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(sc.subprocess, "run",
                        lambda *a, **k: Result(next(moving, "ccccccc\n")))
    first = sc.RunRecord(task={"id": "m1-g8.9.10"}, bundle={}, coverage=[])
    second = sc.RunRecord(task={"id": "m1-ss5"}, bundle={}, coverage=[])
    assert first.git_commit == second.git_commit == "aaaaaaa"
    _fresh()


def test_an_unavailable_git_still_yields_a_stamp(monkeypatch):
    """Provenance degrades to a word, never to an exception mid-run."""
    _fresh()

    def explode(*args, **kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(sc.subprocess, "run", explode)
    assert sc.git_commit() == "unknown"
    _fresh()


def test_an_empty_answer_is_not_mistaken_for_a_commit(monkeypatch):
    _fresh()

    class Result:
        stdout = "  \n"

    monkeypatch.setattr(sc.subprocess, "run", lambda *a, **k: Result())
    assert sc.git_commit() == "unknown"
    _fresh()
