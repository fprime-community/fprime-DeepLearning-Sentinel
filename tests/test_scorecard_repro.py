"""Same detector plus same data must give the same numbers, on any day."""
from __future__ import annotations

import json

from sentinel_eval import harness, splits, synthetic, tasks
from sentinel_models import baselines


def _numbers(record) -> str:
    """Everything except the fields that legitimately vary between runs."""
    payload = json.loads(record.to_json())
    payload.pop("git_commit", None)
    payload.pop("operations", None)
    return json.dumps(payload, sort_keys=True)


def test_two_identical_runs_agree_exactly(loaded, split):
    task = tasks.get("synthetic")
    detectors = lambda: [baselines.MovingAverage(60), baselines.RandomScore(7)]
    first = harness.evaluate(loaded, split, detectors(), task, sweep=True)
    second = harness.evaluate(loaded, split, detectors(), task, sweep=True)
    assert _numbers(first) == _numbers(second)


def test_the_fixture_itself_is_deterministic():
    a, b = synthetic.build(seed=3, n=4_000), synthetic.build(seed=3, n=4_000)
    assert a.objects.keys() == b.objects.keys()
    assert all(a.objects[k] == b.objects[k] for k in a.objects)


def test_a_different_seed_gives_different_data():
    a, b = synthetic.build(seed=1, n=4_000), synthetic.build(seed=2, n=4_000)
    key = "esa-adb/v1/archive/missionX/ch_channel_1.parquet"
    assert a.objects[key] != b.objects[key]


def test_the_fingerprint_tracks_parameters_not_identity():
    assert baselines.MovingAverage(60).fingerprint() == baselines.MovingAverage(60).fingerprint()
    assert baselines.MovingAverage(60).fingerprint() != baselines.MovingAverage(90).fingerprint()


def test_provenance_carries_what_a_rerun_would_need(loaded, split):
    record = harness.evaluate(loaded, split, [baselines.AlwaysQuiet()],
                              tasks.get("synthetic"), sweep=False)
    for field in ("manifest_generated_utc", "manifest_schema_version", "channels",
                  "grid_period_seconds", "normalisation", "grid_steps"):
        assert field in record.bundle
    assert record.harness_version and record.git_commit
    assert record.limitation                      # the caveat travels with the numbers


def test_the_persistence_filter_is_causal_and_recorded(loaded, split):
    import numpy as np

    mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert harness.apply_persistence(mask, 2).tolist() == [False, True, False,
                                                           False, True, True]
    record = harness.evaluate(loaded, split, [baselines.AlwaysQuiet()],
                              tasks.get("synthetic"), sweep=False)
    assert record.scorecards[0].pooled["persistence"] == 1
