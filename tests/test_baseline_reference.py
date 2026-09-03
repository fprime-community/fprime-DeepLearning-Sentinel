"""The Level 1 baseline's Python side: the rule, and the defect it does not copy.

`baseline_reference` is what `flight/src/Baseline.cpp` is transcribed from, so
these tests do two jobs. They pin the reference against an implementation nobody
in this repository wrote -- `numpy.nanstd` -- and they pin its agreement with
`baselines._rolling`.

Three of them were written at work item 9 to pin the SIZE of a divergence, with
instructions to invert themselves if it ever closed. Work item 9.5 closed it, so
they are inverted and the numbers they used to pin are kept in their docstrings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel_models import baseline_reference as br
from sentinel_models.baselines import _rolling

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "flight" / "test" / "vectors"
W = br.WINDOW


def unit_normal(steps=600, channels=5, seed=3):
    return np.random.default_rng(seed).standard_normal((steps, channels)).astype(np.float32)


def offset(steps=600, channels=6, seed=4, mean=1000.0, sigma=3.0):
    """The regime that exposes the cancellation. Tier B3's shape."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((steps, channels)) * sigma + mean).astype(np.float32)


def numpy_windowed_std(values, window=W):
    """A third implementation, using nothing this project wrote."""
    x = np.asarray(values, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    for t in range(x.shape[0]):
        out[t] = np.nanstd(x[max(0, t - window + 1):t + 1], axis=0)
    return out


# -- the reference is right -------------------------------------------------

@pytest.mark.parametrize("values", [unit_normal(), offset()], ids=["unit", "offset"])
def test_the_reference_matches_numpy_nanstd(values):
    """The claim the flight transcription rests on."""
    assert np.abs(br.spread(values) - numpy_windowed_std(values)).max() < 1e-9


def test_the_streaming_and_vectorised_forms_agree(): 
    """Same rule, two shapes. They differ only by summation order."""
    values = unit_normal(steps=400, channels=4)
    stream = br.StreamingBaseline(4, np.full(4, 0.5), 1.0)
    rows = []
    for t in range(values.shape[0]):
        stream.step(values[t])
        rows.append(stream.channel_scores.copy())
    batch = br.score(values, np.full(4, 0.5))
    assert np.abs(np.array(rows) - batch).max() < 1e-13


def test_nan_is_excluded_from_the_count_and_the_sums():
    values = unit_normal(steps=300, channels=3, seed=7)
    values[10:40, 1] = np.nan
    got = br.spread(values)
    assert np.isfinite(got).all(), "a NaN input must not produce a NaN spread"
    assert np.abs(got - numpy_windowed_std(values)).max() < 1e-9


def test_a_channel_that_never_moves_has_zero_spread_not_a_negative_variance():
    values = np.zeros((200, 2), dtype=np.float32)
    values[:, 1] = 1000.0
    assert br.spread(values).max() == 0.0


# -- the defect it deliberately does not copy (D37) --------------------------

def test_rolling_now_agrees_with_the_rule():
    """INVERTED at work item 9.5, which is what the previous version asked for.

    This test used to assert `gap > 1.0` -- that `_rolling` disagreed with the
    flight rule by more than the quantity being measured -- and its own docstring
    said: "If this test starts failing because the gap has CLOSED, `_rolling` has
    been fixed ... and this test should then be inverted rather than deleted."
    It has been (D8 correctness fix, D37, `docs/MODELS.md` 21), so it is.

    The measurement it used to pin, kept because deleting it would delete the
    evidence: on this data the gap was **7.6584e+00** against a true sigma of
    3.0. It is now 1.8284e-08.
    """
    values = offset(steps=8000, channels=12, seed=9)
    gap = np.abs(_rolling(values, W, "std") - br.spread(values))[W:].max()
    assert gap < 1e-5, f"the harness and the flight rule have diverged again: {gap:.4e}"


def test_rolling_no_longer_produces_spurious_zeros():
    """The other half of the defect, also inverted.

    Cancellation drove the variance negative and the floor at `baselines.py:55`
    clamped it to exactly zero: 3,975 of them against float64's 1,123 on this
    project's own fixture. Both are now zero.
    """
    values = offset(steps=8000, channels=12, seed=9)
    defective = (_rolling(values, W, "std")[W:] == 0.0).sum()
    correct = (br.spread(values)[W:] == 0.0).sum()
    assert defective == correct == 0, f"{defective} zeros against {correct}"


def test_the_two_agree_in_the_benign_regime_as_they_always_did():
    """Unit-normal data is where `_rolling` was usable even before the fix.

    Kept because it is the record of why the defect survived so long: nothing in
    the test suite had ever run it on offset data, and in this regime there was
    nothing to see.
    """
    values = unit_normal(steps=400, channels=6, seed=11)
    assert np.abs(_rolling(values, W, "std") - br.spread(values))[W:].max() < 1e-5


# -- the flight semantics ----------------------------------------------------

def test_the_warmup_gate_matches_the_harness_convention():
    """Silent for calls 1..WINDOW, speaking on WINDOW + 1. See StreamingBaseline.warmed."""
    stream = br.StreamingBaseline(2, np.ones(2), -1.0)   # everything crosses
    values = np.array([1.0, 2.0], dtype=np.float32)
    for _ in range(W):
        stream.step(values)
        assert not stream.emitted
    assert stream.crossing, "the gate under test must be warm-up, not the threshold"
    stream.step(values)
    assert stream.emitted


def test_an_invalid_tick_scores_negative_infinity_and_never_alarms():
    stream = br.StreamingBaseline(3, np.ones(3), -1e9)
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    for _ in range(W + 5):
        stream.step(values)
    assert stream.emitted
    stream.step(values, valid=False)
    assert stream.score == float("-inf")
    assert not stream.crossing and not stream.emitted


def test_a_reset_restarts_the_warmup():
    stream = br.StreamingBaseline(2, np.ones(2), -1.0)
    values = np.array([1.0, 2.0], dtype=np.float32)
    for _ in range(W + 5):
        stream.step(values)
    assert stream.emitted
    stream.reset()
    stream.step(values)
    assert not stream.emitted


def test_ties_keep_the_lowest_channel_index():
    """The rule `Detector.cpp:100-106` uses, so the warning names the same channel."""
    combined, peak = br.reduce_scores(np.array([[2.0, 2.0, 1.0]]))
    assert peak[0] == 0 and combined[0] == 2.0


# -- the committed vectors ---------------------------------------------------

def test_the_committed_vectors_regenerate_from_their_seeds():
    """A fresh clone must reproduce them byte for byte, or they are not evidence."""
    manifest = json.loads((VECTORS / "baseline_manifest.json").read_text())
    assert [row["tier"] for row in manifest] == ["b1", "b2", "b3", "b4"]
    for row in manifest:
        path = VECTORS / f"{row['tier']}.bvec"
        assert path.exists(), f"{path} is committed evidence and is missing"
        assert path.stat().st_size == row["vector_bytes"]
        assert row["steps"] == 400 and row["window"] == W
        assert row["crossings"] > 0 and row["emitted"] > 0, (
            "a tier where nothing crosses pins the flags weakly")


def test_the_offset_tier_is_present_because_it_is_the_regression_guard():
    """Tier B3 exists so a float32 reimplementation fails rather than passes."""
    manifest = {row["tier"]: row for row in
                json.loads((VECTORS / "baseline_manifest.json").read_text())}
    assert "1000" in manifest["b3"]["provenance"]
