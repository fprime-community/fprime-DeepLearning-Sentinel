"""Sequence sampling: what a model is allowed to learn from, and in what shape.

Objective.md 6.1 is the constraint. A spacecraft has no failure examples, so a
model learns normality; if a labelled fault leaks into the training set the model
learns that the fault is normal, and the one thing this project cannot tolerate
is a detector taught to ignore the condition it exists to catch. The harness
supplies the mask; these tests assert the sampler respects it.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models import windows as W
from sentinel_models.reference import ReferenceError

WINDOW, PREDICTIONS = 8, 3
SPAN = WINDOW + PREDICTIONS


def _series(n=400, channels=2):
    return (np.arange(n * channels, dtype=np.float32).reshape(n, channels) / 100.0)


def _mask(n=400, holes=((100, 140), (300, 310))):
    usable = np.ones(n, dtype=bool)
    for lo, hi in holes:
        usable[lo:hi] = False
    return usable


# -- runs and splits --------------------------------------------------------
def test_runs_stop_at_every_hole():
    assert W.usable_runs(_mask(), SPAN) == [(0, 100), (140, 300), (310, 400)]


def test_a_run_too_short_to_hold_a_sequence_is_dropped():
    usable = np.zeros(100, dtype=bool)
    usable[10:15] = True                       # 5 steps, a sequence needs 11
    usable[50:90] = True
    assert W.usable_runs(usable, SPAN) == [(50, 90)]


def test_the_validation_split_is_the_chronological_tail():
    train, validate = W.split_runs(W.usable_runs(_mask(), SPAN), 0.2)
    assert max(hi for _, hi in train) <= min(lo for lo, _ in validate)


def test_the_split_cuts_a_run_rather_than_assigning_it_whole():
    """Mission1's longest usable runs span years; handing one over moves the
    boundary by an arbitrary amount."""
    runs = [(0, 1000)]
    train, validate = W.split_runs(runs, 0.25)
    assert train == [(0, 750)] and validate == [(750, 1000)]


def test_an_impossible_validation_fraction_is_refused():
    with pytest.raises(ReferenceError):
        W.split_runs([(0, 100)], 1.0)


# -- sampling ---------------------------------------------------------------
def test_no_sampled_sequence_crosses_an_unusable_step():
    values, usable = _series(), _mask()
    runs = W.usable_runs(usable, SPAN)
    sampler = W.SequenceSampler(values, runs, window=WINDOW, n_predictions=PREDICTIONS)
    for _ in range(50):
        starts = sampler._resolve(np.random.default_rng(0).integers(
            0, sampler.n_positions, size=64))
        for start in starts:
            assert usable[start:start + SPAN].all()


def test_shapes_and_time_alignment():
    values = _series()
    sampler = W.SequenceSampler(values, W.usable_runs(_mask(), SPAN),
                                window=WINDOW, n_predictions=PREDICTIONS)
    inputs, targets = sampler.draw(16, np.random.default_rng(1))
    assert inputs.shape == (16, WINDOW, 2)
    assert targets.shape == (16, PREDICTIONS, 2)
    # the target block is the values immediately after the window, in order
    step = values[1] - values[0]
    assert np.allclose(targets[:, 0], inputs[:, -1] + step)


def test_sampling_is_deterministic_under_a_fixed_seed():
    values = _series()
    sampler = W.SequenceSampler(values, W.usable_runs(_mask(), SPAN),
                                window=WINDOW, n_predictions=PREDICTIONS)
    a = sampler.draw(8, np.random.default_rng(7))[0]
    b = sampler.draw(8, np.random.default_rng(7))[0]
    assert np.array_equal(a, b)


def test_a_window_with_no_room_for_a_sequence_is_refused_not_padded():
    sampler = W.SequenceSampler(_series(), [], window=WINDOW, n_predictions=PREDICTIONS)
    with pytest.raises(ReferenceError, match="fragmented"):
        sampler.draw(4, np.random.default_rng(0))


def test_a_non_finite_value_inside_a_usable_run_is_refused():
    """The mask and the values disagreeing is a broken loader, not a bad batch."""
    values = _series()
    values[50, 0] = np.nan
    sampler = W.SequenceSampler(values, [(0, 400)], window=WINDOW,
                                n_predictions=PREDICTIONS)
    with pytest.raises(ReferenceError, match="non-finite"):
        sampler.gather(np.array([45]))


def test_positions_are_sampled_uniformly_rather_than_by_run():
    """Two runs, one twenty times the other: draws must follow steps, not runs."""
    usable = np.zeros(4200, dtype=bool)
    usable[0:200] = True
    usable[200:4200] = True
    runs = [(0, 200), (1000, 5000)]
    sampler = W.SequenceSampler(_series(5000), runs, window=WINDOW,
                                n_predictions=PREDICTIONS)
    starts = sampler._resolve(np.random.default_rng(0).integers(
        0, sampler.n_positions, size=20000))
    short = float((starts < 200).mean())
    assert 0.03 < short < 0.07, f"short run drew {short:.1%} of samples, expected ~4.7%"


# -- putting forecasts back on the timeline ---------------------------------
def test_aggregation_recovers_a_perfect_forecast():
    steps, predictions, channels = 40, 5, 2
    forecast = np.zeros((steps, predictions, channels), dtype=np.float32)
    for t in range(steps):
        for j in range(predictions):
            forecast[t, j] = t + 1 + j          # forecast for time t+1+j is t+1+j
    recovered = W.aggregate_predictions(forecast)
    assert np.allclose(recovered[1:, 0], np.arange(1, steps))


def test_aggregation_never_reads_the_present_or_the_future():
    """A forecast for t is built from predictions made at t-1 and earlier."""
    steps, predictions = 30, 4
    forecast = np.zeros((steps, predictions, 1), dtype=np.float32)
    forecast[20] = 100.0                        # a prediction made AT step 20
    recovered = W.aggregate_predictions(forecast)
    assert recovered[20, 0] == 0.0              # cannot have reached step 20
    assert recovered[21, 0] > 0.0               # reaches 21 onwards
