"""The detector against the harness's contract, including the paths it takes.

`cli.py` builds one detector object and reuses it across three folds and both
paired channel sets -- twelve channels, then six. Most of what follows exists
because that reuse is easy to get wrong in a way that scores rather than crashes.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.detector import Context, reduce_scores
from sentinel_models import detectors as D
from sentinel_models.lstm import Hyper
from sentinel_models.reference import ReferenceError
from sentinel_models.telemanom import Config

STEPS = 6000


@pytest.fixture(autouse=True)
def _clean_caches():
    D.clear_caches()
    yield
    D.clear_caches()


def _tiny(**kwargs) -> D.ForecastDetector:
    return D.ForecastDetector(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                    max_epochs=2, sequence_budget_divisor=8,
                    max_validation_sequences=64, **kwargs.pop("hyper", {})),
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300, **kwargs)


def _data(channels=3, seed=0):
    rng = np.random.default_rng(seed)
    driver = np.cumsum(rng.standard_normal(STEPS)) * 0.01
    values = np.stack([driver + 0.1 * rng.standard_normal(STEPS)
                       for _ in range(channels)], axis=1).astype(np.float32)
    return values, np.ones(STEPS, dtype=bool)


def _context(channels, fold=0, window=(0, 3000)):
    return Context(mission="missionX", channels=tuple(f"c{i}" for i in range(channels)),
                   groups=(1,), period_seconds=30, fold=fold, window=window)


# -- the contract -----------------------------------------------------------
def test_score_returns_one_value_per_timestep():
    values, usable = _data()
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    scores = detector.score(values, np.ones_like(values, dtype=bool), _context(3))
    assert scores.shape == (STEPS,)
    combined, per_channel = reduce_scores(scores, STEPS)
    assert per_channel is None and combined.shape == (STEPS,)


def test_the_score_is_one_dimensional_on_purpose():
    """(T, C) is permitted by the contract and would cost a gigabyte twice over
    on an eleven-million-step window once `reduce_scores` casts it to float64."""
    values, usable = _data()
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    assert detector.score(values, None, _context(3)).ndim == 1
    assert detector.last_attribution.dtype == np.int8


def test_the_ndt_operating_point_is_the_fixed_one():
    """telemanom's threshold moves, so it lives in the score; the cut is 1.0."""
    assert _tiny().threshold_from(np.array([0.0, 5.0, 100.0])) == 1.0


def test_the_quantile_variant_defers_to_the_harness():
    detector = D.TelemanomQuantile(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3),
        config=Config(error_window=300, stride=30, smoothing_window=15))
    threshold = detector.threshold_from(np.linspace(0.0, 1.0, 10_000))
    assert 0.99 < threshold < 1.0        # the 99.9th percentile, not a constant


def test_warmup_covers_both_the_recurrence_and_the_first_error_window():
    detector = _tiny()
    assert detector.warmup_steps == 40 + 300


def test_scoring_before_fitting_is_refused():
    with pytest.raises(ReferenceError, match="before fit"):
        _tiny().score(np.zeros((10, 3), dtype=np.float32), None, _context(3))


# -- the reuse the CLI actually performs ------------------------------------
def test_one_detector_refits_at_a_different_channel_count():
    """m1-g8.9.10 then m1-ss5: twelve channels, then six, same object.

    A model carried over would either fail on the width change or -- much worse --
    score the second set with the first set's weights.
    """
    twelve, usable = _data(channels=6)
    detector = _tiny()
    detector.fit(twelve[:3000], usable[:3000], _context(6))
    wide = detector.score(twelve, None, _context(6))

    six = np.ascontiguousarray(twelve[:, :3])
    detector.fit(six[:3000], usable[:3000], _context(3))
    narrow = detector.score(six, None, _context(3))

    assert wide.shape == narrow.shape == (STEPS,)
    assert detector._weights.n_channels == 3


def test_refitting_replaces_the_model_rather_than_adding_to_it():
    values, usable = _data()
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3, fold=0))
    first = detector._weights
    detector.fit(values[:3000], usable[:3000], _context(3, fold=1, window=(0, 3000)))
    assert detector._weights is not first


# -- the weight cache -------------------------------------------------------
def test_identical_inputs_reuse_the_trained_weights():
    """Two detectors sharing a forecaster must train once, not twice."""
    values, usable = _data()
    first, second = _tiny(), _tiny()
    first.fit(values[:3000], usable[:3000], _context(3))
    second.fit(values[:3000], usable[:3000], _context(3))
    assert second._weights is first._weights


def test_a_changed_hyperparameter_forces_a_fresh_fit():
    values, usable = _data()
    first = _tiny()
    second = _tiny(hyper={"seed": 41})
    first.fit(values[:3000], usable[:3000], _context(3))
    second.fit(values[:3000], usable[:3000], _context(3))
    assert second._weights is not first._weights


def test_changed_data_forces_a_fresh_fit():
    first, second = _tiny(), _tiny()
    a, usable = _data(seed=0)
    b, _ = _data(seed=1)
    first.fit(a[:3000], usable[:3000], _context(3))
    second.fit(b[:3000], usable[:3000], _context(3))
    assert second._weights is not first._weights


def test_the_cache_can_be_declined():
    values, usable = _data()
    first, second = _tiny(), _tiny(reuse_weights=False)
    first.fit(values[:3000], usable[:3000], _context(3))
    second.fit(values[:3000], usable[:3000], _context(3))
    assert second._weights is not first._weights


# -- gaps -------------------------------------------------------------------
def test_an_unobserved_stretch_does_not_poison_the_state():
    """A NaN entering the recurrence would make every later score NaN."""
    values, usable = _data()
    values[4000:4200, 1] = np.nan
    valid = np.isfinite(values)
    usable[4000:4200] = False

    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    scores = detector.score(values, valid, _context(3))
    assert not np.isnan(scores).any()
    assert (scores[4000:4200] == -np.inf).all()      # nothing measured is no alarm
    assert np.isfinite(scores[4300:]).all()


def test_a_gap_at_the_very_start_is_survivable():
    values, usable = _data()
    values[:50] = np.nan
    usable[:50] = False
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    scores = detector.score(values, np.isfinite(values), _context(3))
    assert np.isfinite(scores[1000:]).all()


# -- causality --------------------------------------------------------------
def test_the_forecast_cannot_see_the_step_it_is_forecasting():
    """A detector that peeks is not something that can fly, and the harness
    should not measure one that does."""
    values, usable = _data()
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    baseline = detector.score(values, None, _context(3))

    disturbed = values.copy()
    disturbed[5000:] += 7.0                      # change only the far future
    D.clear_caches()
    detector._weights, detector._fill = detector._weights, detector._fill
    changed = detector.score(disturbed, None, _context(3))
    assert np.allclose(baseline[:4000], changed[:4000], atol=1e-6)


def test_a_second_detector_reuses_every_fold_not_just_the_last():
    """`harness.evaluate` loops detectors outermost.

    So by the time the second detector reaches fold 0, the first has already
    worked through fold 2. A single-entry cache would have been evicted and the
    whole run would train twice -- which is the only expensive part of it.
    """
    values, usable = _data()
    first, second = _tiny(), _tiny()
    fitted = []
    for fold in range(3):
        first.fit(values[:3000], usable[:3000], _context(3, fold=fold))
        fitted.append(first._weights)
    for fold in range(3):
        second.fit(values[:3000], usable[:3000], _context(3, fold=fold))
        assert second._weights is fitted[fold], f"fold {fold} retrained"
