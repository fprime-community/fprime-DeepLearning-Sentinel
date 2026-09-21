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


def test_fitting_survives_a_usable_mask_that_disagrees_with_the_values():
    """`Bundle.subset` rebuilds truth from labels alone and never reads `valid`.

    So on a subset -- which is how `m1-ss5` is scored -- gap timesteps arrive
    marked usable while the values there are NaN. The trivial baselines are
    NaN-tolerant and never noticed; a forecaster is not, and one NaN entering the
    recurrence makes every state after it NaN. This is the exact shape that
    killed a 75-minute run at its last stage.
    """
    values, usable = _data()
    values[1500:1600, 2] = np.nan          # unobserved, but still marked usable
    assert usable[1500:1600].all()

    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    scores = detector.score(values, np.isfinite(values), _context(3))
    assert np.isfinite(detector._weights.head_w).all(), "NaN reached the weights"
    assert not np.isnan(scores).any()


def test_weights_persist_between_processes_and_are_keyed_by_content(tmp_path, monkeypatch):
    """Rule 1 permits outputs under runs/; a stale checkpoint must be impossible.

    The filename is the content digest -- hyperparameters, channel set, fold
    window, and a strided sample of the data -- so weights fitted on anything
    else cannot be found, and a cache hit is a cache hit on the same fit.
    """
    monkeypatch.setattr(D, "WEIGHT_STORE", tmp_path / "_weights")
    values, usable = _data()

    cold = _tiny()
    cold.fit(values[:3000], usable[:3000], _context(3))
    stored = list((tmp_path / "_weights").glob("*.npz"))
    assert len(stored) == 1, "the fit did not persist"

    D.clear_caches()                                   # forget the in-memory copy
    warm = _tiny()
    warm.fit(values[:3000], usable[:3000], _context(3))
    assert np.array_equal(warm._weights.head_w, cold._weights.head_w)
    assert len(list((tmp_path / "_weights").glob("*.npz"))) == 1, "refitted needlessly"

    D.clear_caches()
    other, _ = _data(seed=99)                          # different data, same shape
    fresh = _tiny()
    fresh.fit(other[:3000], usable[:3000], _context(3))
    assert len(list((tmp_path / "_weights").glob("*.npz"))) == 2, "digest collided"


def test_no_cache_refuses_persisted_weights(tmp_path, monkeypatch):
    """--no-cache is what makes a published result reproducible from cold."""
    monkeypatch.setattr(D, "WEIGHT_STORE", tmp_path / "_weights")
    values, usable = _data()
    _tiny().fit(values[:3000], usable[:3000], _context(3))
    D.clear_caches()

    D.set_caching(False)
    try:
        cold = _tiny()
        cold.fit(values[:3000], usable[:3000], _context(3))
        assert not D._WEIGHTS, "a --no-cache run populated the in-memory cache"
    finally:
        D.set_caching(True)


def test_a_decision_layer_sweep_does_not_recompute_the_forecast():
    """The grid changes nothing upstream of the channel reduction.

    So sweeping agreement must read one cached computation. Checking the cache
    after computing the errors -- which is what the first version did -- would
    recompute the forecast and the smoothing for every point in the grid, the
    expensive two thirds of the work, to answer a question about the cheap third.
    """
    values, usable = _data()
    calls = {"n": 0}

    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))
    original = detector._smoothed_errors

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    detector._smoothed_errors = counted
    for k in (1, 2, 3):
        detector.agreement = k
        # A window distinct from the fitting one, as the harness always scores.
        detector.score(values, None, _context(3, window=(3000, 6000)))
    assert calls["n"] == 1, f"forecast recomputed {calls['n']} times for one sweep"


def test_agreement_requires_that_many_channels_to_be_over_threshold():
    """k=1 is the maximum; k=3 needs all three of this fixture's channels."""
    values, usable = _data()
    values[4000:4300, 0] += 2.0                       # one channel alone
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3))

    fired = {}
    for k in (1, 2, 3):
        detector.agreement = k
        fired[k] = int((detector.score(values, None,
                                       _context(3, window=(3000, 6000))) >= 1.0).sum())
    assert fired[1] >= fired[2] >= fired[3], f"agreement did not tighten: {fired}"


def test_a_quantile_sweep_does_not_recompute_the_forecast():
    """Layer 0 sweeps quantile mode across k, and quantile mode used to miss the cache.

    NDT mode reduces through `top_ratios` and was cached; quantile mode reduced
    straight from the smoothed errors and was not. A 24-cell grid would have paid
    the forecast and the smoothing on every cell -- the expensive two thirds, to
    answer a question about the cheap third. Counted rather than assumed, because
    the ordering is what breaks and an ordering is invisible in a result.
    """
    values, usable = _data()
    calls = {"n": 0}

    detector = D.TelemanomQuantile(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                    max_epochs=2, sequence_budget_divisor=8,
                    max_validation_sequences=64),
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300)
    detector.fit(values[:3000], usable[:3000], _context(3))
    original = detector._smoothed_errors

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    detector._smoothed_errors = counted
    for k in (1, 2, 3):
        detector.agreement = k
        # A window distinct from the fitting one, as the harness always scores.
        detector.score(values, None, _context(3, window=(3000, 6000)))
    assert calls["n"] == 1, f"forecast recomputed {calls['n']} times for one sweep"


def test_the_two_reductions_do_not_share_a_cache_entry():
    """Ratios and raw errors have the same shape and different meaning.

    Both reduce `(T, C)` to `(depth, T)` over the same window, so a key that
    omitted the mode would serve one detector the other's numbers -- silently,
    and with entirely plausible values.
    """
    values, usable = _data()
    shared = dict(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                    max_epochs=2, sequence_budget_divisor=8,
                    max_validation_sequences=64),
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300)

    ndt = D.ForecastDetector(**shared)
    ndt.fit(values[:3000], usable[:3000], _context(3))
    ndt_scores = ndt.score(values, None, _context(3, window=(3000, 6000)))

    quantile = D.TelemanomQuantile(**shared)
    quantile.fit(values[:3000], usable[:3000], _context(3))
    quantile_scores = quantile.score(values, None, _context(3, window=(3000, 6000)))

    assert not np.allclose(ndt_scores, quantile_scores), \
        "quantile mode was served the dynamic threshold's ratios"


def test_agreement_tightens_quantile_mode_too():
    values, usable = _data()
    values[4000:4300, 0] += 2.0
    detector = D.TelemanomQuantile(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                    max_epochs=2, sequence_budget_divisor=8,
                    max_validation_sequences=64),
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300)
    detector.fit(values[:3000], usable[:3000], _context(3))

    tops = {}
    for k in (1, 2, 3):
        detector.agreement = k
        tops[k] = float(np.nanmax(
            detector.score(values, None, _context(3, window=(3000, 6000)))))
    assert tops[1] >= tops[2] >= tops[3], f"k did not tighten: {tops}"


def test_a_commanded_model_survives_the_round_trip(tmp_path, monkeypatch):
    """`n_exogenous` was not being written, so commanded weights would not reload.

    It failed safe -- `_load_weights` catches and returns a cache miss, so the
    consequence was silent refitting rather than a wrong answer. But an hour of
    GPU fitting was thrown away before verification caught it, and "fails safe"
    is not "works".
    """
    monkeypatch.setattr(D, "WEIGHT_STORE", tmp_path / "_weights")
    values, usable = _data()
    commands = np.zeros((values.shape[0], 2), dtype=np.uint8)
    commands[::300] = 1

    context = Context(mission="missionX", channels=("a", "b", "c"), groups=(1,),
                      period_seconds=30, fold=0, window=(0, 3000),
                      commands=commands[:3000], command_ids=("tc_a", "tc_b"))
    detector = D.TelemanomCommanded(
        hyper=Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                    max_epochs=2, sequence_budget_divisor=8,
                    max_validation_sequences=64),
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300)
    detector.fit(values[:3000], usable[:3000], context)
    fitted = detector._weights
    assert fitted.n_exogenous == 4, "two commands, two features each"

    D.clear_caches()
    # (!) THIS MIRRORS `detectors.py`'s FIT KEY BY HAND, ON PURPOSE: the point is
    # that the saved artifact is findable by an INDEPENDENT computation, not by
    # calling the same function that stored it. The cost is that a term added
    # there and not here makes this fail with "did not survive the round trip",
    # which is exactly what happened when Section 74 added the initial-weights
    # digest for D76's warm start. `None` is the cold-fit value and every caller
    # in this file is a cold fit.
    reloaded = D._load_weights(D._digest(
        (detector.hyper.as_dict_key(), context.channels, context.fold, context.window,
         values[:3000].shape, D._sample_digest(values[:3000]),
         D._digest(usable[:3000][::997]), D._sample_digest(commands[:3000]),
         None)))
    assert reloaded is not None, "the commanded model did not survive the round trip"
    assert reloaded[0].n_exogenous == fitted.n_exogenous
    assert reloaded[0].n_inputs == fitted.n_inputs
    assert np.array_equal(reloaded[0].head_w, fitted.head_w)


def test_the_ndt_operating_point_ignores_its_argument():
    """The invariant that makes skipping the fitting window's threshold safe.

    NDT folds its moving threshold into the score, so the operating point is a
    fixed 1.0 and the training scores are never consulted. `score()` relies on
    that to skip the dynamic threshold over the 22.1M-step fitting window -- two
    thirds of a run's scoring cost, spent producing a number nothing reads.

    If this ever stops holding, the skip becomes wrong, so it is pinned here
    rather than left as a comment.
    """
    detector = _tiny()
    for argument in (np.array([0.0]), np.array([1e9, -1e9]),
                     np.linspace(0, 1000, 5000), np.array([np.nan, np.inf])):
        assert detector.threshold_from(argument) == 1.0


def test_the_fitting_window_and_the_scored_window_do_not_collide():
    """The skip keys on the window, so the two must be distinguishable."""
    values, usable = _data()
    detector = _tiny()
    detector.fit(values[:3000], usable[:3000], _context(3, window=(0, 3000)))
    assert detector._fit_window == (0, 3000)

    scored = detector.score(values, None, _context(3, window=(3000, 6000)))
    assert np.isfinite(scored[3000:]).any(), "the scored window produced nothing"


# -- the LSTM's identity is pinned, so adding an architecture cannot move it ---
#: What `Hyper().as_dict()` emitted on the day the twelve production LSTM fits
#: were banked. The weight-cache key (`ForecastDetector.fit`) and every published
#: artifact fingerprint derive from it, so a field added to `Hyper` for another
#: architecture must leave this untouched -- D14: an optional field at its null
#: value leaves the hash unmoved. If this test fails, every banked weight file
#: under runs/_weights/ has just become unreachable and every fingerprint in
#: docs/RESULTS.md has silently changed.
LSTM_HYPER_KEYS = ["window", "hidden", "dropout", "n_predictions", "batch_size",
                   "max_epochs", "patience", "min_improvement", "validation_fraction",
                   "learning_rate", "seed", "sequence_budget_divisor",
                   "max_validation_sequences", "restore_best"]
LSTM_HYPER_KEY_TUPLE = (
    ("batch_size", 70), ("dropout", 0.3), ("hidden", (80, 80)), ("learning_rate", 0.001),
    ("max_epochs", 35), ("max_validation_sequences", 2000), ("min_improvement", 0.001),
    ("n_predictions", 10), ("patience", 10), ("restore_best", True), ("seed", 0),
    ("sequence_budget_divisor", 180), ("validation_fraction", 0.2), ("window", 250),
)


def test_the_lstm_hyper_dict_is_byte_identical_to_the_banked_generation():
    assert list(Hyper().as_dict()) == LSTM_HYPER_KEYS
    assert Hyper().as_dict_key() == LSTM_HYPER_KEY_TUPLE


def test_the_lstm_cache_digest_is_unmoved():
    """A synthetic cache key in the shape `fit` builds, hashed before the change."""
    key = (Hyper().as_dict_key(), ("a", "b"), 0, (0, 10), (10, 2), "x", "y", None)
    assert D._digest(key) == "cc91392d878abda97c881ed8d4f26c56"


def test_the_published_fingerprints_are_unmoved():
    """The suffixes of the banked artifacts:
    runs/m1-g8.9.10/lstm-telemanom/2026-08-27T185132Z-6b9ebb0d.json and
    runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json."""
    assert D.ForecastDetector().fingerprint() == "6b9ebb0d"
    assert D.TelemanomQuantile().fingerprint() == "2717441a"
    assert D.TelemanomSmoke().fingerprint() == "9d5cca78"


# -- work item 5: the GRU through the same detector -----------------------------
def _tiny_gru(**kwargs) -> D.GRUForecastDetector:
    hyper = Hyper(window=40, hidden=(12, 12), n_predictions=3, batch_size=16,
                  max_epochs=2, sequence_budget_divisor=8, max_validation_sequences=64,
                  cell="gru", **kwargs.pop("hyper", {}))
    return D.GRUForecastDetector(
        hyper=hyper,
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300, **kwargs)


def test_the_cell_is_the_only_thing_in_the_key_that_differs():
    assert "cell" not in Hyper().as_dict(), "the LSTM's dict must not grow a field"
    assert Hyper(cell="gru").as_dict()["cell"] == "gru"
    assert Hyper(cell="gru").as_dict_key() != Hyper().as_dict_key()
    assert dict(Hyper(cell="gru").as_dict_key()) | {"cell": "lstm"} == \
        dict(Hyper().as_dict_key()) | {"cell": "lstm"}


def test_gru_and_lstm_fingerprints_differ():
    assert D.GRUForecastDetector().fingerprint() != D.ForecastDetector().fingerprint()
    assert D.GRUQuantile().fingerprint() != D.TelemanomQuantile().fingerprint()
    assert D.GRUSmoke().fingerprint() != D.TelemanomSmoke().fingerprint()


def test_a_detector_refuses_the_other_cell():
    with pytest.raises(ReferenceError, match="cell"):
        D.ForecastDetector(hyper=Hyper(cell="gru"))
    with pytest.raises(ReferenceError, match="cell"):
        D.GRUForecastDetector(hyper=Hyper())
    with pytest.raises(ReferenceError, match="cell"):
        Hyper(cell="transformer")


def test_the_registry_builds_the_gru_arms():
    from sentinel_models import registry

    for name, mode in (("gru-telemanom", D.NDT), ("gru-quantile", D.QUANTILE),
                       ("gru-smoke", D.NDT)):
        built = registry.build(name)
        assert (built.name, built.mode, built.cell, built.hyper.cell) == (name, mode, "gru", "gru")
    assert registry.build("gru-smoke").hyper == D.GRUSmoke().hyper
    assert D.GRUSmoke().hyper.window == D.TelemanomSmoke().hyper.window


def test_the_gru_fits_and_scores_through_the_same_path():
    values, usable = _data()
    detector = _tiny_gru()
    detector.fit(values[:3000], usable[:3000], _context(3))
    assert detector._weights.cell == "gru"
    assert detector._weights.layers[0].w_ih.shape == (36, 3)
    assert detector.report["best_epoch"] >= 0
    scored = detector.score(values, None, _context(3, window=(3000, 6000)))
    assert scored.shape == (STEPS,) and np.isfinite(scored[3000:]).any()


def test_the_two_cells_do_not_share_a_cache_entry():
    """Same data, same fold, same everything but the cell: two fits, two entries."""
    values, usable = _data()
    _tiny().fit(values[:3000], usable[:3000], _context(3))
    _tiny_gru().fit(values[:3000], usable[:3000], _context(3))
    shapes = sorted(w.layers[0].w_ih.shape[0] for w, _ in D._WEIGHTS.values())
    assert shapes == [36, 48], f"expected one GRU (3x12) and one LSTM (4x12) entry, got {shapes}"


def test_gru_weights_survive_the_round_trip_and_the_file_says_what_it_is(tmp_path, monkeypatch):
    """Written with `cell`, read back; read without it (an older file); refused when wrong."""
    monkeypatch.setattr(D, "WEIGHT_STORE", tmp_path / "_weights")
    values, usable = _data()
    detector = _tiny_gru()
    detector.fit(values[:3000], usable[:3000], _context(3))
    fitted = detector._weights
    path, = (tmp_path / "_weights").glob("*.npz")
    digest = path.stem

    D.clear_caches()
    reloaded = D._load_weights(digest)
    assert reloaded is not None and reloaded[0].cell == "gru"
    assert np.array_equal(reloaded[0].layers[0].w_hh, fitted.layers[0].w_hh)
    with np.load(path, allow_pickle=False) as blob:
        assert str(blob["cell"]) == "gru"
        arrays = {name: blob[name] for name in blob.files}

    del arrays["cell"]                                   # a file from before the field
    np.savez(path, **arrays)
    older = D._load_weights(digest)
    assert older is not None and older[0].cell == "gru", "the arrays say what it is"

    np.savez(path, **arrays, cell=np.array("lstm"))      # a file that lies
    assert D._load_weights(digest) is None, "a declared cell that disagrees with the arrays"


# -- work item 6: the TCN through the same detector -----------------------------
def _tiny_tcn(**kwargs) -> D.TCNForecastDetector:
    hyper = Hyper(window=40, hidden=(12, 12, 12), n_predictions=3, batch_size=16,
                  max_epochs=2, sequence_budget_divisor=8, max_validation_sequences=64,
                  cell="tcn", kernel=3, **kwargs.pop("hyper", {}))
    return D.TCNForecastDetector(
        hyper=hyper,
        config=Config(error_window=300, stride=30, smoothing_window=15, error_buffer=10),
        chunks=8, chunk_steps=300, **kwargs)


def test_the_tcn_key_carries_its_kernel_and_the_cells_do_not():
    assert "kernel" not in Hyper().as_dict() and "kernel" not in Hyper(cell="gru").as_dict()
    assert Hyper(cell="tcn").as_dict()["kernel"] == 3
    assert Hyper(cell="tcn").as_dict_key() != Hyper(cell="gru").as_dict_key()
    assert Hyper(cell="tcn", kernel=2).as_dict_key() != Hyper(cell="tcn").as_dict_key()
    with pytest.raises(ReferenceError, match="kernel"):
        Hyper(cell="tcn", kernel=1)


def test_the_registry_builds_the_tcn_arms():
    from sentinel_models import registry

    for name, mode in (("tcn-telemanom", D.NDT), ("tcn-quantile", D.QUANTILE), ("tcn-smoke", D.NDT)):
        built = registry.build(name)
        assert (built.name, built.mode, built.cell, built.hyper.cell) == (name, mode, "tcn", "tcn")
    assert registry.build("tcn-quantile").hyper == D.TCN_HYPER
    assert D.TCN_HYPER.receptive_field == 253 and D.TCN_HYPER.hidden == (50,) * 6
    assert D.TCNQuantile().fingerprint() != D.GRUQuantile().fingerprint() != D.TelemanomQuantile().fingerprint()
    with pytest.raises(ReferenceError, match="cell"):
        D.TCNForecastDetector(hyper=Hyper())


def test_the_tcn_fits_and_scores_through_the_same_path():
    values, usable = _data()
    detector = _tiny_tcn()
    detector.fit(values[:3000], usable[:3000], _context(3))
    assert detector._weights.cell == "tcn"
    assert detector._weights.receptive_field == 1 + 2 * 2 * 7
    assert detector.report["best_epoch"] >= 0
    scored = detector.score(values, None, _context(3, window=(3000, 6000)))
    assert scored.shape == (STEPS,) and np.isfinite(scored[3000:]).any()


def test_the_three_players_do_not_share_a_cache_entry():
    values, usable = _data()
    _tiny().fit(values[:3000], usable[:3000], _context(3))
    _tiny_gru().fit(values[:3000], usable[:3000], _context(3))
    _tiny_tcn().fit(values[:3000], usable[:3000], _context(3))
    cells = sorted(w.cell for w, _ in D._WEIGHTS.values())
    assert cells == ["gru", "lstm", "tcn"], cells


def test_tcn_weights_survive_the_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "WEIGHT_STORE", tmp_path / "_weights")
    values, usable = _data()
    detector = _tiny_tcn()
    detector.fit(values[:3000], usable[:3000], _context(3))
    fitted = detector._weights
    path, = (tmp_path / "_weights").glob("*.npz")
    D.clear_caches()
    reloaded = D._load_weights(path.stem)
    assert reloaded is not None and reloaded[0].cell == "tcn"
    assert reloaded[0].receptive_field == fitted.receptive_field
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(reloaded[0].arrays(), fitted.arrays()))
    assert [b.dilation for b in reloaded[0].blocks] == [1, 2, 4]
    with np.load(path, allow_pickle=False) as blob:
        assert str(blob["cell"]) == "tcn" and "b0_res_w" in blob.files and "b1_res_w" not in blob.files
