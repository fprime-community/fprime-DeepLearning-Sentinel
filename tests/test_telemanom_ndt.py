"""telemanom's detection stack: smoothing, the dynamic threshold, pruning.

The stack is where the method's precision comes from, and precision is what the
gate metric weights twice. Two of these tests exist because the first
implementation got them wrong in ways that produced numbers rather than errors.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models import telemanom as T
from sentinel_models.telemanom import Config

CONFIG = Config()


def _smoothed_noise(n=30000, scale=0.08, seed=0, anomaly=None, size=0.0):
    raw = np.abs(np.random.default_rng(seed).standard_normal((n, 1))) * scale
    if anomaly is not None:
        raw[anomaly[0]:anomaly[1]] += size
    return T.ewma(raw.astype(np.float32), CONFIG.smoothing_window)[:, 0]


# -- smoothing --------------------------------------------------------------
def test_ewma_matches_the_pandas_definition():
    """telemanom smooths with `ewm(span=...).mean()`; this is that, blocked."""
    pd = pytest.importorskip("pandas")
    values = np.random.default_rng(0).standard_normal((5000, 3))
    expected = pd.DataFrame(values).ewm(span=105).mean().values
    assert np.abs(T.ewma(values, 105) - expected).max() < 1e-6


def test_ewma_blocked_equals_ewma_whole():
    """Scoring runs in blocks; the carry has to make that change nothing."""
    values = np.random.default_rng(1).standard_normal((5000, 2))
    whole = T.ewma(values, 105)
    state = T.EwmaState()
    parts = [T.ewma(values[i:i + 700], 105, state) for i in range(0, 5000, 700)]
    assert np.array_equal(np.concatenate(parts), whole)


def test_ewma_corrects_the_opening_bias():
    """An unadjusted average opens by dragging the series toward a zero nobody saw."""
    values = np.full((200, 1), 5.0)
    assert abs(T.ewma(values, 105)[0, 0] - 5.0) < 1e-6


# -- the dynamic threshold --------------------------------------------------
def test_a_quiet_window_yields_no_sequence():
    """Silence is a valid answer. eps falls back to mu + 12 sigma."""
    quiet = _smoothed_noise(n=CONFIG.error_window)[:CONFIG.error_window]
    eps, sequences = T.dynamic_threshold(quiet, CONFIG)
    assert sequences == []
    assert eps > quiet.max()


def test_a_spike_is_isolated():
    window = _smoothed_noise(n=CONFIG.error_window)[:CONFIG.error_window].copy()
    window[1000:1050] += 3.0
    _, sequences = T.dynamic_threshold(window, CONFIG)
    assert sequences, "an obvious excursion produced no sequence"
    assert any(lo <= 1025 < hi for lo, hi in sequences)


def test_a_constant_window_cannot_produce_a_threshold():
    eps, sequences = T.dynamic_threshold(np.full(CONFIG.error_window, 0.5,
                                                 dtype=np.float32), CONFIG)
    assert sequences == [] and np.isfinite(eps)


def test_skipping_unreachable_z_values_changes_nothing():
    """The sweep's early exit must be an optimisation, not a behaviour change."""
    window = _smoothed_noise(n=CONFIG.error_window)[:CONFIG.error_window].copy()
    window[400:460] += 2.0
    mu, sigma = float(window.mean()), float(window.std())

    exhaustive = None
    best = -np.inf
    for z in CONFIG.z_values:
        eps = mu + z * sigma
        above = window >= eps
        if not above.any():
            continue
        remainder = window[~above]
        sequences = T._buffered(above, CONFIG.error_buffer)
        if remainder.size == 0 or not sequences:
            continue
        d_mu = (mu - float(remainder.mean())) / mu
        d_sigma = (sigma - float(remainder.std())) / sigma
        covered = sum(hi - lo for lo, hi in sequences)
        score = (d_mu + d_sigma) / (len(sequences) ** 2 + covered)
        if score >= best:
            best, exhaustive = score, float(eps)
    assert T.dynamic_threshold(window, CONFIG)[0] == pytest.approx(exhaustive)


def test_buffering_by_runs_equals_buffering_by_indices():
    """The dilation is telemanom's index expansion, computed without the indices."""
    mask = np.zeros(500, dtype=bool)
    mask[[10, 11, 200, 400]] = True
    buffer = 20
    naive = set()
    for i in np.flatnonzero(mask):
        for offset in range(-(buffer - 1), buffer):
            if 0 <= i + offset < mask.shape[0]:
                naive.add(int(i + offset))
    covered = set()
    for lo, hi in T._buffered(mask, buffer):
        covered.update(range(lo, hi))
    assert covered == naive


# -- pruning ----------------------------------------------------------------
def test_pruning_keeps_a_sequence_that_stands_clear():
    e_s = np.zeros(1000, dtype=np.float32)
    e_s[100:110] = 10.0
    e_s[500] = 1.0                                   # the largest normal value
    kept = T.prune(e_s, [(100, 110)], eps=5.0, p=0.13)
    assert kept == [True]


def test_pruning_drops_a_sequence_that_does_not():
    e_s = np.zeros(1000, dtype=np.float32)
    e_s[100:110] = 10.0
    e_s[500] = 9.5                                   # only 5% below the peak
    assert T.prune(e_s, [(100, 110)], eps=9.6, p=0.13) == [False]


def test_pruning_keeps_the_prefix_above_the_last_large_drop():
    """Sequences separated from the rest survive; the tail continuous with it does not.

    Peaks of 10.0 and 9.6 sit far above everything else, so both are kept even
    though they are close to *each other* -- the separation that matters is from
    what lies beneath. Peaks of 3.0 and 2.9 run continuously down into the
    largest normal error at 2.55, so neither is separated from nominal and both
    go. That reset-on-a-large-drop is telemanom's, and it is why the method
    reports the events that stand out rather than every event that cleared a line.
    """
    e_s = np.zeros(2000, dtype=np.float32)
    for i, peak in enumerate((10.0, 9.6, 3.0, 2.9)):
        e_s[100 * (i + 1):100 * (i + 1) + 5] = peak
    e_s[1500] = 2.55                                 # below eps: the normal maximum
    sequences = [(100, 105), (200, 205), (300, 305), (400, 405)]
    assert T.prune(e_s, sequences, eps=2.6, p=0.13) == [True, True, False, False]


# -- the property that nearly went missing ----------------------------------
@pytest.mark.parametrize("length", [100, 500, 2100, 8828])
def test_long_anomalies_are_detected_at_every_length(length):
    """The reason the window overlap is telemanom's and not ours.

    A threshold derived from the moments of the window it is judging cannot see
    an anomaly that fills that window. Stepping a whole window at a time missed
    everything above ~500 timesteps, and `m1-g8.9.10`'s headline-cell events have
    a median footprint of 1,951 and a 75th percentile of 8,828. If this test ever
    fails, the detector is blind to the primary recall set.
    """
    e_s = _smoothed_noise(anomaly=(12000, 12000 + length), size=2.0)
    fired = T.channel_ratios(e_s, CONFIG) >= 1.0
    assert fired[12000:12000 + length].any()


@pytest.mark.parametrize("seed", range(6))
def test_nominal_data_raises_no_alarm_at_all(seed):
    """Carpet-bombing is the failure mode the gate metric exists to catch.

    Several seeds rather than one: a threshold that is quiet on a lucky draw and
    noisy on the next is not quiet, and the trivial baseline already showed how
    convincing a bad detector looks when only recall is examined.
    """
    assert not (T.channel_ratios(_smoothed_noise(seed=seed), CONFIG) >= 1.0).any()


def test_a_detection_costs_few_claims():
    """Precision is weighted twice; a detector that fragments its answer loses."""
    from sentinel_eval.metrics.ranges import mask_to_ranges

    e_s = _smoothed_noise(anomaly=(12000, 14100), size=2.0)
    ranges = mask_to_ranges(T.channel_ratios(e_s, CONFIG) >= 1.0)
    assert 0 < len(ranges) <= 3, f"{len(ranges)} separate alarms for one event"


def test_suppressed_values_can_never_reach_the_operating_point():
    """Pruned sequences must rank, but must not alarm."""
    e_s = _smoothed_noise(anomaly=(12000, 12100), size=2.0)
    scored = T.channel_ratios(e_s, CONFIG)
    assert scored.min() >= 0.0
    assert ((scored < 1.0) | (scored >= 1.0)).all()      # no NaN anywhere
    assert np.isfinite(scored).all()


def test_the_threshold_trails_and_never_reads_past_its_own_segment():
    """The bound on what a score at step t can depend on.

    `sentinel_models.baselines` states the rule -- a detector that peeks at
    future samples is not one that can fly, and `rstd`, the number this has to
    beat, is strictly trailing. What remains is a fixed `stride`-step batching
    latency: a threshold is chosen from history and applied to the next segment,
    so the segment is scored together. That delays detection; it does not
    improve it.
    """
    e_s = _smoothed_noise(seed=0)
    disturbed = e_s.copy()
    disturbed[20000:] += 5.0
    before = T.channel_ratios(e_s, CONFIG)
    after = T.channel_ratios(disturbed, CONFIG)
    assert np.array_equal(before[:20000 - CONFIG.stride], after[:20000 - CONFIG.stride])


def test_a_point_anomaly_one_timestep_long_is_still_reachable():
    """Ten of ESA-ADB's eleven point anomalies are shorter than one grid period."""
    e_s = _smoothed_noise(anomaly=(12000, 12001), size=2.0)
    assert (T.channel_ratios(e_s, CONFIG) >= 1.0).any()


def test_combine_reduces_channels_and_names_the_one_responsible():
    smoothed = np.stack([_smoothed_noise(seed=1),
                         _smoothed_noise(seed=2, anomaly=(12000, 12500), size=2.0)],
                        axis=1)
    combined, who = T.combine(smoothed, CONFIG)
    assert combined.shape == (smoothed.shape[0],)
    assert who.dtype == np.int8
    fired = combined >= 1.0
    assert fired[12000:12500].any()
    assert (who[12000:12500][fired[12000:12500]] == 1).all()


def test_the_z_sweep_is_configuration_not_a_module_constant():
    """It has to be fitted per model, so it cannot live as a global.

    z counts standard deviations and is immune to rescaling -- but not to a
    change in the shape of the error distribution. A fortyfold better forecast
    turned the same range from 182 alarm ranges into 3,548 (docs/DECISIONS.md).
    """
    published = Config()
    assert published.z_floor == 2.5 and published.z_ceiling == 12.0
    assert published.z_values[0] == 2.5

    raised = Config(z_floor=8.0, z_ceiling=20.0, z_step=1.0)
    assert raised.z_values[0] == 8.0 and raised.z_values[-1] == 19.0
    assert raised.as_dict()["z_floor"] == 8.0


def test_raising_the_floor_makes_the_detector_quieter():
    """The knob has to move the thing it is being swept for."""
    e_s = _smoothed_noise(anomaly=(12000, 12600), size=2.0)
    fired = {}
    for floor in (2.5, 8.0, 20.0):
        cfg = Config(z_floor=floor, z_ceiling=max(12.0, floor + 8.0))
        fired[floor] = int((T.channel_ratios(e_s, cfg) >= 1.0).sum())
    assert fired[2.5] >= fired[8.0] >= fired[20.0], fired


def test_the_silence_fallback_follows_the_ceiling():
    """If no z qualifies, eps sits at the ceiling -- which must move with it."""
    quiet = _smoothed_noise(n=CONFIG.error_window)[:CONFIG.error_window]
    low = T.dynamic_threshold(quiet, Config(z_ceiling=12.0))[0]
    high = T.dynamic_threshold(quiet, Config(z_ceiling=30.0))[0]
    assert high > low
