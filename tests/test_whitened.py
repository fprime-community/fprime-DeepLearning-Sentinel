"""The relationship test, on generated data with a known relationship.

docs/DECISIONS.md D23, docs/MODELS.md sections 11 and 12. These pin the property
the stage exists for -- that an in-pattern excursion and an off-pattern one of the
SAME per-channel magnitude are scored differently -- rather than pinning numbers,
which pass for the wrong reasons the moment the data moves.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models import telemanom, whiten
from sentinel_models.reference import ReferenceError


def _related(n=120_000, seed=0):
    """Three channels: two that move together, one independent.

    Channel 1 follows channel 0 at 0.8. That is the relationship; breaking it is
    what this stage is supposed to notice and what a per-channel rule cannot.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return np.stack([a,
                     0.8 * a + rng.normal(scale=0.2, size=n),
                     rng.normal(size=n)], axis=1).astype(np.float32)


def _fitted(residual, config=None):
    config = config or whiten.Config()
    acc = whiten.Accumulator(residual.shape[1], config.sample_stride)
    for lo in range(0, residual.shape[0], 25_000):
        acc.add(residual[lo:lo + 25_000])
    return config, acc.finish(config)


# -- the property the stage exists for --------------------------------------
def test_a_broken_link_outscores_an_in_pattern_move_of_the_same_size():
    """The whole case for the stage, as an assertion.

    Both injections add exactly 3.0 to the residual, so a per-channel rule sees
    the same number twice. One respects the learned relationship (channel 1
    follows channel 0 at 0.8); the other breaks it. Whitening must separate them.
    """
    residual = _related()
    config, w = _fitted(residual)

    in_pattern = residual.copy()
    in_pattern[1_000:1_100] += np.array([3.0, 2.4, 0.0], dtype=np.float32)

    broken = residual.copy()
    broken[1_000:1_100] += np.array([0.0, 3.0, 0.0], dtype=np.float32)

    peak = lambda x: whiten.whitened_length(x, w.mean, w.scale, w.precision)[1_000:1_100].max()
    # Measured 1.83x on this fixture. The theoretical separation is far larger;
    # what dilutes it is the ordinary residual noise present at those timesteps,
    # which is honest -- a real excursion never arrives alone.
    assert peak(broken) > 1.5 * peak(in_pattern), (
        f"whitening did not separate a broken link ({peak(broken):.2f}) from an "
        f"in-pattern excursion of the same per-channel size ({peak(in_pattern):.2f})"
    )


def test_the_per_channel_view_cannot_separate_them():
    """The control for the test above: this is what the current layer sees."""
    residual = _related()
    in_pattern = residual.copy()
    in_pattern[1_000:1_100] += np.array([3.0, 2.4, 0.0], dtype=np.float32)
    broken = residual.copy()
    broken[1_000:1_100] += np.array([0.0, 3.0, 0.0], dtype=np.float32)
    biggest = lambda x: np.abs(x[1_000:1_100]).max()
    assert abs(biggest(broken) - biggest(in_pattern)) < 0.6 * biggest(broken), (
        "the per-channel magnitudes were not comparable, so the contrast above "
        "does not isolate the relationship"
    )


def test_nominal_data_admits_about_what_it_was_asked_for():
    residual = _related()
    for rate in (0.01, 0.001):
        config, w = _fitted(residual, whiten.Config(admission_rate=rate))
        d = whiten.whitened_length(residual, w.mean, w.scale, w.precision)
        admitted = float(np.mean(d >= w.threshold))
        assert abs(admitted - rate) <= 0.4 * rate, (
            f"asked for {rate}, admitted {admitted}"
        )


def test_the_score_is_one_series_not_one_per_channel():
    residual = _related()
    config, w = _fitted(residual)
    out, who = whiten.ratios(residual, config, w)
    assert out.ndim == 1 and out.shape[0] == residual.shape[0]
    assert who.shape[0] == residual.shape[0]


def test_attribution_names_the_channel_carrying_the_whitened_length():
    residual = _related()
    config, w = _fitted(residual)
    broken = residual.copy()
    broken[1_000:1_100, 1] += 3.0
    who = whiten.contributions(broken, w.mean, w.scale, w.precision)
    assert np.bincount(who[1_000:1_100], minlength=3).argmax() in (0, 1), (
        "attribution pointed at the independent channel, which took no injection"
    )


# -- conditioning and refusals ----------------------------------------------
def test_a_constant_channel_does_not_make_the_covariance_uninvertible():
    residual = _related()
    residual[:, 2] = 0.5                       # a channel that never moves
    config, w = _fitted(residual)
    assert np.all(np.isfinite(w.precision))
    assert np.isfinite(w.threshold)


def test_two_near_identical_channels_stay_conditioned():
    """Shrinkage toward the identity handles a duplicate pair; it did not need to
    raise. Recorded as the measured behaviour rather than the expected one -- the
    first version of this test asserted a refusal and the code was right."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=60_000)
    residual = np.stack([a, a + rng.normal(scale=1e-6, size=60_000)], 1).astype(np.float32)
    config, w = _fitted(residual)
    assert w.condition_number < whiten.MAX_CONDITION
    assert np.all(np.isfinite(w.precision)) and np.isfinite(w.threshold)


def test_an_ill_conditioned_correlation_is_refused_rather_than_amplified():
    """With the ridge removed, a duplicate pair is singular and must not score.

    The guard exists for a broken shrinkage path or a degenerate pool, not for
    the ordinary duplicate case above -- an inverse of a near-singular matrix
    produces large numbers that look exactly like detections.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(size=60_000)
    residual = np.stack([a, a + rng.normal(scale=1e-9, size=60_000)], 1).astype(np.float32)
    with pytest.raises(ReferenceError, match="ill-conditioned"):
        _fitted(residual, whiten.Config(shrinkage=0.0))


def test_channels_of_wildly_different_variance_stay_conditioned():
    """The defect that shipped: shrinking toward diag(S) leaves scale disparity.

    ESA min-max scales within channel groups, so residual variances differ by
    orders of magnitude. The first version inverted the covariance and reached a
    condition number of 4.3e8; standardising first makes the inverted matrix a
    correlation and the disparity cannot reach it.
    """
    rng = np.random.default_rng(2)
    a = rng.normal(size=80_000)
    residual = np.stack([a, 1e-5 * rng.normal(size=80_000),
                         1e3 * rng.normal(size=80_000)], 1).astype(np.float32)
    config, w = _fitted(residual)
    assert w.condition_number < whiten.MAX_CONDITION
    assert np.isfinite(w.threshold)


def test_a_whitening_that_is_not_whitening_refuses_to_score():
    """The sqrt(C) check, as an assertion rather than as arithmetic nobody did.

    The first run of this stage produced thresholds of 25.7 to 36.3 against
    sqrt(12) = 3.5 and scored anyway, silently, for a whole run.
    """
    residual = _related()
    config, w = _fitted(residual)
    assert abs(w.median_length - np.sqrt(residual.shape[1])) < 1.0, (
        f"median whitened length {w.median_length} is not near "
        f"sqrt({residual.shape[1]})"
    )


def test_the_calibration_pool_can_be_masked_and_the_mask_moves_the_threshold():
    """The contamination defect, pinned. docs/NARRATIVE.md section 6."""
    residual = _related()
    dirty = residual.copy()
    dirty[50_000:50_400] += 40.0                      # an anomaly inside the window
    usable = np.ones(dirty.shape[0], dtype=bool)
    usable[50_000:50_400] = False

    config = whiten.Config()
    contaminated = whiten.Accumulator(dirty.shape[1], config.sample_stride)
    cleaned = whiten.Accumulator(dirty.shape[1], config.sample_stride)
    for lo in range(0, dirty.shape[0], 25_000):
        block = slice(lo, lo + 25_000)
        contaminated.add(dirty[block])
        cleaned.add(dirty[block], usable[block])
    assert cleaned.finish(config).threshold < contaminated.finish(config).threshold, (
        "masking the anomalies out of the calibration pool did not lower the "
        "threshold, so the mask is not reaching the quantile"
    )


def test_shrinkage_is_recorded_rather_than_hidden():
    assert whiten.Config().as_dict()["shrinkage"] == whiten.SHRINKAGE


def test_a_covariance_needs_samples():
    with pytest.raises(ReferenceError, match="at least two"):
        whiten.Accumulator(3, 37).finish(whiten.Config())


def test_ratios_refuses_a_single_channel_series():
    config, w = _fitted(_related())
    with pytest.raises(ReferenceError, match="steps, channels"):
        whiten.ratios(np.zeros(100, dtype=np.float32), config, w)


def test_alarm_shaping_is_telemanoms_so_the_comparison_isolates_the_rule():
    """error_buffer and pruning are held identical -- docs/DECISIONS.md D21."""
    config = whiten.Config()
    assert config.error_buffer == telemanom.ERROR_BUFFER
    assert config.pruning_p == telemanom.PRUNING_P
    assert config.smoothing_window == telemanom.Config().smoothing_window
    assert config.error_window == telemanom.Config().error_window
