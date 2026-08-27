"""The proposed rule, and the guard-cell variant, on generated data.

Pre-registered in docs/MODELS.md section 10; justified in docs/DECISIONS.md D20.
These pin the properties the design claims, not the numbers a run produces --
a test that pins a number passes for the wrong reasons the moment the data moves,
which is how the min_delta defect survived three work items.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.errors import DetectorError  # noqa: F401  (import-path check)
from sentinel_models import oscfar, telemanom
from sentinel_models.reference import ReferenceError


def _nominal(steps=40_000, channels=3, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.normal(scale=scale, size=(steps, channels)))
    return telemanom.ewma(raw, oscfar.Config().smoothing_window)


# -- the calibration is label-free and scale-free ---------------------------
def test_calibration_is_invariant_to_a_rescaling_of_the_residual():
    """The failure D17 records is a scale collapse, so the fit must not care.

    Multiply every residual by 1/40 -- roughly what the training fix did to the
    forecast -- and the local multiplier must not move. `alpha` divides a
    quantile by a quantile, so it is dimensionless by construction; `floor` is in
    the units of the error and moves with it, which is the point of having both.
    """
    config = oscfar.Config()
    big = _nominal(scale=1.0)
    small = big / 40.0
    a, b = oscfar.calibrate(big, config), oscfar.calibrate(small, config)
    assert np.allclose(a.alpha, b.alpha, rtol=1e-6), "alpha is not dimensionless"
    assert np.allclose(a.floor / 40.0, b.floor, rtol=1e-6), "floor did not follow the scale"


def test_calibration_never_reads_beyond_what_it_was_given():
    config = oscfar.Config()
    nominal = _nominal()
    assert oscfar.calibrate(nominal, config).nominal_steps == nominal.shape[0]
    with pytest.raises(ReferenceError, match="steps, channels"):
        oscfar.calibrate(nominal[:, 0], config)


def test_a_calibration_belongs_to_the_fit_that_produced_it():
    """Scoring a different channel count with someone else's calibration raises."""
    config = oscfar.Config()
    calibration = oscfar.calibrate(_nominal(channels=3), config)
    with pytest.raises(ReferenceError, match="belongs to the fit"):
        oscfar.top_ratios(_nominal(channels=4), config, calibration)


# -- what the rule does -----------------------------------------------------
def test_nominal_data_calibrated_on_itself_is_quiet():
    """A detector fitted at the 99.9th percentile must not alarm on its own pool."""
    config = oscfar.Config()
    nominal = _nominal()
    calibration = oscfar.calibrate(nominal, config)
    top, _, _ = oscfar.top_ratios(nominal, config, calibration)
    fired = int((top[0] >= 1.0).sum())
    assert fired / nominal.shape[0] < 0.01, f"{fired} alarming steps on nominal data"


def test_an_injected_excursion_is_detected():
    config = oscfar.Config()
    nominal = _nominal()
    calibration = oscfar.calibrate(nominal, config)
    dirty = np.asarray(nominal).copy()
    dirty[20_000:20_500, 0] += 6.0 * float(np.std(nominal[:, 0]))
    top, who, _ = oscfar.top_ratios(dirty, config, calibration)
    assert (top[0][20_000:20_500] >= 1.0).any(), "a 500-step excursion was missed"
    assert who[20_000:20_500].max() == 0, "attribution named the wrong channel"


def test_the_floor_bounds_the_collapse_that_D17_measured():
    """The whole point: a quieter residual must not drag the threshold with it.

    Scoring residuals forty times smaller than the ones the rule was calibrated
    on must stay silent. `mu + z*sigma` does not: its threshold follows the local
    scale to the floor, which is the 50-to-1,505 alarm explosion D18 measures.
    """
    config = oscfar.Config()
    nominal = _nominal()
    calibration = oscfar.calibrate(nominal, config)
    quiet = np.asarray(nominal) / 40.0
    top, _, _ = oscfar.top_ratios(quiet, config, calibration)
    assert int((top[0] >= 1.0).sum()) == 0, "the floor did not hold under a scale collapse"


def test_the_binding_rate_is_reported_and_is_the_falsification_condition():
    config = oscfar.Config()
    nominal = _nominal()
    calibration = oscfar.calibrate(nominal, config)
    _, local_bound = oscfar.channel_ratios(nominal[:, 0], config,
                                           calibration.alpha[0], calibration.floor[0])
    rate = oscfar.binding_rate(local_bound)
    assert 0.0 <= rate <= 1.0
    # Scored far below its own calibration the local term must lose to the floor,
    # which is the collapse-to-lstm-quantile regime section 10.3 names.
    _, quiet_bound = oscfar.channel_ratios(nominal[:, 0] / 40.0, config,
                                           calibration.alpha[0], calibration.floor[0])
    assert oscfar.binding_rate(quiet_bound) == 0.0


# -- guard cells ------------------------------------------------------------
def test_guard_cells_change_the_telemanom_default_for_nobody_by_default():
    """`guard_segment=False` must be bit-identical to the published behaviour."""
    e_s = np.asarray(_nominal(channels=1))[:, 0]
    plain = telemanom.channel_ratios(e_s, telemanom.Config())
    explicit = telemanom.channel_ratios(e_s, telemanom.Config(guard_segment=False))
    assert np.array_equal(plain, explicit)


def test_guard_cells_stop_an_event_raising_its_own_bar():
    """The CFAR argument, on a sustained excursion that fills its own window.

    Unguarded, the segment under test is inside the window its threshold comes
    from, so a long event inflates the bar it has to clear. Excluding it can only
    lower that bar, so the guarded arm must alarm at least as much on the event.
    """
    config_off = telemanom.Config()
    config_on = telemanom.Config(guard_segment=True)
    rng = np.random.default_rng(3)
    raw = np.abs(rng.normal(size=(40_000, 1)))
    raw[15_000:22_000, 0] += 3.0                      # longer than the 2,100 window
    e_s = np.asarray(telemanom.ewma(raw, config_off.smoothing_window))[:, 0]
    inside = slice(17_500, 22_000)
    off = int((telemanom.channel_ratios(e_s, config_off)[inside] >= 1.0).sum())
    on = int((telemanom.channel_ratios(e_s, config_on)[inside] >= 1.0).sum())
    assert on >= off, f"guard cells lost coverage inside a long event: {on} < {off}"


def test_the_guarded_window_excludes_the_segment_and_the_plain_one_does_not():
    """Pins the fact deviation 6 was corrected for, arithmetically."""
    config = telemanom.Config()
    span, stride = config.error_window, config.stride
    steps = 3 * span
    seg_lo = span
    plain = steps and len(range(max(0, seg_lo - span), min(seg_lo + stride, steps)))
    guarded = len(range(max(0, seg_lo - span), seg_lo))
    assert plain == span + stride == 2170
    assert guarded == span == 2100
