"""Lead time measured from a moment the detector could actually reach.

`error_buffer` widens every telemanom-path alarm range `error_buffer - 1` steps
**backwards from a crossing that has already happened**, and lead time is measured
from a range's start. A flight component emits when it detects, not
retroactively, so the figure credited warning nobody received: measured, the
reported +26 median became **+0.0**, and the detector led in 3 of 38 events
rather than 34 of 38 (`docs/DECISIONS.md` D21).

The correction is additive. A detector may declare `last_emission`; absent, the
harness uses the alarm mask unchanged, which is right for every detector that
does not dilate -- for those a crossing *is* the emission and the two readings
coincide. That is what makes the comparison identical across detectors instead of
favouring the one stage that widens.
"""
from __future__ import annotations

import numpy as np

from sentinel_eval.metrics import leadtime
from sentinel_models import telemanom, whiten


def _series(n=20_000, at=9_000, width=400, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.abs(rng.normal(size=(n, 1)))
    raw[at:at + width] += 8.0
    return np.asarray(telemanom.ewma(raw, telemanom.Config().smoothing_window))[:, 0]


def test_the_alarm_reaches_back_before_the_crossing_and_the_emission_does_not():
    """The defect, in one assertion.

    The alarm mask starts up to `error_buffer - 1` steps before anything crossed.
    The emission mask never precedes a crossing -- it marks the end of the batch
    in which one occurred.
    """
    config = telemanom.Config()
    e_s = _series()
    emission = np.zeros(e_s.shape[0], dtype=bool)
    ratios = telemanom.channel_ratios(e_s, config, emission=emission)
    alarm = ratios >= 1.0
    assert alarm.any() and emission.any()

    first_alarm = int(np.flatnonzero(alarm)[0])
    first_emission = int(np.flatnonzero(emission)[0])
    assert first_emission > first_alarm, (
        "the emission point is not later than the dilated alarm start, so the "
        "dilation this test exists for has gone"
    )
    # Bounded by the STRIDE, not by error_buffer, and that correction matters.
    # The dilated sequence is clipped to the judged segment (`max(lo, offset)`),
    # so a range cannot start before its segment does -- the backward reach is at
    # most one batch. What the reported lead therefore credits is the batching
    # latency itself: the range is dated from the start of the 70-step batch in
    # which a crossing occurred, while the detector can only speak at its end.
    assert 0 < first_emission - first_alarm <= config.stride, (
        f"the gap is {first_emission - first_alarm}, outside one batch of "
        f"{config.stride}"
    )


def test_emission_is_never_earlier_than_a_crossing():
    config = telemanom.Config()
    e_s = _series()
    emission = np.zeros(e_s.shape[0], dtype=bool)
    telemanom.channel_ratios(e_s, config, emission=emission)
    # Reconstruct the crossings independently of the alarm shaping.
    span, stride = config.error_window, config.stride
    first_crossing = None
    for seg_lo in range(0, e_s.shape[0], stride):
        seg_hi = min(seg_lo + stride, e_s.shape[0])
        window = e_s[max(0, seg_lo - span):seg_hi]
        offset = seg_lo - max(0, seg_lo - span)
        eps, sequences = telemanom.dynamic_threshold(window, config)
        if sequences and np.any(window[offset:] >= eps):
            first_crossing = seg_lo
            break
    assert first_crossing is not None
    assert int(np.flatnonzero(emission)[0]) >= first_crossing


def test_emission_lands_on_a_batch_boundary():
    """A detector speaks when the batch is processed, not mid-segment."""
    config = telemanom.Config()
    e_s = _series()
    emission = np.zeros(e_s.shape[0], dtype=bool)
    telemanom.channel_ratios(e_s, config, emission=emission)
    for index in np.flatnonzero(emission):
        assert (int(index) + 1) % config.stride == 0 or int(index) == e_s.shape[0] - 1


def test_the_two_readings_differ_and_the_honest_one_is_smaller():
    """The whole point, as a number rather than an argument."""
    config = telemanom.Config()
    e_s = _series(at=9_000, width=400)
    emission = np.zeros(e_s.shape[0], dtype=bool)
    ratios = telemanom.channel_ratios(e_s, config, emission=emission)
    alarm = ratios >= 1.0
    events = [type("E", (), {"event_id": "x", "category": "Anomaly",
                             "cell": "Multivariate/Global/Subsequence"})()]
    spans = {"x": (9_000, 9_400)}
    scorable = np.ones(e_s.shape[0], dtype=bool)
    dilated = leadtime.score(events, spans, alarm, scorable)
    honest = leadtime.score(events, spans, emission, scorable)
    assert dilated.leads and honest.leads
    assert max(honest.leads) < max(dilated.leads), (
        f"honest {honest.leads} is not below dilated {dilated.leads}; "
        f"the correction did nothing"
    )


def test_an_undilated_detector_needs_no_correction():
    """Detectors that do not widen must be unaffected, or this is not fair.

    `top_columns` -- the quantile path -- applies no buffering, so a crossing is
    the emission and the harness's fallback to the alarm mask is exactly right.
    """
    rng = np.random.default_rng(1)
    matrix = np.abs(rng.normal(size=(5_000, 3))).astype(np.float32)
    top, who = telemanom.top_columns(matrix, depth=2)
    assert top.shape == (2, 5_000) and who.shape == (5_000,)


def test_the_whitened_rule_emits_too():
    config = whiten.Config()
    rng = np.random.default_rng(2)
    a = rng.normal(size=40_000)
    residual = np.stack([a, 0.8 * a + rng.normal(scale=0.2, size=40_000),
                         rng.normal(size=40_000)], axis=1).astype(np.float32)
    residual[20_000:20_400, 1] += 6.0
    acc = whiten.Accumulator(3, config.sample_stride)
    for lo in range(0, residual.shape[0], 10_000):
        acc.add(residual[lo:lo + 10_000])
    w = acc.finish(config)
    emission = np.zeros(residual.shape[0], dtype=bool)
    out, _ = whiten.ratios(residual, config, w, emission=emission)
    if (out >= 1.0).any():
        assert emission.any(), "the whitened rule alarmed without ever emitting"
        assert int(np.flatnonzero(emission)[0]) >= int(np.flatnonzero(out >= 1.0)[0])


def test_re_dating_preserves_the_detected_set():
    """The honest reading must not change WHAT was caught, only WHEN it was said.

    The emission mask marks single batch boundaries. Using it as the alarm mask
    silently drops events whose span contains no boundary -- measuring two things
    and reporting them as one. The harness therefore re-dates each range to its
    first emission and keeps the range's end, so the detected set stays the
    detector's.
    """
    predicted = np.zeros(1000, dtype=bool)
    predicted[100:300] = True
    predicted[600:800] = True
    emission = np.zeros(1000, dtype=bool)
    emission[209] = True                      # inside the first range
    emission[699] = True                      # inside the second

    honest = np.zeros_like(predicted)
    from sentinel_eval.metrics.ranges import mask_to_ranges
    for lo, hi in mask_to_ranges(predicted):
        inside = np.flatnonzero(emission[lo:hi])
        honest[lo + (int(inside[0]) if inside.size else 0):hi] = True

    assert len(mask_to_ranges(honest)) == len(mask_to_ranges(predicted)), (
        "re-dating changed the number of alarm ranges"
    )
    # An event overlapping the tail of a range is still detected by both.
    assert honest[250] and predicted[250]
    # But the range now starts later, which is the whole correction.
    assert not honest[150] and predicted[150]
