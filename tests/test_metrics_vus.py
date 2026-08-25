"""VUS-PR: the properties that must hold, since exact equivalence is not claimed."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.metrics.vus import (average_precision, soft_labels, tie_groups,
                                       vus_pr)

N = 1_000
TRUTH = [(200, 240), (600, 610)]
BASE_RATE = 50 / N


def _perfect() -> np.ndarray:
    scores = np.zeros(N)
    for lo, hi in TRUTH:
        scores[lo:hi] = 1.0
    return scores


def test_soft_labels_ramp_into_the_event():
    label = soft_labels(TRUTH, N, 5)
    assert label[200] == 1.0
    assert np.all(np.diff(label[195:201]) > 0)      # rises approaching the event
    assert label[194] == 0.0
    assert soft_labels(TRUTH, N, 0).sum() == pytest.approx(50)


def test_a_perfect_detector_reaches_exactly_one():
    volume, detail = vus_pr(_perfect(), TRUTH, N)
    assert volume == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in detail["per_buffer"].values())


def test_a_constant_score_earns_the_base_rate_and_no_more():
    # the regression that motivated tie-awareness: without it, a silent detector
    # inherited whatever merit array order happened to give it
    assert average_precision(np.zeros(N), soft_labels(TRUTH, N, 0)) == pytest.approx(BASE_RATE)


def test_ordering_cannot_flatter_a_constant_score():
    front = np.zeros(N)
    label = np.zeros(N)
    label[:50] = 1.0                                 # anomalies at the very front
    assert average_precision(front, label) == pytest.approx(BASE_RATE)


def test_buffer_rewards_a_detection_that_is_merely_early():
    early = np.zeros(N)
    for lo, _hi in TRUTH:
        early[lo - 8:lo] = 1.0
    assert vus_pr(early, TRUTH, N, buffers=(10,))[0] > vus_pr(early, TRUTH, N, buffers=(0,))[0]


def test_over_covering_cannot_exceed_one():
    wide = np.zeros(N)
    for lo, hi in TRUTH:
        wide[max(0, lo - 10):hi + 10] = 1.0
    assert vus_pr(wide, TRUTH, N)[0] <= 1.0


def test_tie_groups_handle_masked_minus_infinity():
    scores = np.array([3.0, 3.0, 1.0, -np.inf, -np.inf, -np.inf])
    assert tie_groups(scores).tolist() == [1, 2, 5]


def test_no_events_means_undefined_not_zero():
    assert vus_pr(_perfect(), [], N)[0] is None
