"""k/n, never a bare rate."""
from __future__ import annotations

import pytest

from sentinel_eval.metrics.counts import UNDERPOWERED_BELOW, Count, f_beta


def test_rate_and_resolution():
    count = Count(19, 31)
    assert count.rate == pytest.approx(19 / 31)
    assert count.resolution == pytest.approx(1 / 31)


def test_small_denominator_is_flagged():
    assert Count(4, 11).underpowered
    assert "UNDERPOWERED" in Count(4, 11).render()
    assert not Count(4, UNDERPOWERED_BELOW).underpowered


def test_nothing_to_measure_is_undefined_not_zero():
    empty = Count(0, 0)
    assert empty.undefined
    assert empty.rate is None                 # never a silent 0.0
    assert "undefined" in empty.render()


def test_pooling_folds_sums_disjoint_sets():
    pooled = Count(5, 13) + Count(4, 12) + Count(4, 17)
    assert (pooled.k, pooled.n) == (13, 42)


def test_impossible_counts_are_refused():
    with pytest.raises(ValueError):
        Count(5, 3)
    with pytest.raises(ValueError):
        Count(-1, 3)


def test_f_beta_weights_precision_above_recall():
    precise = f_beta(Count(8, 10), Count(4, 10))    # P .8 R .4
    recalled = f_beta(Count(4, 10), Count(8, 10))   # P .4 R .8
    assert precise > recalled
    assert precise == pytest.approx(2 / 3)


def test_f_beta_undefined_when_a_side_is():
    assert f_beta(Count(0, 0), Count(1, 2)) is None
    assert f_beta(Count(0, 2), Count(0, 2)) is None
