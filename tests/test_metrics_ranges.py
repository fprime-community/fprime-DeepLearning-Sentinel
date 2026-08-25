"""Interval algebra, against values computed by hand."""
from __future__ import annotations

import numpy as np

from sentinel_eval.metrics.ranges import (dilate, mask_to_ranges, merge, overlap,
                                          overlaps, ranges_to_mask)


def test_mask_to_ranges_is_half_open():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert mask_to_ranges(mask) == [(1, 3), (5, 6), (7, 10)]


def test_edges():
    assert mask_to_ranges(np.zeros(5, dtype=bool)) == []
    assert mask_to_ranges(np.ones(3, dtype=bool)) == [(0, 3)]
    assert mask_to_ranges(np.array([], dtype=bool)) == []
    assert mask_to_ranges(np.array([1, 0, 1], dtype=bool)) == [(0, 1), (2, 3)]


def test_round_trip():
    mask = np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert np.array_equal(ranges_to_mask(mask_to_ranges(mask), mask.size), mask)


def test_merge_unions_touching_and_overlapping():
    assert merge([(0, 3), (2, 5), (7, 9), (9, 11)]) == [(0, 5), (7, 11)]
    assert merge([(5, 6), (0, 1)]) == [(0, 1), (5, 6)]
    assert merge([(3, 3)]) == []          # empty ranges are dropped


def test_overlap_counts_timesteps():
    assert overlap((0, 5), (3, 8)) == 2
    assert overlap((0, 5), (5, 8)) == 0   # half-open: touching is not overlapping
    assert not overlaps((0, 5), (5, 8))
    assert overlaps((0, 5), (4, 8))


def test_dilate_widens_and_clips():
    assert dilate([(4, 6)], 2, 10) == [(2, 8)]
    assert dilate([(0, 2)], 3, 10) == [(0, 5)]        # clipped at zero
    assert dilate([(8, 10)], 3, 10) == [(5, 10)]      # clipped at n
    assert dilate([(0, 2), (8, 10)], 3, 10) == [(0, 10)]   # merged after widening
