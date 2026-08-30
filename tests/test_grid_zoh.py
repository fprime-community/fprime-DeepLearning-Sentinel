"""Zero-order hold, and the staleness guard that stops it inventing data."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.errors import CoverageError
from sentinel_eval.grid import Grid, difference, resample_zoh
from sentinel_eval.read import Series

PERIOD = np.timedelta64(30, "s")


def _grid(minutes=60):
    return Grid.spanning("2000-01-01",
                         np.datetime64("2000-01-01") + np.timedelta64(minutes, "m"), PERIOD)


def _series(times, values):
    return Series(channel=None, t_anon=np.asarray(times, dtype="datetime64[ns]"),
                  value=np.asarray(values, dtype=np.float32))


def test_slicing_is_half_open():
    grid = _grid()
    assert grid.slice_of(np.datetime64("2000-01-01T00:00"),
                         np.datetime64("2000-01-01T00:02")) == (0, 4)


def test_a_zero_length_point_event_still_claims_one_timestep():
    # ten of ESA-ADB's eleven point anomalies are shorter than one grid period;
    # rasterising them to nothing would silently depress point recall
    grid = _grid()
    lo, hi = grid.slice_of(np.datetime64("2000-01-01T00:05"), np.datetime64("2000-01-01T00:05"))
    assert hi - lo == 1


def test_hold_stops_at_max_hold_rather_than_fabricating_a_flat_line():
    times = np.concatenate([
        np.datetime64("2000-01-01T00:00") + np.arange(10) * PERIOD,
        np.datetime64("2000-01-01T00:25") + np.arange(10) * PERIOD,
    ])
    values, valid = resample_zoh(_series(times, np.arange(20)), _grid())
    assert valid[:10].all()                      # covered by real samples
    assert valid[10:20].all()                    # held, still within max_hold
    assert np.unique(values[10:20]) == pytest.approx(9.0)
    assert not valid[20:50].any()                # the gap: refused
    assert np.isnan(values[20:50]).all()
    assert valid[50:70].all()                    # samples resume


def test_timesteps_before_the_first_sample_are_invalid():
    times = np.datetime64("2000-01-01T00:30") + np.arange(5) * PERIOD
    _values, valid = resample_zoh(_series(times, np.arange(5)), _grid())
    assert not valid[0]


def test_difference_does_not_fabricate_a_first_value():
    out = difference(np.array([1.0, 3.0, 6.0, 10.0], dtype=np.float32))
    assert np.isnan(out[0])
    assert out[1:].tolist() == [2.0, 3.0, 4.0]


def test_an_empty_or_inverted_span_is_refused():
    with pytest.raises(CoverageError):
        Grid.spanning("2000-01-02", "2000-01-01", PERIOD)
