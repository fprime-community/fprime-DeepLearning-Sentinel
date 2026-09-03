"""`baselines._rolling` computes the statistic it says it computes.

The D8 correctness fix of work item 9.5 (D37, `docs/MODELS.md` 21). Before it,
`_rolling` squared float32 values and accumulated their prefix sums in float32,
then promoted to float64 only where `np.concatenate` met a float64 `np.zeros`.
Differencing two large float32 prefix sums to recover a small second moment is
catastrophic cancellation, and the variance floor then clamped the negative
result to zero.

These tests exist so it cannot come back. They pin `_rolling` against
`numpy.nanstd` -- an implementation nobody in this repository wrote -- in the
regime that exposes the defect: data whose mean is large relative to its spread,
which is what real telemetry looks like. A battery bus at 28 V with millivolt
noise, or a radiator at 300 K with 0.1 K noise, is the offset case, not the
exotic one.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models.baselines import EPSILON, _rolling

W = 120


def windowed_nanstd(values, window=W):
    """A third implementation, using nothing this project wrote."""
    x = np.asarray(values, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    for t in range(x.shape[0]):
        out[t] = np.nanstd(x[max(0, t - window + 1):t + 1], axis=0)
    return out


@pytest.mark.parametrize("mean,sigma", [
    (0.0, 1.0),          # the benign regime, where the defect never showed
    (28.0, 0.01),        # a battery bus in volts with millivolt noise
    (300.0, 0.1),        # a radiator in kelvin
    (1000.0, 3.0),       # the tier B3 regime, where the defect was measured
], ids=["unit", "battery", "radiator", "b3"])
def test_rolling_std_matches_numpy_nanstd(mean, sigma):
    """The whole point. Before the fix the b3 row was off by 7.66 on sigma 3.0.

    The tolerance is the project's own 1e-5, relative, and every one of these
    clears it by three orders or better. It is not tighter because the rule is
    stated as `sqrt(S2/n - (S1/n)^2)` and that form has a regime limit even in
    float64 -- see the test below, which pins where.
    """
    rng = np.random.default_rng(4)
    values = (rng.standard_normal((900, 6)) * sigma + mean).astype(np.float32)
    got = _rolling(values, W, "std")[W:]
    want = windowed_nanstd(values)[W:]
    relative = (np.abs(got - want) / np.maximum(want, 1e-300)).max()
    assert relative <= 1e-5, f"|mean|/sigma = {abs(mean) / sigma:g}: {relative:.3e}"


def test_the_naive_two_moment_form_has_a_regime_limit_and_this_is_where():
    """Recorded rather than left to be rediscovered. FINDING, work item 9.5.

    The float64 fix removes eight orders of error and is sufficient for every
    input this project scores: ESA-ADB is min-max scaled within channel groups,
    so `|mean|/sigma` runs about 2 to 6, where the error is 1e-13. It does not
    make `sqrt(S2/n - (S1/n)^2)` unconditionally safe, because that form's
    relative error grows as the SQUARE of `|mean|/sigma` whatever the precision.

    Measured, float64, after the fix:

        |mean|/sigma       relative error
                   6            1.9e-13
               2.8e3            4.6e-08
                 1e6            6.7e-03
                 1e8            1.0        (the statistic is gone)
                 1e9            0.0        (the variance floor clamps, silently)

    The last row is the same silent-zero failure the fix removed, at a higher
    threshold. A channel whose mean is a billion times its noise -- a wide
    counter, say -- would still read as perfectly quiet. The flight reference
    `baseline_reference.spread` shares the limit exactly, because it transcribes
    the same rule.

    This test pins the boundary so it cannot drift unnoticed. Making the form
    unconditionally stable is an algorithm change, not a precision fix, and it
    would move the flight golden vectors; it is reported, not taken here.
    """
    rng = np.random.default_rng(4)
    safe = (rng.standard_normal((600, 4)) * 0.01 + 28.0).astype(np.float32)
    got = _rolling(safe, W, "std")[W:]
    want = windowed_nanstd(safe)[W:]
    assert (np.abs(got - want) / want).max() < 1e-6, "the flown regime must be safe"

    hostile = (rng.standard_normal((600, 4)) * 1.0e-3 + 1.0e6).astype(np.float32)
    lost = _rolling(hostile, W, "std")[W:]
    assert lost.max() == 0.0, (
        "the regime limit has moved; if the form was made stable, this test "
        "should be rewritten to assert the new behaviour rather than deleted")


@pytest.mark.parametrize("mean,sigma", [(0.0, 1.0), (1000.0, 3.0), (1.0e5, 1.0e-3)],
                         ids=["unit", "b3", "extreme"])
def test_rolling_mean_matches_numpy(mean, sigma):
    """`_rolling` serves `mavg` too, and the mean path was affected less, not none."""
    rng = np.random.default_rng(5)
    values = (rng.standard_normal((900, 4)) * sigma + mean).astype(np.float32)
    got = _rolling(values, W, "mean")[W:]
    want = np.array([np.nanmean(values[max(0, t - W + 1):t + 1].astype(np.float64), axis=0)
                     for t in range(values.shape[0])])[W:]
    assert np.abs(got - want).max() <= 1e-9 * max(abs(mean), 1.0)


def test_the_variance_floor_no_longer_fires_on_well_conditioned_data():
    """The floor is real and must stay; what must not happen is it firing because
    of arithmetic. Before the fix this produced 3,975 exact zeros on the fixture
    regime against float64's 1,123."""
    rng = np.random.default_rng(9)
    values = (rng.standard_normal((4000, 12)) * 3.0 + 1000.0).astype(np.float32)
    spread = _rolling(values, W, "std")[W:]
    assert (spread == 0.0).sum() == 0, "the variance floor is still clamping real spread"
    assert spread.min() > 0.0


def test_a_constant_channel_still_reads_exactly_zero():
    """The floor still fires where it should: no spread is no spread."""
    values = np.full((300, 2), 1000.0, dtype=np.float32)
    assert _rolling(values, W, "std").max() == 0.0


def test_the_accumulation_is_float64_not_merely_the_result():
    """The defect was invisible in the return dtype -- it was always float64.

    It lived in `np.cumsum` running at float32 before the promotion. This reads
    the source, because a dtype check on the output cannot see it.
    """
    import inspect
    source = inspect.getsource(_rolling)
    assert "astype(np.float64)" in source, "filled must be promoted before it is squared"
    assert "dtype=np.float64" in source, "np.cumsum must accumulate in float64"


def test_nan_handling_is_unchanged_by_the_fix():
    """A correctness fix that quietly changed NaN semantics would be a scope change."""
    rng = np.random.default_rng(6)
    values = rng.standard_normal((400, 3)).astype(np.float32)
    values[50:80, 1] = np.nan
    got = _rolling(values, W, "std")
    assert np.isfinite(got).all()
    assert EPSILON == 1e-12
