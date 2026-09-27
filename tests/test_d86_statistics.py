"""D86's algebra, pinned before any run (docs/MODELS.md 79).

Each identity here is a claim the pre-registration makes about how the new terms
relate to the flown ones; if any fails, the pre-registration is wrong, not the run.
"""
import numpy as np

from sentinel_toolkit.statistic import derivative, motion, slope_mismatch, zstat

RNG = np.random.default_rng(86)
X = np.cumsum(RNG.normal(size=500))
F = X + RNG.normal(scale=0.3, size=500)          # an imperfect forecast


def test_a2_is_the_first_difference_of_the_signed_residual():
    e = X - F
    assert np.allclose(motion(X, F)[1:], np.abs(np.diff(e)))


def test_a2_is_not_the_residual_a_constant_offset_gives_zero_motion():
    assert np.allclose(motion(X, X - 5.0), 0.0)


def test_a2_equals_the_derivative_only_where_the_forecast_is_flat():
    assert np.allclose(motion(X, np.full_like(X, 3.0)), derivative(X))
    assert not np.allclose(motion(X, F), derivative(X))


def test_a2_is_zero_when_the_forecaster_tracks_the_motion_however_large():
    ramp = np.arange(500) * 7.0
    assert np.allclose(motion(ramp, ramp - 1.0), 0.0)
    assert derivative(ramp)[1:].min() == 7.0


def test_a2_obeys_the_triangle_bounds():
    dx, df, m = derivative(X), derivative(F), motion(X, F)
    assert np.all(m <= dx + df + 1e-12)
    assert np.all(m >= np.abs(dx - df) - 1e-12)


def test_the_expected_rate_form_reduces_exactly_to_the_residual():
    # dx(t) - (f(t) - x(t-1)) == x(t) - f(t): why it is not run.
    lhs = np.diff(X) - (F[1:] - X[:-1])
    assert np.allclose(lhs, X[1:] - F[1:])


def test_a5_is_a_residual_difference_within_one_origin():
    one, two = X + RNG.normal(size=500), X + RNG.normal(size=500)
    x_prev = np.concatenate([[X[0]], X[:-1]])
    want = np.abs((X - two) - (x_prev - one))
    got = slope_mismatch(X, one, two)
    assert np.allclose(got[2:], want[2:])
    assert np.all(got[:2] == 0.0)


def test_a5_is_neither_the_derivative_nor_the_motion():
    one, two = F - 0.1, F + 0.2
    got = slope_mismatch(X, one, two)
    assert not np.allclose(got, derivative(X))
    assert not np.allclose(got, motion(X, F))


def test_every_new_term_is_causal():
    t = 300
    one, two = F + 0.5, F - 0.5
    for fn, args in ((motion, (X, F)), (slope_mismatch, (X, one, two))):
        base = fn(*args)
        moved = [np.array(a, copy=True) for a in args]
        for a in moved:
            a[t + 1:] += 1e3
        after = fn(*moved)
        assert np.array_equal(base[: t + 1], after[: t + 1])
        zb, za = zstat(base, 50), zstat(after, 50)
        assert np.array_equal(zb[: t + 1], za[: t + 1])
