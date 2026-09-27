"""D68's flown statistic, in one place, moved rather than rewritten.

`max(z_residual, z_derivative)`, each standardised against its own trailing
window. These definitions were `scripts/decision_layer_arms.py:53-62` and
`:224`, which is the arm D65 measured and D68 adopted; they are moved here so
the toolkit and that script share one definition rather than two that agree
today. **Nothing about the arithmetic changed in the move**, and
`tests/test_toolkit.py` pins it against the script's own committed behaviour.

The flight transcription of the same thing is `flight/src/DerivativeStream.cpp`
and `flight/src/TrailingWindow.cpp`, held to a NumPy reference at 1e-05 by
committed vectors.
"""
from __future__ import annotations

import numpy as np


def trailing_stats(x: np.ndarray, span: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and population sd over the last `span` samples, inclusive.

    F64 cumulative sums, and the window is trailing rather than centred: at
    step `t` it has seen `t` and nothing after it. Causal by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    lo = np.maximum(np.arange(n) + 1 - span, 0)
    cnt = np.arange(n) + 1 - lo
    s1 = c1[np.arange(n) + 1] - c1[lo]
    s2 = c2[np.arange(n) + 1] - c2[lo]
    mu = s1 / cnt
    return mu, np.sqrt(np.maximum(s2 / cnt - mu * mu, 0.0))


def zstat(x: np.ndarray, span: int) -> np.ndarray:
    """`(x - mu) / max(sd, 1e-12)` against the series' own trailing window."""
    mu, sd = trailing_stats(x, span)
    return (np.asarray(x, dtype=np.float64) - mu) / np.maximum(sd, 1e-12)


def derivative(x: np.ndarray) -> np.ndarray:
    """`|x[t] - x[t-1]|`, with `x[-1] := x[0]` so step 0 is 0 and not undefined."""
    x = np.asarray(x, dtype=np.float64)
    return np.abs(np.diff(x, prepend=x[0]))


def motion(x: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """D86.A2: `|(x[t] - x[t-1]) - (f[t] - f[t-1])|`, the motion mismatch.

    Algebraically `|e[t] - e[t-1]|` with `e = x - f`, the first difference of the
    SIGNED residual, and `e[-1] := e[0]` as `derivative` does. Causal: `f[t]` is
    built from inputs before `t` (`windows.aggregate_predictions`). Research only
    (D86); nothing flown reads it.
    """
    e = np.asarray(x, dtype=np.float64) - np.asarray(forecast, dtype=np.float64)
    return np.abs(np.diff(e, prepend=e[0]))


def slope_mismatch(x: np.ndarray, one_step: np.ndarray, two_step: np.ndarray) -> np.ndarray:
    """D86.A5: `|dx[t] - (two_step[t] - one_step[t])|`, the one-forecast slope mismatch.

    `one_step[t]` and `two_step[t]` are the forecasts OF `t-1` and OF `t` made at
    the single origin `t-2` (inputs through `t-2`), so their difference is the
    slope the forecaster predicted into `t`. Algebraically
    `|[x[t] - two_step[t]] - [x[t-1] - one_step[t]]|`. Steps 0 and 1 have no
    origin and are 0, as `derivative`'s step 0 is. Research only (D86).
    """
    x = np.asarray(x, dtype=np.float64)
    slope = np.asarray(two_step, dtype=np.float64) - np.asarray(one_step, dtype=np.float64)
    out = np.abs(np.diff(x, prepend=x[0]) - slope)
    out[:2] = 0.0
    return out


def fused_per_channel(smoothed_error: np.ndarray, values: np.ndarray,
                      span: int) -> np.ndarray:
    """`max(z_residual, z_derivative)` for every channel. `(T, C) -> (T, C)`.

    `smoothed_error` is the EWMA'd absolute residual the forecaster produces;
    `values` is the raw telemetry the derivative is taken of.
    """
    smoothed_error = np.asarray(smoothed_error, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if smoothed_error.shape != values.shape:
        raise ValueError(f"residual {smoothed_error.shape} and values "
                         f"{values.shape} are different shapes")
    out = np.empty(smoothed_error.shape, dtype=np.float64)
    for c in range(smoothed_error.shape[1]):
        z_residual = zstat(smoothed_error[:, c], span)
        z_derivative = zstat(derivative(values[:, c]), span)
        out[:, c] = np.maximum(z_residual, z_derivative)
    return out


def reduce_across_channels(per_channel: np.ndarray) -> np.ndarray:
    """The maximum across channels. `(T, C) -> (T,)`.

    One channel diverging from its forecast is a divergence, and which one it
    was is what the warning names -- the same reduction
    `src/sentinel_eval/detector.py:136` makes and the same one
    `flight/src/Detector.cpp` makes on the fused score.
    """
    return np.max(np.asarray(per_channel, dtype=np.float64), axis=1)
