"""What the toolkit refuses, and why each refusal exists.

Every check here wraps a failure the code below it already has -- the point is
to catch it at the boundary, name the reason, and exit non-zero, rather than
let a mission read a traceback out of the training loop. Registered at
`docs/MODELS.md` 40.8.
"""
from __future__ import annotations

import numpy as np

from .errors import CalibrationError, ShapeError, TelemetryError
from .limits import FLIGHT_ERROR_WINDOW, FLIGHT_LIMITS


def check_shape(values: np.ndarray) -> None:
    """Two dimensions, at least one channel, and inside the flight maxima."""
    if values.ndim != 2:
        raise TelemetryError(
            f"telemetry is {values.ndim}-dimensional; it must be (timesteps, "
            "channels), one column per channel")
    steps, channels = values.shape
    if channels == 0 or steps == 0:
        raise TelemetryError(f"telemetry is empty at {values.shape}")
    if channels > FLIGHT_LIMITS["MAX_CHANNELS"]:
        raise ShapeError(
            f"{channels} channels; the flight component is compiled for at most "
            f"{FLIGHT_LIMITS['MAX_CHANNELS']} (`flight/include/sentinel/Config.hpp`), "
            "and a model file wider than that is refused at load with TOO_LARGE. "
            "Split the mission across several models, or raise the maximum and "
            "rebuild the component")


def check_finite(values: np.ndarray) -> None:
    """No NaN, no infinity, anywhere.

    A non-finite value entering the recurrence makes every subsequent hidden
    state non-finite, so this cannot be repaired downstream. The fitting path
    intersects its usable mask with `isfinite` and the sampler raises when a
    sampled sequence still carries one; this says so before the fit starts.
    """
    finite = np.isfinite(values)
    if finite.all():
        return
    bad = np.argwhere(~finite)
    steps = sorted({int(i) for i, _ in bad})
    channels = sorted({int(c) for _, c in bad})
    raise TelemetryError(
        f"{len(bad)} non-finite value(s) in the telemetry, on channel(s) "
        f"{channels}, first at timestep {steps[0]}. Healthy telemetry is what "
        "this trains on, so gaps are removed or interpolated before it reaches "
        "here rather than filled silently inside it")


def check_history(values: np.ndarray, window: int, n_predictions: int) -> None:
    """Enough contiguous history to form one training sequence."""
    span = window + n_predictions
    if len(values) < span:
        raise TelemetryError(
            f"{len(values):,} timesteps, and one training sequence needs "
            f"{span:,} (a {window}-step window plus a {n_predictions}-step "
            "horizon). There is nothing to fit on")


def check_calibration_window(n_fit: int, n_holdout: int,
                             span: int = FLIGHT_ERROR_WINDOW) -> None:
    """(!) Each half must outlast the trailing window the statistic uses.

    The fused statistic standardises against its own trailing `span` samples,
    and `span` is **not a field in the model file**: `DerivativeStream` takes it
    from `Config::ERROR_WINDOW` at compile time. A half shorter than `span` is
    not a short measurement of the right thing -- every one of its z-scores is
    computed against a window that never filled, so the cut derived from it
    describes a statistic the flight component does not compute.
    """
    for name, n in (("calibration", n_fit), ("held-out", n_holdout)):
        if n < span:
            raise CalibrationError(
                f"the {name} half is {n:,} timesteps and the trailing window the "
                f"statistic standardises against is {span:,}. Every z-score in it "
                "would be taken against a window that never filled. Supply at "
                f"least {2 * span:,} usable timesteps after warm-up")
