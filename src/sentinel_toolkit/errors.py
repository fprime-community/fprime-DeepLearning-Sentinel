"""Refusals the toolkit raises deliberately, rather than producing a model.

The same shape and the same reason as `src/sentinel_eval/errors.py`: a toolkit
that returns a plausible-looking `model.bin` from a broken setup is worse than
one that stops. The CLI catches this class once, prints `REFUSED: ...` to
stderr and exits 2, exactly as `src/sentinel_eval/cli.py:288-292` does.
"""
from __future__ import annotations


class ToolkitError(RuntimeError):
    """A refusal. Every one names what was wrong and what would fix it."""


class TelemetryError(ToolkitError):
    """The telemetry cannot be trained on, or cannot be calibrated on."""


class ShapeError(ToolkitError):
    """The telemetry does not fit the flight component's compile-time maxima."""


class CalibrationError(ToolkitError):
    """The calibration window cannot support the estimator it is asked for."""
