"""Failures the harness raises deliberately, rather than producing a number.

A harness that returns a plausible-looking score from a broken setup is worse
than one that stops, because the number is indistinguishable from a real one and
outlives the run that produced it. Every class here marks a condition where
continuing would be exactly that.
"""
from __future__ import annotations


class HarnessError(RuntimeError):
    """Base for every deliberate refusal."""


class IntegrityError(HarnessError):
    """An object did not match the checksum the manifest carries for it."""


class OpsTripwire(HarnessError):
    """One run spent enough operations to suggest the script is shaped wrongly.

    Not the billing limit -- see :class:`OpsCeilingExceeded` for that. This is
    the "stop and fetch a human" signal from docs/DATA.md, and a human may
    acknowledge it and continue.
    """


class OpsCeilingExceeded(HarnessError):
    """The calendar month's hard operations ceiling. No override exists."""


class SplitTooThin(HarnessError):
    """The test side of a split holds too few labelled events to score.

    Raised rather than returning a confident number over three anomalies.
    """


class CoverageError(HarnessError):
    """A channel does not cover the requested time range."""


class TaskError(HarnessError):
    """A task definition cannot be resolved against the manifest."""


class DetectorError(HarnessError):
    """A detector returned something the harness cannot score."""


class NormalisationError(HarnessError):
    """A per-channel rescaling was attempted.

    Objective.md 14.8 is RESOLVED as identity: ESA min-max normalised within each
    channel group, so a per-channel scaler erases the amplitude ratios between
    related channels -- precisely the information this project exists to detect.
    """
