"""Does the model actually use the commands? Answered offline, before the run.

Layer 1's whole claim is that a forecaster told a command was issued can predict
the response, and therefore stops alarming on it. That is a mechanism, and a
mechanism can be checked on generated data where the commanded response is known
by construction -- `sentinel_eval.synthetic` gives every priority-3 command a
decaying signature in the telemetry it touches.

**If commands ON does not reduce prediction error at commanded timesteps here,
the wiring is wrong**, and finding that out costs nothing. Finding it out on
ESA-ADB costs ninety minutes of refitting and seventeen Class B.

Deliberately a *mechanism* test, not a metric one. The direction of an F0.5 on
generated data says nothing about ESA-ADB, and asserting one would invite tuning
against the fixture. What is asserted is only that the input reaches the model
and changes its forecast where it should.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import synthetic, tasks
from sentinel_eval.catalog import Catalog
from sentinel_eval.detector import Context
from sentinel_eval.labels import LabelSet
from sentinel_eval.read import read_annotation
from sentinel_models import detectors as D
from sentinel_models.lstm import Hyper
from sentinel_models.telemanom import Config

STEPS = 30_000


def _detector(commanded: bool) -> D.ForecastDetector:
    build = D.TelemanomCommanded if commanded else D.ForecastDetector
    return build(
        hyper=Hyper(window=60, hidden=(32, 32), n_predictions=3, batch_size=32,
                    max_epochs=12, patience=4, sequence_budget_divisor=3,
                    max_validation_sequences=256, seed=0),
        config=Config(error_window=600, stride=20, smoothing_window=30, error_buffer=20),
        chunks=16, chunk_steps=600)


@pytest.fixture(scope="module")
def commanded_bundle():
    bucket = synthetic.build(seed=0, n=STEPS)
    catalog = Catalog.load(bucket)
    labels = LabelSet.from_table(read_annotation(bucket, catalog, "labels"))
    ids = tasks.get("synthetic").selection.resolve(catalog)
    return bundle_mod.load(bucket, catalog, labels, mission="missionX",
                           channel_ids=ids, telecommands=3)


@pytest.fixture(scope="module")
def arms(commanded_bundle):
    """Both arms, identical in everything but the command input."""
    D.clear_caches()
    train_hi = STEPS // 2
    context = Context(
        mission="missionX", channels=commanded_bundle.channel_ids,
        groups=commanded_bundle.groups, period_seconds=30, fold=0,
        window=(0, train_hi),
        commands=commanded_bundle.commands[:train_hi],
        command_ids=commanded_bundle.command_ids,
    )
    usable = np.isfinite(commanded_bundle.values[:train_hi]).all(axis=1)

    out = {}
    for name, commanded in (("off", False), ("on", True)):
        detector = _detector(commanded)
        detector.fit(commanded_bundle.values[:train_hi], usable, context)
        scoring = Context(**{**vars(context), "window": (0, STEPS),
                             "commands": commanded_bundle.commands})
        errors = detector._smoothed_errors(commanded_bundle.values, scoring)
        out[name] = errors.mean(axis=1)
    D.clear_caches()
    return out, commanded_bundle


def _commanded_mask(bundle) -> np.ndarray:
    """Timesteps within a command's response window."""
    from sentinel_eval.synthetic import COMMAND_SPAN

    fired = bundle.commands.any(axis=1)
    mask = np.zeros(fired.shape[0], dtype=bool)
    for step in np.flatnonzero(fired):
        mask[step:step + COMMAND_SPAN] = True
    return mask


def test_the_commands_reach_the_model(arms):
    errors, bundle = arms
    from sentinel_models.lstm import Hyper  # noqa: F401

    detector = _detector(True)
    assert detector.wants_commands
    assert not _detector(False).wants_commands


def test_knowing_a_command_fired_reduces_the_error_it_causes(arms):
    """The mechanism. Commanded responses are what a blind forecaster cannot predict."""
    errors, bundle = arms
    commanded = _commanded_mask(bundle)
    assert commanded.sum() > 200, "fixture produced too few commanded steps"

    off = float(errors["off"][commanded].mean())
    on = float(errors["on"][commanded].mean())
    assert on < off, (
        f"commands did not reduce error at commanded timesteps: {on:.5f} against "
        f"{off:.5f}. The input is reaching the model but is not being used, or is "
        f"misaligned against the telemetry it explains."
    )


def test_it_helps_more_at_commanded_steps_than_away_from_them(arms):
    """Otherwise the model simply got better, and the commands proved nothing."""
    errors, bundle = arms
    commanded = _commanded_mask(bundle)
    quiet = ~commanded

    gain_commanded = 1.0 - float(errors["on"][commanded].mean()) / float(
        errors["off"][commanded].mean())
    gain_quiet = 1.0 - float(errors["on"][quiet].mean()) / float(
        errors["off"][quiet].mean())
    assert gain_commanded > gain_quiet, (
        f"error fell by {gain_commanded:.1%} at commanded steps and {gain_quiet:.1%} "
        f"elsewhere; the improvement is not attributable to the commands"
    )


def test_the_commanded_model_records_its_exogenous_width(arms, commanded_bundle):
    D.clear_caches()
    detector = _detector(True)
    train_hi = STEPS // 2
    context = Context(
        mission="missionX", channels=commanded_bundle.channel_ids,
        groups=commanded_bundle.groups, period_seconds=30, fold=0,
        window=(0, train_hi), commands=commanded_bundle.commands[:train_hi],
        command_ids=commanded_bundle.command_ids)
    usable = np.isfinite(commanded_bundle.values[:train_hi]).all(axis=1)
    detector.fit(commanded_bundle.values[:train_hi], usable, context)

    # Two features per command: the impulse, and how recently it fired.
    assert detector._weights.n_exogenous == 2 * commanded_bundle.commands.shape[1]
    assert detector._weights.n_channels == commanded_bundle.values.shape[1]
    D.clear_caches()


def test_a_commanded_detector_without_commands_refuses(commanded_bundle):
    """Rather than silently training a model that is missing a third of its input."""
    detector = _detector(True)
    context = Context(mission="missionX", channels=commanded_bundle.channel_ids,
                      groups=commanded_bundle.groups, period_seconds=30, fold=0,
                      window=(0, 5000))
    usable = np.ones(5000, dtype=bool)
    with pytest.raises(Exception, match="telecommands"):
        detector.fit(commanded_bundle.values[:5000], usable, context)
