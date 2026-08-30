"""Do the commands reach the model and change its forecast? Answered offline.

**This tests wiring, not benefit, and the distinction cost a rewrite.** The first
version asserted that supplying commands *reduced* prediction error at commanded
timesteps -- which is a claim that commands **help**, not a claim that they are
plumbed in. It passed while training was broken and flipped sign the moment
training was fixed, which is exactly what a test measuring the wrong thing does.

Whether commands help is what Layer 1 measures on ESA-ADB. Generated data cannot
answer it: the fixture's commanded response is a decaying transient at
semi-regular intervals, and a properly trained forecaster can learn to predict its
continuation from its onset without ever being told a command fired. So the
fixture says nothing about the real question, and asserting a direction on it
would only invite tuning the fixture until the answer came out right -- which had
already happened once.

What generated data *can* settle is whether the input is connected: same weights,
same telemetry, different commands, different forecast. That is deterministic,
cannot flake, and is the thing worth knowing before ninety minutes of refitting.

Recorded for the ablation's interpretation: **with proper training the fixture
shows no benefit from commands** (-0.9 percentage points at commanded timesteps
against elsewhere). Reported rather than hidden, and it predicts nothing about
ESA-ADB, where the responses are real and the model is twenty times larger.
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
    """Both arms, identical in everything but the command input.

    Kept to *record* the fixture's commanded-vs-elsewhere differential, not to
    assert a direction on it -- see the module docstring.
    """
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


def test_the_forecast_changes_when_the_commands_change(commanded_bundle):
    """The wiring, proved directly: same weights, same telemetry, other commands.

    If the command columns were dropped, mis-sliced, or zeroed anywhere between
    `Bundle` and the forward pass, this forecast would not move. Nothing about
    whether the model uses them *well* -- only that it reads them at all.
    """
    D.clear_caches()
    train_hi = STEPS // 2
    context = Context(
        mission="missionX", channels=commanded_bundle.channel_ids,
        groups=commanded_bundle.groups, period_seconds=30, fold=0,
        window=(0, train_hi), commands=commanded_bundle.commands[:train_hi],
        command_ids=commanded_bundle.command_ids)
    usable = np.isfinite(commanded_bundle.values[:train_hi]).all(axis=1)

    detector = _detector(True)
    detector.fit(commanded_bundle.values[:train_hi], usable, context)

    window = slice(0, 4000)
    real = Context(**{**vars(context), "window": (0, 4000),
                      "commands": commanded_bundle.commands[window]})
    with_commands = detector._smoothed_errors(commanded_bundle.values[window], real)

    D.clear_caches()
    detector._impulses = None                      # keep the weights, drop the input
    silent = Context(**{**vars(real),
                        "commands": np.zeros_like(commanded_bundle.commands[window])})
    detector._impulses = silent.commands
    without = detector._smoothed_errors(commanded_bundle.values[window], silent)

    assert not np.allclose(with_commands, without), (
        "the forecast is identical with and without command activity, so the "
        "command columns are not reaching the forward pass"
    )
    D.clear_caches()


def test_the_control_arm_ignores_commands_entirely(commanded_bundle):
    """The other half of the ablation: the control must see exactly what it always saw."""
    assert not _detector(False).wants_commands
    assert _detector(True).wants_commands


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
