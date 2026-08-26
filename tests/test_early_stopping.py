"""The stopping rule, and the guard that fails when it rejects everything.

telemanom's published `min_delta` is 3e-4, applied as an **absolute** subtraction:
`current < best_loss - min_delta`. Validation MSE on ESA-ADB is ~1e-4, so after
the first epoch the bar became `current < 1.7e-4 - 3e-4 = -1.3e-4` -- negative,
and unreachable by any mean-squared error. Every fit in the project kept its first
epoch's weights and discarded up to 8.8x better ones.

The replacement is a **fraction of the standing best**, which has no units and
cannot exceed the quantity it is compared against whatever the data's scale.
docs/DECISIONS.md states the general form: *dimensionless constants transfer,
absolute ones in data units do not.*
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models.lstm import (SUSPICIOUS_AFTER_EPOCHS, Hyper, TrainingReport,
                                  _refuse_a_stopping_rule_that_rejects_everything)
from sentinel_models.reference import ReferenceError


def _accepts(best: float, current: float, hyper: Hyper) -> bool:
    """The condition as `train` applies it."""
    return current < best * (1.0 - hyper.min_improvement)


# -- the bar itself, against hand-computed values ---------------------------
def test_a_clear_improvement_is_accepted():
    """0.2% better than the standing best, against a 0.1% bar."""
    hyper = Hyper()
    assert _accepts(1.0e-4, 1.0e-4 * 0.998, hyper)


def test_a_negligible_improvement_is_rejected():
    """0.05% better, against a 0.1% bar."""
    hyper = Hyper()
    assert not _accepts(1.0e-4, 1.0e-4 * 0.9995, hyper)


def test_the_bar_sits_exactly_where_the_fraction_puts_it():
    hyper = Hyper(min_improvement=0.01)           # a 1% bar
    assert _accepts(2.0e-4, 1.979e-4, hyper)      # 1.05% better -> accepted
    assert not _accepts(2.0e-4, 1.99e-4, hyper)   # 0.50% better -> rejected


@pytest.mark.parametrize("scale", [1e-2, 1e-4, 1e-6, 1e-8])
def test_the_rule_behaves_identically_at_every_loss_scale(scale):
    """The property the absolute version did not have.

    A relative bar is the same rule whether the loss is 1e-2 or 1e-8. An absolute
    one is a different rule at each scale, and at some scales it is no rule at
    all -- which is what happened.
    """
    hyper = Hyper()
    assert _accepts(scale, scale * 0.99, hyper)         # 1% better -> accepted
    assert not _accepts(scale, scale * 0.9999, hyper)   # 0.01% better -> rejected


def test_the_historical_defect_is_no_longer_reachable():
    """Reconstructed exactly: the published absolute bar against our loss scale.

    `1.7e-4 - 3e-4` is negative, so nothing clears it. The relative bar accepts
    the same improvement the absolute one refused.
    """
    absolute_bar = 1.7e-4 - 3e-4
    assert absolute_bar < 0, "the defect depended on the bar going negative"

    improved = 1.0e-4                       # a real 41% improvement on 1.7e-4
    assert not (improved < absolute_bar), "the published rule refuses this"
    assert _accepts(1.7e-4, improved, Hyper()), "the relative rule accepts it"


# -- the guard --------------------------------------------------------------
def _report(best_epoch: int, epochs: int, history: list[float]) -> TrainingReport:
    return TrainingReport(epochs_run=epochs, best_epoch=best_epoch,
                          best_validation_mse=history[0], history=history)


def test_the_guard_fails_when_training_kept_its_first_epoch():
    """A failure, not a warning.

    Nothing was reading the training reports, which is why the defect survived
    three work items. A line of output nobody looks at is indistinguishable from
    silence, so this raises.
    """
    history = [1.7e-4, 1.2e-4, 0.9e-4, 1.1e-4, 0.8e-4, 1.0e-4, 0.9e-4, 1.2e-4,
               1.0e-4, 1.1e-4, 0.9e-4]
    with pytest.raises(ReferenceError, match="kept its FIRST epoch"):
        _refuse_a_stopping_rule_that_rejects_everything(
            _report(best_epoch=0, epochs=11, history=history), Hyper())


def test_the_guard_names_the_parameter_to_look_at_first():
    history = [1.7e-4] + [1.0e-4] * 10
    with pytest.raises(ReferenceError, match="min_improvement"):
        _refuse_a_stopping_rule_that_rejects_everything(
            _report(best_epoch=0, epochs=11, history=history), Hyper())


def test_the_guard_reports_how_much_was_discarded():
    history = [8.8e-4] + [1.0e-4] * 10        # the worst case actually observed
    with pytest.raises(ReferenceError, match=r"8\.8x better, discarded"):
        _refuse_a_stopping_rule_that_rejects_everything(
            _report(best_epoch=0, epochs=11, history=history), Hyper())


def test_the_guard_is_quiet_when_a_later_epoch_won():
    """The normal case: training improved and kept the improvement."""
    history = [1.7e-4, 1.2e-4, 0.9e-4]
    _refuse_a_stopping_rule_that_rejects_everything(
        _report(best_epoch=2, epochs=3, history=history), Hyper())


def test_the_guard_tolerates_a_genuinely_short_run():
    """Two epochs peaking at the first is not evidence of anything."""
    _refuse_a_stopping_rule_that_rejects_everything(
        _report(best_epoch=0, epochs=2, history=[1.0e-4, 1.1e-4]), Hyper())


def test_the_tripwire_threshold_is_stated_rather_than_buried():
    assert SUSPICIOUS_AFTER_EPOCHS >= 3


# -- end to end -------------------------------------------------------------
def test_a_real_fit_now_improves_past_its_first_epoch(loaded):
    """The defect, at the scale it actually occurred: a fit on real fixture data."""
    from sentinel_models.lstm import train

    values = loaded.values[:6000]
    usable = np.isfinite(values).all(axis=1)
    _, report = train(values, usable,
                      Hyper(window=30, hidden=(16, 16), n_predictions=3,
                            batch_size=32, max_epochs=12, patience=6,
                            sequence_budget_divisor=6, max_validation_sequences=128))
    assert report.best_validation_mse < 1e-2, "loss scale is far below the old 3e-4 bar"
    assert report.best_epoch > 0, (
        f"kept epoch 1 of {report.epochs_run}; the stopping rule is rejecting "
        f"improvements again"
    )
