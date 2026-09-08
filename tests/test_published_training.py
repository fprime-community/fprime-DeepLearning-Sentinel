"""The published training configuration is additive, and it cannot move a banked fit.

`docs/MODELS.md` 28 adds telemanom's own training settings as fields on
:class:`~sentinel_models.lstm.Hyper`. Every one defaults to today's behaviour and
is emitted into the weight-cache key **only when it is not the default** -- D14's
rule, and the way `cell` was added for the GRU. T5 is the structural stop these
tests are: if a banked ESA-ADB fit could be keyed differently after this change,
the change is reverted before anything runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_models import telemanom
from sentinel_models.reference import ReferenceError
from sentinel_models.lstm import Hyper
from sentinel_models.windows import aggregate_predictions

PUBLISHED = dict(train_batch_size=64, min_delta_absolute=0.0003,
                 validation_split="shuffled", full_epochs=True,
                 restore_best=False)


def test_the_new_fields_are_absent_from_the_key_at_their_defaults():
    """T5. An optional field at its null value must leave the hash unmoved."""
    keys = set(Hyper().as_dict())
    for field in ("train_batch_size", "min_delta_absolute", "validation_split",
                  "full_epochs"):
        assert field not in keys, f"{field} is emitted at its default; D14 forbids it"


def test_the_default_hyper_is_byte_identical_to_the_pre_change_dict():
    """The banked LSTM fits were keyed before any of this existed."""
    assert Hyper().as_dict() == {
        "window": 250, "hidden": [80, 80], "dropout": 0.3, "n_predictions": 10,
        "batch_size": 70, "max_epochs": 35, "patience": 10,
        "min_improvement": 0.001, "validation_fraction": 0.2,
        "learning_rate": 0.001, "seed": 0, "sequence_budget_divisor": 180,
        "max_validation_sequences": 2000, "restore_best": True,
    }


@pytest.mark.parametrize("field,value", sorted(PUBLISHED.items()))
def test_each_published_setting_changes_the_key(field, value):
    """The other half of the rule: a differently-trained model may never be served
    weights fitted under a different configuration."""
    assert Hyper(**{field: value}).as_dict_key() != Hyper().as_dict_key()


def test_the_published_configuration_is_expressible_in_one_hyper():
    h = Hyper(**PUBLISHED)
    assert h.train_batch_size == 64 and h.min_delta_absolute == 0.0003
    assert h.validation_split == "shuffled" and h.full_epochs and not h.restore_best
    assert "train_batch_size" in h.as_dict() and "full_epochs" in h.as_dict()


def test_an_unknown_validation_split_is_refused():
    with pytest.raises(ReferenceError, match="validation_split"):
        Hyper(validation_split="random")


def test_the_config_emits_the_aggregation_only_when_it_is_not_the_default():
    assert "aggregate" not in telemanom.Config().as_dict()
    assert telemanom.Config(aggregate="first").as_dict()["aggregate"] == "first"


def test_first_aggregation_is_the_one_step_ahead_prediction():
    """`modeling.py:133-136` -- the diagonal's first element, shifted by one step.

    Built so mean and first cannot agree by accident: prediction `j` steps ahead
    carries the value `j`, so the mean is the average of what is available and
    the first is always the one-step-ahead value.
    """
    steps, n_pred = 6, 4
    predictions = np.zeros((steps, n_pred, 1), dtype=np.float32)
    for s in range(steps):
        for j in range(n_pred):
            predictions[s, j, 0] = j
    first = aggregate_predictions(predictions, "first")
    mean = aggregate_predictions(predictions, "mean")
    assert np.allclose(first, 0.0), "first must take the j=0 column everywhere"
    assert mean[-1, 0] > 0.0, "the mean must mix the further-ahead predictions"
    assert not np.allclose(first, mean)


def test_an_unknown_aggregation_is_refused():
    with pytest.raises(ReferenceError, match="aggregation"):
        aggregate_predictions(np.zeros((3, 2, 1), dtype=np.float32), "median")


def test_the_relative_improvement_rule_is_identical_for_a_non_negative_loss():
    """`docs/MODELS.md` 31: the sign-safe form must move no MSE fit.

    `best * (1 - m)` and `best - |best| * m` agree exactly whenever `best >= 0`,
    which every validation MSE in this repository is. They differ only where the
    loss can go negative, which is the Gaussian NLL case the form was changed for.
    """
    m = Hyper().min_improvement
    for best in (1.0, 1e-4, 3.7, 0.0, 1e-9):
        assert best * (1.0 - m) == pytest.approx(best - abs(best) * m, rel=0, abs=1e-18)
    for best in (-1.0, -0.25):
        assert best * (1.0 - m) > best - abs(best) * m, "the old form raises the bar"


def test_the_gaussian_head_doubles_the_output_and_the_point_head_does_not():
    from sentinel_models.lstm import build_model
    point = build_model(1, Hyper(cell="gru"), 0)
    gauss = build_model(1, Hyper(cell="gru", head="gaussian"), 0)
    assert gauss.head.out_features == 2 * point.head.out_features
    x = np.zeros((2, Hyper().window, 1), dtype="float32")
    import torch
    with torch.no_grad():
        assert tuple(point(torch.from_numpy(x)).shape) == (2, 10, 1)
        assert tuple(gauss(torch.from_numpy(x)).shape) == (2, 10, 1, 2)
