"""Manifest resolution -- and no DataFrame attribute access anywhere."""
from __future__ import annotations

import pytest

from sentinel_eval.errors import TaskError


def test_keys_come_only_from_the_manifest(catalog):
    for channel in catalog.channels("missionX"):
        assert channel.keys
        assert all(k.startswith("esa-adb/v1/archive/") for k in channel.keys)


def test_selection_is_by_manifest_attribute_not_naming_convention(catalog):
    group_one = catalog.channels("missionX", groups=[1])
    assert {c.channel_id for c in group_one} == {"channel_1", "channel_2", "channel_3"}
    assert all(c.group == 1 for c in group_one)


def test_target_and_numeric_filters(catalog):
    everything = catalog.channels("missionX")
    numeric = catalog.channels("missionX", numeric_only=True)
    assert len(numeric) == len(everything) - 1        # one categorical channel
    assert all(c.numeric for c in numeric)


def test_unknown_names_fail_loudly(catalog):
    with pytest.raises(TaskError, match="no such channel"):
        catalog.channel("missionX", "channel_999")
    with pytest.raises(TaskError, match="no such mission"):
        catalog.channels("missionZ")
    with pytest.raises(TaskError, match="annotation"):
        catalog.annotation("nonesuch")


def test_channels_are_frozen_dataclasses_not_dataframe_rows(catalog):
    """`frame.sub` silently returns pandas' subtract method; this cannot."""
    channel = catalog.channel("missionX", "channel_1")
    with pytest.raises(AttributeError):
        _ = channel.sub
    with pytest.raises(AttributeError):
        _ = channel.cat
    with pytest.raises(Exception):
        channel.group = 99                            # frozen


def test_the_prescribed_resampling_rule_is_distinguished_from_a_derived_one(catalog):
    period, kind = catalog.resample_period("missionX")
    assert kind == "prescribed"
    assert int(period / __import__("numpy").timedelta64(1, "s")) == 30


def test_provenance_pins_the_dataset_revision(catalog):
    provenance = catalog.provenance()
    assert provenance["manifest_schema_version"] == "1.1"
    assert provenance["manifest_generated_utc"]
    assert provenance["license"]


def test_the_held_back_sets_are_registered_and_unpaired(catalog):
    """Nominated before tuning (docs/MODELS.md section 5); run once, at the end.

    Neither is paired with anything: pairing exists so a demoted set cannot be
    quietly dropped, and a held-back set has the opposite requirement -- it must
    not be dragged into a run by accident.
    """
    from sentinel_eval import tasks

    for held in ("m1-g3", "m2-ss1"):
        task = tasks.get(held)
        assert tasks.paired_with(held) == (held,), f"{held} would be run alongside others"
        assert "HELD BACK" in task.headline or "HELD BACK" in task.note

    g3 = tasks.get("m1-g3")
    assert g3.selection.groups == (3,) and g3.selection.mission == "mission1"
    assert g3.scores_recall


def test_registering_the_held_back_set_left_the_others_alone(catalog):
    """Additive only: the sets that already produced numbers are untouched."""
    from sentinel_eval import tasks

    primary = tasks.get("m1-g8.9.10")
    assert primary.selection.groups == (8, 9, 10)
    assert primary.split == "forward_chaining"
    assert primary.split_kwargs == {"seed_fraction": 0.25, "folds": 3}
    assert primary.persistence == 1
    assert tasks.paired_with("m1-g8.9.10") == ("m1-g8.9.10", "m1-ss5")
    assert tasks.get("m1-ss5").selection.subsystem == "subsystem_5"
