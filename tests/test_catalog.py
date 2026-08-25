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
