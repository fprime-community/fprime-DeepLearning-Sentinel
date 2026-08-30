"""The sharded path, which is the one that breaks naive loaders.

A sharded manifest entry carries ``key: null`` **and** an empty parent
``sha256``: `prepare.py` fills the digest only for single-object channels. So a
loader that reads ``key`` gets None on four real channels, and one that verifies
the parent's checksum verifies nothing at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval.errors import IntegrityError
from sentinel_eval.read import read_channel

SHARDED = "channel_7"


def test_the_raw_manifest_entry_really_is_the_awkward_shape(bucket):
    entry = next(c for c in
                 bucket.manifest["datasets"]["esa-adb"]["missions"]["missionX"]["channels"]
                 if c["channel_id"] == SHARDED)
    assert entry["key"] is None
    assert entry["sha256"] == ""
    assert len(entry["shards"]) == 2


def test_the_catalog_flattens_both_forms_to_objects(catalog):
    sharded = catalog.channel("missionX", SHARDED)
    plain = catalog.channel("missionX", "channel_1")
    assert sharded.sharded and len(sharded.objects) == 2
    assert not plain.sharded and len(plain.objects) == 1
    assert all(o.sha256 for o in sharded.objects)      # per shard, never the parent


def test_shards_concatenate_in_time_order(bucket, catalog):
    channel = catalog.channel("missionX", SHARDED)
    series = read_channel(bucket, channel)
    assert len(series) == channel.rows
    assert np.all(series.t_anon[:-1] <= series.t_anon[1:])


def test_the_series_cannot_be_mistaken_for_utc(bucket, catalog):
    series = read_channel(bucket, catalog.channel("missionX", "channel_1"))
    assert hasattr(series, "t_anon")
    assert not hasattr(series, "timestamp")


def test_a_corrupt_shard_is_rejected(bucket, catalog):
    class Corrupt:
        def get(self, key):
            blob = bytearray(bucket.get(key))
            if key.endswith("part001.parquet"):
                blob[-40] ^= 0xFF
            return bytes(blob)

    with pytest.raises(IntegrityError, match="sha256"):
        read_channel(Corrupt(), catalog.channel("missionX", SHARDED))


def test_a_truncated_object_is_rejected(bucket, catalog):
    class Truncated:
        def get(self, key):
            blob = bucket.get(key)
            return blob[:-10] if "channel_1." in key else blob

    with pytest.raises(IntegrityError):
        read_channel(Truncated(), catalog.channel("missionX", "channel_1"))


def test_categorical_channels_are_visible_but_not_numeric(catalog):
    categorical = catalog.channel("missionX", "channel_8")
    assert categorical.is_categorical and not categorical.numeric
    assert categorical not in catalog.channels("missionX", numeric_only=True)
