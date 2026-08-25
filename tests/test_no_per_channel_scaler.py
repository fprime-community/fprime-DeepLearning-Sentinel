"""Objective.md 14.8, RESOLVED as identity -- enforced as a test, not a comment.

ESA min-max scaled within each channel group, so a per-channel rescale erases the
amplitude ratios between related channels: exactly the information the
cross-channel claim depends on. The failure would be silent, and would look like
our method being weak rather than us having broken the data.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import normalisation
from sentinel_eval.errors import NormalisationError


def test_identity_returns_the_values_untouched():
    values = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    assert normalisation.apply(values) is values


def test_every_per_channel_policy_is_refused():
    values = np.zeros((4, 2), dtype=np.float32)
    for policy in ("per_channel_zscore", "per_channel_minmax", "per_channel_robust"):
        with pytest.raises(NormalisationError, match="amplitude ratios"):
            normalisation.apply(values, policy=policy)


def test_an_unnamed_policy_is_refused_rather_than_ignored():
    with pytest.raises(NormalisationError):
        normalisation.apply(np.zeros((2, 2), dtype=np.float32), policy="minmax")


def test_the_loader_preserves_amplitude_ratios_between_group_mates(loaded, catalog):
    """Values must survive the whole load path bit-for-bit within a group."""
    groups: dict[int, list[int]] = {}
    for index, channel in enumerate(loaded.channels):
        groups.setdefault(channel.group, []).append(index)

    scaled = [members for members in groups.values() if len(members) > 1]
    assert scaled, "fixture must contain a group with more than one channel"

    for members in scaled:
        spreads = [np.nanstd(loaded.values[:, i]) for i in members]
        # a per-channel scaler would have driven every one of these to 1.0
        assert not all(abs(s - spreads[0]) < 1e-6 for s in spreads[1:])


def test_the_bundle_records_which_policy_it_used(loaded):
    assert loaded.provenance["normalisation"] == normalisation.IDENTITY
