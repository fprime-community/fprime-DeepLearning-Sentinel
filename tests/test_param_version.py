"""`param_version` is checked, and the check is what makes D68 safe.

Two PARAMS generations exist with an **identical byte layout**, so
`format_version` stays 1 and D30's freeze is untouched. What differs is which
statistic `threshold` cuts: version 1 the EWMA of the absolute residual (D25),
version 2 the fused `max(z_residual, z_derivative)` (D65, adopted by D68). The
two are on different scales, so a reader that accepted any value would apply one
cut to the other statistic and say nothing -- the silent drift
`docs/MODEL_FILE.md` 1 forbids.

Both sides of the format are tested here, because a refusal that exists only in
C++ is a refusal the toolkit can write a file straight past.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import importlib.util
import sys

from sentinel_export import format as fmt
from sentinel_export.reader import read_model
from sentinel_export.writer import write_model

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "flight" / "test" / "vectors"


def _load(name, rel):
    spec_ = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec_)
    sys.modules[name] = module
    spec_.loader.exec_module(module)
    return module


GV = _load("make_golden_vectors", "scripts/make_golden_vectors.py")


def spec(param_version: int) -> dict:
    """A small valid model at a given PARAMS generation.

    Built through `scripts/make_golden_vectors.py`'s own helpers rather than by
    hand: the writer checks the array names and their order, and a second copy of
    that list here would be a second thing to keep right.
    """
    weights = GV.seeded_weights(2, [4], 2, seed=68)
    out = GV.to_spec(weights, 1.5, 12, "test")
    out["params"]["param_version"] = param_version
    return out


@pytest.mark.parametrize("version", fmt.SUPPORTED_PARAM_VERSIONS)
def test_a_known_param_version_round_trips(version) -> None:
    status, model = read_model(write_model(spec(version)))
    assert status is fmt.Status.OK, status
    assert model["params"]["param_version"] == version


@pytest.mark.parametrize("version", [0, 3, 255, 4096])
def test_an_unknown_param_version_is_refused(version) -> None:
    """Refused by name, not read and hoped about."""
    status, model = read_model(write_model(spec(version)))
    assert status is fmt.Status.BAD_PARAM_VERSION, status
    assert model is None


def test_the_two_generations_are_byte_identical_in_layout() -> None:
    """The whole argument for keeping `format_version` at 1 rests on this."""
    one, two = write_model(spec(1)), write_model(spec(2))
    assert len(one) == len(two), "a version bump must not change the file's length"
    differing = [i for i, (a, b) in enumerate(zip(one, two)) if a != b]
    # param_version itself, and the two CRCs that cover it.
    assert len(differing) <= 10, f"more than the version and its CRCs moved: {differing}"


def test_the_status_codes_agree_across_the_two_implementations() -> None:
    """A refusal that exists only in C++ is one the toolkit writes straight past."""
    cpp = (ROOT / "flight" / "include" / "sentinel" / "Status.hpp").read_text()
    assert "BAD_PARAM_VERSION = 12U" in cpp
    assert int(fmt.Status.BAD_PARAM_VERSION) == 12


@pytest.mark.skipif(not (VECTORS / "p1.bin").exists(), reason="tier p1 not generated")
def test_the_committed_flight_model_is_version_two() -> None:
    status, model = read_model((VECTORS / "p1.bin").read_bytes())
    assert status is fmt.Status.OK, status
    assert model["params"]["param_version"] == fmt.PARAM_VERSION_FUSED
    assert model["params"]["warmup_steps"] == 2350, (
        "the warm-up must outlast the 2,100 the fused statistic's windows need")


@pytest.mark.skipif(not (VECTORS / "p1.pvec").exists(), reason="tier p1 not generated")
def test_the_flight_tier_emits_often_enough_to_pin_a_flag() -> None:
    blob = (VECTORS / "p1.pvec").read_bytes()
    assert blob[:4] == b"SNTP"
    channels, steps = struct.unpack_from("<HI", blob, 6)[0], struct.unpack_from("<I", blob, 8)[0]
    record = channels * 4 + 10
    emitted = sum(blob[20 + t * record + record - 1] for t in range(steps))
    assert emitted > 20, f"only {emitted} emissions: the flag assertion is near-vacuous"
