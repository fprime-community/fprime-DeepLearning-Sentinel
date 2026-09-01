"""The `model.bin` format, against `docs/MODEL_FILE.md`, which is normative.

These tests are the Python half of the contract. The C++ half lives in
`flight/test/` and shares the `Status` values, so a file refused here is refused
there for the same reason.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from sentinel_export import ModelFileError, Status, read_model, replace_params, write_model
from sentinel_export import format as fmt

#: The compile-time maxima the flight core budgets for
#: (`flight/include/sentinel/Config.hpp`). Passed to the reader to get the same
#: TOO_LARGE refusal the flight reader gives.
FLIGHT_LIMITS = {"MAX_CHANNELS": 16, "MAX_LAYERS": 2, "MAX_HIDDEN": 80,
                 "MAX_PREDICTIONS": 10, "MAX_INPUTS": 16}

#: The calibrated global cut of `gru-quantile` fold 0 on the gate set, read from
#: runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json. Used because a
#: real fitted threshold is the value the F64 field exists to carry exactly.
THRESHOLD = 0.013237688725814186


def seeded_arrays(n_inputs, hidden, n_predictions, n_channels, seed=0):
    """PyTorch's own GRU initialiser, by the recipe in `docs/MODEL_FILE.md` 10."""
    rng = np.random.default_rng(seed)
    out = []
    previous = n_inputs
    sizes = {}
    for i, h in enumerate(hidden):
        for suffix in fmt.ARRAY_SUFFIXES:
            sizes[f"l{i}_{suffix}"] = h
        previous = h
    sizes["head_w"] = sizes["head_b"] = hidden[-1]
    for name, shape in fmt.array_shapes(n_inputs, hidden, n_predictions, n_channels):
        k = 1.0 / np.sqrt(sizes[name])
        out.append((name, rng.uniform(-k, k, shape).astype(np.float32)))
    return out


def make_spec(n_channels=3, hidden=(4, 4), n_predictions=2, n_exogenous=0, seed=0):
    hidden = list(hidden)
    n_inputs = n_channels + n_exogenous
    return {
        "arch": "gru",
        "n_channels": n_channels,
        "n_exogenous": n_exogenous,
        "window": 250,
        "n_predictions": n_predictions,
        "hidden": hidden,
        "channels": [{"id": 100 + i, "name": f"channel_{i}"} for i in range(n_channels)],
        "arrays": seeded_arrays(n_inputs, hidden, n_predictions, n_channels, seed),
        "params": {
            "ewma_span": 105, "agreement": 1, "persistence": 1,
            "warmup_steps": 2350, "baseline_only": False, "tier": 3,
            "threshold": THRESHOLD, "provenance": "seeded fixture",
            "norm_offset": np.zeros(n_channels, dtype=np.float32),
            "norm_scale": np.ones(n_channels, dtype=np.float32),
        },
    }


def corrupt(data, offset, value=None):
    out = bytearray(data)
    out[offset] = (out[offset] ^ 0xFF) if value is None else value
    return bytes(out)


# -- the round trip ---------------------------------------------------------

def test_a_file_round_trips_through_the_reader():
    spec = make_spec()
    status, back = read_model(write_model(spec), FLIGHT_LIMITS)
    assert status is Status.OK
    assert back["hidden"] == spec["hidden"]
    assert back["n_channels"] == spec["n_channels"]
    assert back["window"] == spec["window"]
    assert back["n_predictions"] == spec["n_predictions"]
    assert [c["name"] for c in back["channels"]] == [c["name"] for c in spec["channels"]]
    for (name, written), (back_name, read) in zip(spec["arrays"], back["arrays"]):
        assert name == back_name
        assert np.array_equal(written, read), name


def test_writing_the_read_spec_reproduces_the_bytes():
    """Python -> bytes -> Python -> bytes. The C++ leg is `flight/test/`."""
    first = write_model(make_spec())
    status, back = read_model(first)
    assert status is Status.OK
    assert write_model(back) == first


@pytest.mark.parametrize("n_channels, hidden, l_p, expected", [
    (3, (4, 4), 2, 1_276),          # G1, the committed byte-frozen tier
    (7, (24, 24), 5, 27_760),       # G2, fixture scale
    (12, (80, 80), 10, 285_136),    # G3/G4, the flown shape
])
def test_the_file_is_exactly_the_size_the_pre_registration_predicted(
        n_channels, hidden, l_p, expected):
    """`docs/MODELS.md` 19, prediction F6, and 19.6's tier table."""
    blob = write_model(make_spec(n_channels, hidden, l_p))
    assert len(blob) == expected


def test_the_flown_shape_carries_the_parameter_count_phase_1_measured():
    """71,160 parameters, 284,640 bytes (`docs/MODELS.md` 3)."""
    assert fmt.parameter_count(12, [80, 80], 10, 12) == 71_160
    assert fmt.weight_bytes(12, [80, 80], 10, 12) == 284_640


# -- what the header must say, rather than leave to be inferred -------------

def test_the_gate_order_and_architecture_are_named_in_the_header():
    """`docs/MODELS.md` 3: gate order in the format, not in the reader's memory."""
    blob = write_model(make_spec())
    header = fmt.HEADER_STRUCT.unpack(blob[:fmt.HEADER_BYTES])
    assert header[0] == b"SNTL"
    assert header[1] == fmt.FORMAT_VERSION
    assert header[3] == fmt.ARCH_GRU
    assert header[4] == fmt.GATE_ORDER_RESET_UPDATE_NEW


def test_both_bias_vectors_are_stored_unsummed():
    """`b_hn` sits inside the reset product and cannot be folded (D26).

    The format's job is to make folding impossible to do by accident: both vectors
    occupy their own bytes, and neither is recoverable from the other.
    """
    spec = make_spec()
    _, back = read_model(write_model(spec))
    stored = dict(back["arrays"])
    written = dict(spec["arrays"])
    for layer in ("l0", "l1"):
        assert np.array_equal(stored[f"{layer}_b_ih"], written[f"{layer}_b_ih"])
        assert np.array_equal(stored[f"{layer}_b_hh"], written[f"{layer}_b_hh"])
        assert not np.allclose(stored[f"{layer}_b_hh"], 0.0), (
            "a zero b_hh would make the fold undetectable")


def test_the_threshold_survives_as_float64():
    """`harness.py:169-172` compares in float64; an F32 field would round the cut."""
    spec = make_spec()
    assert float(np.float32(THRESHOLD)) != THRESHOLD, "pick a value F32 cannot hold"
    _, back = read_model(write_model(spec))
    assert back["params"]["threshold"] == THRESHOLD


def test_the_parameter_block_carries_the_decision_layer_constants():
    _, back = read_model(write_model(make_spec()))
    params = back["params"]
    assert params["ewma_span"] == 105          # D25, telemanom smoothing_perc 0.05
    assert params["agreement"] == 1            # k-of-n = 1, the maximum
    assert params["persistence"] == 1          # N=1, no persistence filter
    assert params["warmup_steps"] == 2350      # window 250 + error_window 2100
    assert params["norm_policy"] == "identity"  # D2
    assert params["baseline_only"] is False    # Objective.md 14.10
    assert np.array_equal(params["norm_offset"], np.zeros(3, dtype=np.float32))
    assert np.array_equal(params["norm_scale"], np.ones(3, dtype=np.float32))


def test_the_file_is_little_endian_whatever_the_host_is():
    blob = write_model(make_spec())
    assert blob[4:6] == b"\x01\x00", "format_version 1 as little-endian U16"
    assert int.from_bytes(blob[6:8], "little") == fmt.HEADER_BYTES


# -- refusal, and the order the checks happen in ---------------------------

def test_a_bad_magic_is_refused():
    assert read_model(corrupt(write_model(make_spec()), 0))[0] is Status.BAD_MAGIC


def test_a_bad_version_is_refused():
    blob = bytearray(write_model(make_spec()))
    blob[4:6] = (fmt.FORMAT_VERSION + 1).to_bytes(2, "little")
    assert read_model(bytes(blob))[0] is Status.BAD_VERSION


def test_a_bad_header_crc_is_refused():
    assert read_model(corrupt(write_model(make_spec()), 60))[0] is Status.BAD_HEADER_CRC


def test_a_bad_static_crc_is_refused():
    blob = write_model(make_spec())
    weight_lo = fmt.HEADER_BYTES + fmt.channel_bytes(3)
    assert read_model(corrupt(blob, weight_lo + 8))[0] is Status.BAD_STATIC_CRC


def test_a_bad_param_crc_is_refused():
    blob = write_model(make_spec())
    assert read_model(corrupt(blob, len(blob) - 4))[0] is Status.BAD_PARAM_CRC


def test_a_truncated_file_is_refused():
    blob = write_model(make_spec())
    assert read_model(blob[:-1])[0] is Status.TRUNCATED
    assert read_model(blob[:10])[0] is Status.TRUNCATED
    assert read_model(b"")[0] is Status.TRUNCATED


def test_a_model_past_the_compile_time_maxima_is_refused():
    """The header is validated against `Config.hpp`'s budget, not obeyed blindly."""
    blob = write_model(make_spec(n_channels=20, hidden=(4, 4), n_predictions=2))
    assert read_model(blob, FLIGHT_LIMITS)[0] is Status.TOO_LARGE
    assert read_model(blob)[0] is Status.OK, "the format itself permits it; the build does not"


def test_a_third_layer_is_refused_by_the_build_and_allowed_by_the_format():
    blob = write_model(make_spec(hidden=(4, 4, 4)))
    assert read_model(blob)[0] is Status.OK
    assert read_model(blob, FLIGHT_LIMITS)[0] is Status.TOO_LARGE


def test_a_wrong_architecture_or_gate_order_is_refused_by_name():
    blob = bytearray(write_model(make_spec()))
    header = bytearray(blob)
    header[8:10] = fmt.ARCH_LSTM.to_bytes(2, "little")
    assert read_model(_resign(header))[0] is Status.BAD_ARCH
    header = bytearray(blob)
    header[10:12] = fmt.GATE_ORDER_INPUT_FORGET_CELL_OUTPUT.to_bytes(2, "little")
    assert read_model(_resign(header))[0] is Status.BAD_GATE_ORDER


def test_a_non_zero_reserved_field_is_refused():
    """A reserved field that is tolerated is a field that drifts silently."""
    header = bytearray(write_model(make_spec()))
    header[52:56] = (1).to_bytes(4, "little")
    assert read_model(_resign(header))[0] is Status.BAD_SHAPE


def test_a_shape_that_does_not_account_for_the_payload_is_refused():
    """`weight_bytes` must equal exactly what the declared shape implies (D16)."""
    header = bytearray(write_model(make_spec()))
    header[14:16] = (4).to_bytes(2, "little")        # n_channels 3 -> 4
    assert read_model(_resign(header))[0] is Status.BAD_SHAPE


def test_the_header_crc_is_checked_before_any_length_field_is_used():
    """A corrupted length must never drive an over-read.

    `weight_bytes` is set to 4 GB with the header CRC left stale. The reader must
    answer BAD_HEADER_CRC -- not BAD_SHAPE, and certainly not by reading.
    """
    blob = bytearray(write_model(make_spec()))
    blob[36:40] = (0xFFFFFFF0).to_bytes(4, "little")
    assert read_model(bytes(blob))[0] is Status.BAD_HEADER_CRC


def test_a_normalisation_policy_other_than_identity_is_refused():
    """Identity is the only policy this version accepts (D2, Objective.md 14.8)."""
    blob = write_model(make_spec())
    param_lo = len(blob) - fmt.param_bytes(3)
    patched = bytearray(blob)
    patched[param_lo + 2:param_lo + 4] = (1).to_bytes(2, "little")
    assert read_model(_reseal_params(patched, 3))[0] is Status.BAD_NORM_POLICY


# -- what the writer refuses to produce ------------------------------------

def test_the_writer_refuses_float64_arrays():
    spec = make_spec()
    name, array = spec["arrays"][0]
    spec["arrays"][0] = (name, array.astype(np.float64))
    with pytest.raises(ModelFileError, match="float32"):
        write_model(spec)


def test_the_writer_refuses_a_channel_name_that_would_lose_its_terminator():
    spec = make_spec()
    spec["channels"][0]["name"] = "x" * fmt.CHANNEL_NAME_BYTES
    with pytest.raises(ModelFileError, match="NUL-terminated"):
        write_model(spec)


def test_the_writer_refuses_a_non_identity_normalisation_policy():
    spec = make_spec()
    spec["params"]["norm_policy"] = "zscore"
    with pytest.raises(ModelFileError, match="identity"):
        write_model(spec)


def test_the_writer_refuses_arrays_out_of_order():
    spec = make_spec()
    spec["arrays"][0], spec["arrays"][1] = spec["arrays"][1], spec["arrays"][0]
    with pytest.raises(ModelFileError, match="in that order"):
        write_model(spec)


# -- the in-orbit recalibration path ---------------------------------------

def test_replacing_the_parameters_leaves_the_weights_and_their_crc_untouched():
    """`docs/MODEL_FILE.md` 6.1, and prediction F8 of `docs/MODELS.md` 19.

    This is the whole reason the parameter block carries its own CRC: D29 measured
    a floor that sat under 86.7% of a later window on the same spacecraft, so a
    threshold must be replaceable in orbit without retraining.
    """
    spec = make_spec()
    original = write_model(spec)

    spec["params"]["threshold"] = 0.0109
    spec["params"]["param_version"] = 2
    spec["params"]["provenance"] = "recalibrated on the later window"
    updated = replace_params(original, spec)

    assert len(updated) == len(original)
    static_end = len(original) - fmt.param_bytes(3)
    assert updated[fmt.HEADER_BYTES:static_end] == original[fmt.HEADER_BYTES:static_end], (
        "the weights were rewritten")
    assert updated[44:48] == original[44:48], "static_crc32 moved"
    assert updated[48:52] != original[48:52], "param_crc32 did not move"

    status, back = read_model(updated, FLIGHT_LIMITS)
    assert status is Status.OK
    assert back["params"]["threshold"] == 0.0109
    assert back["params"]["param_version"] == 2
    assert all(np.array_equal(a, b) for (_, a), (_, b)
               in zip(read_model(original)[1]["arrays"], back["arrays"]))


def test_only_the_parameter_block_and_two_header_words_change():
    spec = make_spec()
    original = write_model(spec)
    spec["params"]["threshold"] = 0.5
    updated = replace_params(original, spec)
    differing = {i for i, (a, b) in enumerate(zip(original, updated)) if a != b}
    param_lo = len(original) - fmt.param_bytes(3)
    allowed = set(range(48, 52)) | set(range(60, 64)) | set(range(param_lo, len(original)))
    assert differing <= allowed, sorted(differing - allowed)[:8]


# -- helpers ---------------------------------------------------------------

def _resign(header_and_body: bytearray) -> bytes:
    """Re-sign the header so a field test reaches the check it is aiming at."""
    out = bytearray(header_and_body)
    out[60:64] = (zlib.crc32(bytes(out[:60])) & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)


def _reseal_params(blob: bytearray, n_channels: int) -> bytes:
    """Recompute param_crc32 and the header CRC after editing the parameter block."""
    out = bytearray(blob)
    param_lo = len(out) - fmt.param_bytes(n_channels)
    out[48:52] = (zlib.crc32(bytes(out[param_lo:])) & 0xFFFFFFFF).to_bytes(4, "little")
    return _resign(out)
