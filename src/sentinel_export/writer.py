"""Write a `model.bin`. `docs/MODEL_FILE.md` is normative; this obeys it.

Takes a plain dictionary of arrays and metadata, not a model object, because the
format is a written specification and this package depends on nothing else in the
repository. Whoever has a `reference.Weights` converts it; see
`scripts/make_golden_vectors.py`.
"""
from __future__ import annotations

import zlib

import numpy as np

from . import format as fmt


class ModelFileError(ValueError):
    """The spec handed in cannot be written as a valid file.

    Ground-side only. The flight reader returns a status and never throws
    (F' CPP-25); refusing to *write* a bad file is a different obligation from
    refusing to read one, and a Python exception is the right shape for it.
    """


def _ascii_field(text: str, width: int, what: str) -> bytes:
    """NUL-padded ASCII, refusing anything that would not survive the round trip."""
    raw = str(text).encode("ascii", errors="strict")
    if len(raw) >= width:
        raise ModelFileError(
            f"{what} is {len(raw)} bytes and must be under {width} so it stays "
            f"NUL-terminated: {text!r}")
    return raw.ljust(width, b"\x00")


def build_params(spec: dict) -> bytes:
    """The parameter block: everything replaceable in orbit, and nothing else."""
    params = spec["params"]
    n_channels = int(spec["n_channels"])

    policy = params.get("norm_policy", "identity")
    if policy != "identity" and policy != fmt.NORM_IDENTITY:
        raise ModelFileError(
            f"normalisation policy {policy!r}: identity is the only policy this "
            f"format version accepts (D2, Objective.md 14.8)")

    offset = np.asarray(params["norm_offset"], dtype=np.float32)
    scale = np.asarray(params["norm_scale"], dtype=np.float32)
    for name, array in (("norm_offset", offset), ("norm_scale", scale)):
        if array.shape != (n_channels,):
            raise ModelFileError(
                f"{name} is {array.shape}, expected ({n_channels},)")

    head = fmt.PARAM_STRUCT.pack(
        int(params.get("param_version", fmt.PARAM_VERSION)),
        fmt.NORM_IDENTITY,
        int(params["ewma_span"]),
        int(params["agreement"]),
        int(params["persistence"]),
        0,                                       # reserved0
        int(params["warmup_steps"]),
        1 if params.get("baseline_only", False) else 0,
        int(params.get("tier", 3)),
        0,                                       # reserved1
        0,                                       # reserved2
        float(params["threshold"]),              # F64 -- see MODEL_FILE.md 6
        _ascii_field(params.get("provenance", ""), fmt.PROVENANCE_BYTES,
                     "provenance"),
    )
    block = head + offset.tobytes() + scale.tobytes()
    expected = fmt.param_bytes(n_channels)
    if len(block) != expected:
        raise ModelFileError(f"parameter block is {len(block)} bytes, expected {expected}")
    return block


def build_channels(spec: dict) -> bytes:
    """The channel map, in the model's channel order (Objective.md 4.1)."""
    channels = spec["channels"]
    n_channels = int(spec["n_channels"])
    if len(channels) != n_channels:
        raise ModelFileError(
            f"{len(channels)} channel records for n_channels={n_channels}")
    out = bytearray()
    for channel in channels:
        out += fmt.CHANNEL_STRUCT.pack(
            int(channel["id"]),
            _ascii_field(channel["name"], fmt.CHANNEL_NAME_BYTES, "channel name"))
    return bytes(out)


def build_weights(spec: dict) -> bytes:
    """Float32, C order, in `reference.Weights.arrays()` order.

    Both bias vectors are written unsummed. For a GRU that is not a preference:
    `b_hn` sits inside the reset product and cannot be folded into `b_in`
    (`docs/MODELS.md` 3, D26).
    """
    hidden = [int(h) for h in spec["hidden"]]
    n_channels = int(spec["n_channels"])
    n_inputs = n_channels + int(spec.get("n_exogenous", 0))
    n_predictions = int(spec["n_predictions"])
    supplied = dict(spec["arrays"])

    expected = fmt.array_shapes(n_inputs, hidden, n_predictions, n_channels)
    if list(supplied) != [name for name, _ in expected]:
        raise ModelFileError(
            f"arrays are {list(supplied)}, expected {[n for n, _ in expected]} "
            f"in that order")

    out = bytearray()
    for name, shape in expected:
        array = np.asarray(supplied[name])
        if array.shape != shape:
            raise ModelFileError(f"{name} is {array.shape}, expected {shape}")
        if array.dtype != np.float32:
            raise ModelFileError(
                f"{name} is {array.dtype}; the format stores float32 and a silent "
                f"cast here would be a different model")
        out += np.ascontiguousarray(array).tobytes()
    return bytes(out)


def write_model(spec: dict) -> bytes:
    """Assemble the whole file. Returns the bytes; the caller decides where.

    Three CRCs: one over the header, one over the static payload, and one over the
    parameter block alone -- which is what lets a mission recalibrate in orbit
    without touching the weights (`docs/MODEL_FILE.md` 2, 6.1).
    """
    arch = spec.get("arch", "gru")
    if arch != "gru" and arch != fmt.ARCH_GRU:
        raise ModelFileError(
            f"architecture {arch!r}: the GRU is the flight architecture (D28) and "
            f"the only one this writer emits")

    hidden = [int(h) for h in spec["hidden"]]
    if not 1 <= len(hidden) <= fmt.HEADER_LAYER_SLOTS:
        raise ModelFileError(
            f"{len(hidden)} layers; the header carries {fmt.HEADER_LAYER_SLOTS} slots")

    n_channels = int(spec["n_channels"])
    n_exogenous = int(spec.get("n_exogenous", 0))
    n_inputs = n_channels + n_exogenous

    channels = build_channels(spec)
    weights = build_weights(spec)
    params = build_params(spec)

    static_crc = zlib.crc32(channels + weights) & 0xFFFFFFFF
    param_crc = zlib.crc32(params) & 0xFFFFFFFF

    slots = hidden + [0] * (fmt.HEADER_LAYER_SLOTS - len(hidden))
    header = fmt.HEADER_STRUCT.pack(
        fmt.MAGIC,
        fmt.FORMAT_VERSION,
        fmt.HEADER_BYTES,
        fmt.ARCH_GRU,
        fmt.GATE_ORDER_RESET_UPDATE_NEW,
        len(hidden),
        n_channels,
        n_inputs,
        n_exogenous,
        int(spec["window"]),
        int(spec["n_predictions"]),
        *slots,
        len(channels),
        len(weights),
        len(params),
        static_crc,
        param_crc,
        0,                       # reserved0
        0,                       # reserved1
        0,                       # header_crc32, filled below
    )
    header = header[:fmt.HEADER_CRC_OFFSET] + _pack_u32(
        zlib.crc32(header[:fmt.HEADER_CRC_OFFSET]) & 0xFFFFFFFF)

    return header + channels + weights + params


def _pack_u32(value: int) -> bytes:
    return int(value).to_bytes(4, "little", signed=False)


def replace_params(data: bytes, spec: dict) -> bytes:
    """Rewrite only the parameter block, as an in-orbit recalibration does.

    `docs/MODEL_FILE.md` 6.1: overwrite PARAMS in place, patch `param_crc32` and
    `header_crc32`, and leave `static_crc32` and the weights untouched. This
    function exists so that property is executable rather than asserted.
    """
    params = build_params(spec)
    n_channels = int(spec["n_channels"])
    if len(params) != fmt.param_bytes(n_channels):
        raise ModelFileError("replacement parameter block changed size")

    head = bytearray(data[:fmt.HEADER_BYTES])
    body_end = len(data) - len(params)
    if body_end < fmt.HEADER_BYTES:
        raise ModelFileError("file is shorter than its own parameter block")

    head[48:52] = _pack_u32(zlib.crc32(params) & 0xFFFFFFFF)
    head[fmt.HEADER_CRC_OFFSET:fmt.HEADER_BYTES] = _pack_u32(
        zlib.crc32(bytes(head[:fmt.HEADER_CRC_OFFSET])) & 0xFFFFFFFF)
    return bytes(head) + data[fmt.HEADER_BYTES:body_end] + params
