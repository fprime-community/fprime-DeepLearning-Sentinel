"""Read a `model.bin`, returning a status. The C++ in `flight/` mirrors this.

The flight reader cannot throw -- flight code compiles `-fno-exceptions`
(F' CPP-25) -- so it returns a `LoadStatus`. This reader returns the same value
for the same file, which is what lets one test assert that both sides refuse the
same bytes for the same reason.

Order of checks matters and is fixed by `docs/MODEL_FILE.md` 8: magic, version,
then the header CRC, **before any length field is used**, so a corrupted
`weight_bytes` can never drive an over-read.
"""
from __future__ import annotations

import zlib

import numpy as np

from . import format as fmt
from .format import Status


def read_model(data: bytes, limits: dict | None = None) -> tuple[Status, dict | None]:
    """Parse and verify. Returns `(status, spec)`; `spec` is None unless OK.

    `limits` mirrors the C++ compile-time maxima in
    `flight/include/sentinel/Config.hpp`. Pass them to get the same TOO_LARGE
    refusal the flight reader gives; omit them and only the format's own bounds
    apply.
    """
    if len(data) < fmt.HEADER_BYTES:
        return Status.TRUNCATED, None
    if data[:4] != fmt.MAGIC:
        return Status.BAD_MAGIC, None

    header = fmt.HEADER_STRUCT.unpack(data[:fmt.HEADER_BYTES])
    (_magic, format_version, header_bytes, arch_id, gate_order_id, n_layers,
     n_channels, n_inputs, n_exogenous, window, n_predictions,
     h0, h1, h2, h3,
     channel_bytes, weight_bytes, param_bytes,
     static_crc, param_crc, reserved0, reserved1, header_crc) = header

    if format_version != fmt.FORMAT_VERSION:
        return Status.BAD_VERSION, None

    computed = zlib.crc32(data[:fmt.HEADER_CRC_OFFSET]) & 0xFFFFFFFF
    if computed != header_crc:
        return Status.BAD_HEADER_CRC, None

    # Only now is anything in the header trusted enough to bound a read.
    if header_bytes != fmt.HEADER_BYTES or reserved0 != 0 or reserved1 != 0:
        return Status.BAD_SHAPE, None
    if arch_id != fmt.ARCH_GRU:
        return Status.BAD_ARCH, None
    if gate_order_id != fmt.GATE_ORDER_RESET_UPDATE_NEW:
        return Status.BAD_GATE_ORDER, None

    slots = [h0, h1, h2, h3]
    if not 1 <= n_layers <= fmt.HEADER_LAYER_SLOTS:
        return Status.BAD_SHAPE, None
    hidden = slots[:n_layers]
    if any(h == 0 for h in hidden) or any(h != 0 for h in slots[n_layers:]):
        return Status.BAD_SHAPE, None
    if n_inputs != n_channels + n_exogenous or n_channels == 0:
        return Status.BAD_SHAPE, None

    if limits is not None:
        if (n_channels > limits["MAX_CHANNELS"] or n_layers > limits["MAX_LAYERS"]
                or max(hidden) > limits["MAX_HIDDEN"]
                or n_predictions > limits["MAX_PREDICTIONS"]
                or n_inputs > limits["MAX_INPUTS"]):
            return Status.TOO_LARGE, None

    # Every declared size must account for the payload exactly. A field written
    # and not checked is D16.
    if channel_bytes != fmt.channel_bytes(n_channels):
        return Status.BAD_SHAPE, None
    if weight_bytes != fmt.weight_bytes(n_inputs, hidden, n_predictions, n_channels):
        return Status.BAD_SHAPE, None
    if param_bytes != fmt.param_bytes(n_channels):
        return Status.BAD_SHAPE, None

    total = fmt.HEADER_BYTES + channel_bytes + weight_bytes + param_bytes
    if len(data) != total:
        return Status.TRUNCATED, None

    channel_lo = fmt.HEADER_BYTES
    weight_lo = channel_lo + channel_bytes
    param_lo = weight_lo + weight_bytes

    if zlib.crc32(data[channel_lo:param_lo]) & 0xFFFFFFFF != static_crc:
        return Status.BAD_STATIC_CRC, None
    if zlib.crc32(data[param_lo:]) & 0xFFFFFFFF != param_crc:
        return Status.BAD_PARAM_CRC, None

    channels = []
    for i in range(n_channels):
        lo = channel_lo + i * fmt.CHANNEL_RECORD_BYTES
        cid, raw = fmt.CHANNEL_STRUCT.unpack(data[lo:lo + fmt.CHANNEL_RECORD_BYTES])
        if raw[-1] != 0:
            return Status.BAD_SHAPE, None          # not NUL-terminated
        channels.append({"id": int(cid), "name": raw.rstrip(b"\x00").decode("ascii")})

    arrays = []
    cursor = weight_lo
    for name, shape in fmt.array_shapes(n_inputs, hidden, n_predictions, n_channels):
        size = 1
        for dim in shape:
            size *= dim
        chunk = np.frombuffer(data, dtype="<f4", count=size,
                              offset=cursor).reshape(shape)
        arrays.append((name, np.array(chunk, dtype=np.float32)))
        cursor += 4 * size

    (param_version, norm_policy, ewma_span, agreement, persistence, p_res0,
     warmup_steps, baseline_only, tier, p_res1, p_res2, threshold,
     provenance) = fmt.PARAM_STRUCT.unpack(
        data[param_lo:param_lo + fmt.PARAM_FIXED_BYTES])

    if p_res0 != 0 or p_res1 != 0 or p_res2 != 0:
        return Status.BAD_SHAPE, None

    # (!) VALIDATED, NOT MERELY READ. A version-1 block's `threshold` cuts the
    # EWMA of the absolute residual; a version-2 block's cuts D65's fused
    # `max(z_residual, z_derivative)`. Different scales, identical bytes. A
    # reader that accepted any value would apply one to the other in silence.
    if int(param_version) not in fmt.SUPPORTED_PARAM_VERSIONS:
        return Status.BAD_PARAM_VERSION, None
    if norm_policy != fmt.NORM_IDENTITY:
        return Status.BAD_NORM_POLICY, None
    if baseline_only > 1 or not 1 <= tier <= 3:
        return Status.BAD_SHAPE, None
    if provenance[-1] != 0:
        return Status.BAD_SHAPE, None

    constants_lo = param_lo + fmt.PARAM_FIXED_BYTES
    norm_offset = np.array(np.frombuffer(data, dtype="<f4", count=n_channels,
                                         offset=constants_lo), dtype=np.float32)
    norm_scale = np.array(np.frombuffer(data, dtype="<f4", count=n_channels,
                                        offset=constants_lo + 4 * n_channels),
                          dtype=np.float32)

    spec = {
        "arch": "gru",
        "n_channels": int(n_channels),
        "n_exogenous": int(n_exogenous),
        "window": int(window),
        "n_predictions": int(n_predictions),
        "hidden": [int(h) for h in hidden],
        "channels": channels,
        "arrays": arrays,
        "params": {
            "param_version": int(param_version),
            "norm_policy": "identity",
            "ewma_span": int(ewma_span),
            "agreement": int(agreement),
            "persistence": int(persistence),
            "warmup_steps": int(warmup_steps),
            "baseline_only": bool(baseline_only),
            "tier": int(tier),
            "threshold": float(threshold),
            "provenance": provenance.rstrip(b"\x00").decode("ascii"),
            "norm_offset": norm_offset,
            "norm_scale": norm_scale,
        },
    }
    return Status.OK, spec
