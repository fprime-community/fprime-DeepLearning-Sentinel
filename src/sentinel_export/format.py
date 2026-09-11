"""The `model.bin` layout, as constants. `docs/MODEL_FILE.md` is normative.

This module is the single place a byte offset or a magic value is written down on
the Python side. It holds no logic beyond arithmetic on those constants, so the
writer and the reader cannot drift from each other, and the C++ in `flight/` is
written against the same document rather than against this file.

Standard library and numpy only. This package depends on nothing else in this
repository, deliberately: the format is defined by a written specification, not by
a Python class, because the C++ loader cannot import from here.
"""
from __future__ import annotations

import struct
from enum import IntEnum

#: Bumped for any change to the byte layout. A version-1 reader refuses anything else.
FORMAT_VERSION = 1

#: Bumped when only the parameter block changes -- a recalibration in orbit leaves
#: `FORMAT_VERSION` alone, which is what lets a threshold have a provenance separate
#: from the model it belongs to (D29).
PARAM_VERSION = 1

#: D68's flight configuration. A version-2 PARAMS block is byte-identical in
#: LAYOUT to a version-1 one -- no field is added, moved or resized, so
#: `FORMAT_VERSION` stays 1 and D30's freeze is untouched -- but its `threshold`
#: cuts a different statistic: the fused `max(z_residual, z_derivative)` of
#: `docs/DECISIONS.md` D65, not the EWMA of the absolute residual.
#:
#: (!) THIS IS WHY THE FIELD IS VALIDATED RATHER THAN READ. The two thresholds
#: are on different scales, and a reader that accepted either would apply a
#: 99.9th-percentile EWMA cut to a z-score stream and never say a word. That is
#: the silent drift `docs/MODEL_FILE.md` 1 exists to forbid, so a reader refuses
#: a `param_version` it does not know with `Status.BAD_PARAM_VERSION`.
PARAM_VERSION_FUSED = 2

#: What a reader of this generation accepts.
SUPPORTED_PARAM_VERSIONS = (PARAM_VERSION, PARAM_VERSION_FUSED)

MAGIC = b"SNTL"

#: Architectures. Only the GRU flies (D28); the other two are reserved so that a
#: file claiming to be one is refused by name rather than misread.
ARCH_GRU = 1
ARCH_LSTM = 2
ARCH_TCN = 3

#: Gate orders, named in the header so no reader ever infers one from a slice index
#: (`docs/MODELS.md` 3). 1 is `reference.GRU_GATES == ("reset", "update", "new")`.
GATE_ORDER_RESET_UPDATE_NEW = 1
GATE_ORDER_INPUT_FORGET_CELL_OUTPUT = 2

#: Normalisation policies. Identity is the only one a version-1 reader accepts (D2).
NORM_IDENTITY = 0

#: The header carries four hidden slots so the format can describe a deeper model
#: than this build budgets for. `flight/include/sentinel/Config.hpp` budgets two and
#: refuses more with TOO_LARGE, which is what that status exists to demonstrate.
HEADER_LAYER_SLOTS = 4

HEADER_BYTES = 64
HEADER_CRC_OFFSET = 60          # header_crc32 covers bytes [0, 60)
HEADER_STRUCT = struct.Struct("<4s10H4H8I")

CHANNEL_NAME_BYTES = 16
CHANNEL_RECORD_BYTES = 20
CHANNEL_STRUCT = struct.Struct("<I16s")

PROVENANCE_BYTES = 64
PARAM_FIXED_BYTES = 96
PARAM_STRUCT = struct.Struct("<6HIBBHId64s")

#: The GRU's three gate blocks, stacked along axis 0 of `w_ih` and `w_hh`.
N_GATES = 3

#: The on-disk array order, which is `reference.Weights.arrays()` exactly:
#: layers in index order, `w_ih, w_hh, b_ih, b_hh` within each, then the head.
ARRAY_SUFFIXES = ("w_ih", "w_hh", "b_ih", "b_hh")


class Status(IntEnum):
    """What a reader returns. The values are shared with the C++ `LoadStatus`.

    Flight code cannot throw (F' CPP-25, `-fno-exceptions`), so the C++ reader
    returns one of these. The Python reader returns the same value for the same
    file, which is what lets one test assert both sides agree.
    """

    OK = 0
    BAD_MAGIC = 1
    BAD_VERSION = 2
    BAD_HEADER_CRC = 3
    BAD_STATIC_CRC = 4
    BAD_PARAM_CRC = 5
    BAD_ARCH = 6
    BAD_GATE_ORDER = 7
    BAD_SHAPE = 8
    TOO_LARGE = 9
    TRUNCATED = 10
    BAD_NORM_POLICY = 11
    #: `param_version` names a PARAMS block this reader cannot read. D68: a
    #: version-1 threshold cuts the EWMA statistic, a version-2 one cuts the
    #: fused `max(z_residual, z_derivative)`, and they are different scales.
    BAD_PARAM_VERSION = 12



def array_names(n_layers: int) -> list[str]:
    """The array names, in the order the file stores them."""
    out = []
    for i in range(n_layers):
        out += [f"l{i}_{suffix}" for suffix in ARRAY_SUFFIXES]
    return out + ["head_w", "head_b"]


def array_shapes(n_inputs: int, hidden: list[int], n_predictions: int,
                 n_channels: int) -> list[tuple[str, tuple[int, ...]]]:
    """Every array's name and shape, derived from the header fields alone.

    This is what makes `weight_bytes` checkable rather than merely readable: the
    declared shape must account for the payload exactly, to the byte (D16).
    """
    out: list[tuple[str, tuple[int, ...]]] = []
    previous = n_inputs
    for i, h in enumerate(hidden):
        gates = N_GATES * h
        out += [(f"l{i}_w_ih", (gates, previous)),
                (f"l{i}_w_hh", (gates, h)),
                (f"l{i}_b_ih", (gates,)),
                (f"l{i}_b_hh", (gates,))]
        previous = h
    outputs = n_predictions * n_channels
    out += [("head_w", (outputs, previous)), ("head_b", (outputs,))]
    return out


def parameter_count(n_inputs: int, hidden: list[int], n_predictions: int,
                    n_channels: int) -> int:
    """Total float32 elements. 71,160 at the flown shape (`docs/MODELS.md` 3)."""
    total = 0
    for _, shape in array_shapes(n_inputs, hidden, n_predictions, n_channels):
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


def channel_bytes(n_channels: int) -> int:
    return CHANNEL_RECORD_BYTES * n_channels


def weight_bytes(n_inputs: int, hidden: list[int], n_predictions: int,
                 n_channels: int) -> int:
    return 4 * parameter_count(n_inputs, hidden, n_predictions, n_channels)


def param_bytes(n_channels: int) -> int:
    """The parameter block is fixed-size for a given model, so an in-orbit
    recalibration is a fixed-length overwrite (`docs/MODEL_FILE.md` 6.1)."""
    return PARAM_FIXED_BYTES + 8 * n_channels
