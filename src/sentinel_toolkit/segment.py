"""D85: the training segment, kept beside every model the toolkit fits.

`docs/DECISIONS.md` D85, owner decision 1: the control model for D78's gate is
reproduced ON THE GROUND, and the ground can only do that if it still holds the
telemetry the flying model was trained on. Nothing required keeping it before
(D78 consequence 3). This module makes `fit` keep it.

**The model file format does not change.** `model.bin` has no room for a segment
(reserved fields must be zero; a new field is a `format_version` bump,
`docs/MODEL_FILE.md` 11), so the segment is a SIDECAR, ``<model>.segment.npz``,
bound to the model by two numbers the model already carries or implies:

* the sha256 of the model file's bytes, and
* its ``static_crc32`` (header offset 44), which covers the channel records and
  the weights and is unchanged by a PARAMS replacement -- so it identifies the
  WEIGHTS even if the threshold is later re-calibrated.

A sidecar whose numbers do not match the model it is handed is refused, because a
control reproduced from the wrong segment would make part (i) compare two unrelated
fits and certify on the difference.

(!) THE SIDECAR IS MISSION TELEMETRY. It lives where the mission keeps its data. In
this repository that is only ever under gitignored ``runs/`` or a test's temporary
directory (``tests/test_no_local_persistence.py``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ToolkitError

SUFFIX = ".segment.npz"
STATIC_CRC_OFFSET = 44


def sidecar_path(model_path: Path) -> Path:
    return Path(str(model_path) + SUFFIX)


def static_crc32(model_bytes: bytes) -> int:
    if len(model_bytes) < STATIC_CRC_OFFSET + 4:
        raise ToolkitError("a model file shorter than its own header")
    return int.from_bytes(model_bytes[STATIC_CRC_OFFSET:STATIC_CRC_OFFSET + 4], "little")


@dataclass(frozen=True)
class Segment:
    train: np.ndarray          #: (rows, channels) float32, what the model was fitted on
    meta: dict                 #: the fit's identity and hyperparameters


def write(model_path: Path, model_bytes: bytes, train: np.ndarray, meta: dict) -> Path:
    """Write the sidecar beside the model. `meta` gains the model's identity."""
    identity = dict(meta)
    identity["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    identity["static_crc32"] = static_crc32(model_bytes)
    identity["rows"] = int(train.shape[0])
    identity["channels"] = int(train.shape[1])
    path = sidecar_path(model_path)
    with open(path, "wb") as handle:
        np.savez(handle, train=np.ascontiguousarray(train, dtype=np.float32),
                 meta=np.frombuffer(json.dumps(identity, sort_keys=True).encode("ascii"),
                                    dtype=np.uint8))
    return path


def read(model_path: Path, model_bytes: bytes) -> Segment:
    """Read and VERIFY the sidecar of `model_path`. Refuses a mismatch."""
    path = sidecar_path(model_path)
    if not path.exists():
        raise ToolkitError(
            f"no training segment beside {model_path} ({path.name}). D85: the ground "
            "reproduces the control from the segment the flying model was trained on, "
            "and `sentinel_toolkit fit` writes it; a model without one cannot be gated.")
    with np.load(path, allow_pickle=False) as data:
        train = np.array(data["train"], dtype=np.float32)
        meta = json.loads(bytes(data["meta"]).decode("ascii"))
    want_sha = hashlib.sha256(model_bytes).hexdigest()
    if meta.get("model_sha256") != want_sha:
        raise ToolkitError(
            f"{path.name} belongs to a different model (sha256 {meta.get('model_sha256')}, "
            f"model is {want_sha}). Refused: a control reproduced from another model's "
            "segment would make part (i) compare two unrelated fits.")
    if meta.get("static_crc32") != static_crc32(model_bytes):
        raise ToolkitError(f"{path.name}'s static_crc32 does not match the model's")
    if train.shape != (meta.get("rows"), meta.get("channels")):
        raise ToolkitError(f"{path.name}'s array is {train.shape}, its record says "
                           f"({meta.get('rows')}, {meta.get('channels')})")
    return Segment(train=train, meta=meta)
