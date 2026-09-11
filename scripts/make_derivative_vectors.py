"""Generate the `DerivativeStream` golden vectors, tiers F1-F2.

`flight/src/DerivativeStream.cpp` is held to `scripts/decision_layer_arms.py`
`:224` -- `zstat(np.abs(np.diff(x, prepend=x[0])), span)` -- which is arm 2's
second stream, the one D65 measured at EVAL 17 of 19 fused with the residual.

Two things these pin that nothing else does:

  the prepend        `np.diff(x, prepend=x[0])` makes `dx[0]` exactly zero. Taking
                     the first difference against nothing instead would put a
                     spurious excursion at every reset, on every channel, at the
                     one moment a detector is least able to judge it.
  the span           2,100 -- `error_window`, not the threshold's 2,170. The two
                     windows are different lengths over the same kind of data and
                     the core carries both on one ring.

Format, little-endian:

    off  size          field
      0     4          magic 'SNTF'
      4     2          version U16 = 1
      6     2          n_channels U16
      8     4          steps U32
     12     4          span U32     = 2100
    ...    per step:   values F32[C], z_derivative F64[C]

Run: `PYTHONPATH=src .venv/bin/python scripts/make_derivative_vectors.py`
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decision_layer_arms import zstat  # noqa: E402

VECTORS = ROOT / "flight" / "test" / "vectors"
MAGIC = b"SNTF"
VERSION = 1
SPAN = 2100
STEPS = SPAN + 200


def series(tier: str, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(STEPS, dtype=np.float64)
    out = rng.normal(0.0, 1.0, (STEPS, channels))
    if tier == "f1":
        out[:, 0] = np.sin(t / 40.0)                 # smooth: small derivatives
        out[900:905, 1] += 20.0                      # a step change
        # channel 2 is left as noise
    else:
        out[:, 0] = np.cumsum(rng.normal(0.0, 0.1, STEPS))   # a drift
        out[:, 1] = 3.0                                       # flat: dx is zero
        out[1500:1502, 2] += 40.0                             # a spike and its return
        out[:, 3] = np.sign(np.sin(t / 60.0))                 # a square wave
    return np.ascontiguousarray(out, dtype=np.float32)


def reference(values: np.ndarray) -> np.ndarray:
    steps, channels = values.shape
    out = np.zeros((steps, channels), dtype=np.float64)
    for c in range(channels):
        x = values[:, c].astype(np.float64)
        out[:, c] = zstat(np.abs(np.diff(x, prepend=x[0])), SPAN)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=VECTORS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for tier, channels, seed in (("f1", 3, 31), ("f2", 6, 32)):
        rng = np.random.default_rng(seed)
        values = series(tier, channels, rng)
        z = reference(values)
        blob = bytearray(MAGIC + struct.pack("<HHII", VERSION, channels, STEPS, SPAN))
        for t in range(STEPS):
            blob += values[t].astype("<f4").tobytes()
            blob += z[t].astype("<f8").tobytes()
        (args.out / f"{tier}.fvec").write_bytes(bytes(blob))
        manifest[tier] = {
            "n_channels": channels, "steps": STEPS, "span": SPAN, "seed": seed,
            "vector_bytes": len(blob),
            "provenance": "scripts/make_derivative_vectors.py, reference "
                          "scripts/decision_layer_arms.py:224",
        }
        print(f"  {tier}: {channels:2d} ch x {STEPS} steps  {len(blob):>9,} B")

    (args.out / "derivative_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  total {sum(m['vector_bytes'] for m in manifest.values()):,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
