"""Generate the `TrailingWindow` golden vectors, tiers T1-T3.

`flight/src/TrailingWindow.cpp` is held to `scripts/decision_layer_arms.py`'s
`trailing_stats` (`:53-58`), which is the reference `docs/MODELS.md` 39.2 names
for the trailing half of the new path. The two compute the same window by
different means and that is the point of the comparison: the reference differences
float64 prefix sums over the whole series, the flight core carries a running
window and subtracts the departing sample. They must agree to 1e-5 (N1).

**The window is 2,100, so a vector shorter than that never exercises a wrap.**
Every tier runs past it deliberately, which is most of what these files cost.

Tiers, and each exists for a reason:

  T1   3 channels, smooth and small        the arithmetic, cheaply
  T2   6 channels                          mixed scales and a channel that dies.
                                           Six rather than twelve on purpose: T3
                                           already covers the flown width, and a
                                           second tier at the same width buys
                                           coverage nobody has and 276 KB of
                                           tracked content somebody pays for
  T3  12 channels, N(1000, 3) float32      D37's regime. A large offset with small
                                           variation is where differencing two
                                           big sums to recover a small second
                                           moment loses everything -- 7.6584e+00
                                           of error against a true sigma of 3.0,
                                           measured at `baselines.py:39-51`. If
                                           anyone reintroduces F32 accumulation
                                           this tier fails loudly and the others
                                           do not.

Format, little-endian, matching the lean shape `docs/MODELS.md` 39.7 registers --
no hidden-state trace, because the forward pass is already pinned at 1e-5 by the
`.vec` tiers:

    off  size          field
      0     4          magic 'SNTT'
      4     2          version U16 = 1
      6     2          n_channels U16
      8     4          steps U32
     12     4          span U32          must equal Config::ERROR_WINDOW
     16     C*4        scale F32[C]      per-channel input scale, for the record
    ...    per step:   values F32[C], mean F64[C], sd F64[C]

`z` is not stored: it is `(value - mean) / max(sd, 1e-12)` and the test derives it,
so the file carries the two quantities the core actually keeps.

Run: `PYTHONPATH=src .venv/bin/python scripts/make_trailing_vectors.py`
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

from decision_layer_arms import trailing_stats  # noqa: E402

VECTORS = ROOT / "flight" / "test" / "vectors"
MAGIC = b"SNTT"
VERSION = 1
SPAN = 2100          # Config::ERROR_WINDOW, and telemanom.py:84-85's 70 x 30

#: 200 steps past the wrap: enough that every slot has been overwritten once and
#: the running accumulators have subtracted a full window's worth of samples.
STEPS = SPAN + 200


def series(tier: str, channels: int, rng: np.random.Generator) -> np.ndarray:
    """Inputs per tier. float32, because that is what a bundle carries."""
    if tier == "t1":
        t = np.arange(STEPS, dtype=np.float64)
        out = np.stack([np.sin(t / 50.0), np.cos(t / 31.0), t / STEPS], axis=1)
    elif tier == "t2":
        out = rng.normal(0.0, 1.0, (STEPS, channels))
        out[:, 0] *= 50.0                       # a loud channel
        out[:, 1] = 0.25                        # a dead one: sd is exactly zero
        out[:, 2] = np.linspace(-3.0, 3.0, STEPS)
    else:                                        # t3, D37's regime
        out = rng.normal(1000.0, 3.0, (STEPS, channels))
    return np.ascontiguousarray(out[:, :channels], dtype=np.float32)


def reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel trailing mean and sd, from the reference implementation."""
    steps, channels = values.shape
    mean = np.zeros((steps, channels), dtype=np.float64)
    sd = np.zeros((steps, channels), dtype=np.float64)
    for c in range(channels):
        mu, spread = trailing_stats(values[:, c], SPAN)
        mean[:, c], sd[:, c] = mu, spread
    return mean, sd


def write(path: Path, values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> int:
    steps, channels = values.shape
    scale = np.abs(values).max(axis=0).astype(np.float32)
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<HHII", VERSION, channels, steps, SPAN)
    blob += scale.tobytes()
    for t in range(steps):
        blob += values[t].astype("<f4").tobytes()
        blob += mean[t].astype("<f8").tobytes()
        blob += sd[t].astype("<f8").tobytes()
    path.write_bytes(bytes(blob))
    return len(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=VECTORS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for tier, channels, seed in (("t1", 3, 11), ("t2", 6, 12), ("t3", 12, 13)):
        rng = np.random.default_rng(seed)
        values = series(tier, channels, rng)
        mean, sd = reference(values)
        size = write(args.out / f"{tier}.tvec", values, mean, sd)
        manifest[tier] = {
            "n_channels": channels, "steps": STEPS, "span": SPAN, "seed": seed,
            "vector_bytes": size,
            "provenance": "scripts/make_trailing_vectors.py, reference "
                          "scripts/decision_layer_arms.py:53-58 trailing_stats",
        }
        print(f"  {tier}: {channels:2d} ch x {STEPS} steps  {size:>9,} B")

    (args.out / "trailing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  total {sum(m['vector_bytes'] for m in manifest.values()):,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
