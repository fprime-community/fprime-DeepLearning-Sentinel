"""Generate the `DynamicThreshold` golden vectors, tiers D1-D2.

`flight/src/DynamicThreshold.cpp` is held to
`src/sentinel_models/flight_reference.py`, which states the streamed rule in
NumPy and is itself pinned to `telemanom.channel_ratios` by
`tests/test_flight_reference.py` -- with backward dilation restored the two
produce byte-identical emission arrays, so the streaming is a restatement and not
a second opinion.

**Forward-only dilation** (`docs/MODELS.md` 39.5): what these vectors carry is
the flight rule, not the published one. The published one cannot be emitted.

Tiers:

  D1   3 channels    the arithmetic, and a channel that never crosses
  D2  12 channels    the flown width, with events of different shapes: a long
                     sustained excursion, a short sharp one, and a channel whose
                     errors are pure noise so its sweep finds nothing and the
                     rule must stay silent rather than guess

Format, little-endian, the lean shape 39.7 registers -- no hidden-state trace:

    off  size          field
      0     4          magic 'SNTD'
      4     2          version U16 = 1
      6     2          n_channels U16
      8     4          steps U32
     12     4          span U32     = error_window, 2100
     16     4          stride U32   = 70
     20     4          solve U32    = span + stride, 2170
    ...    per step:   e_s F32[C], eps_held F64[C], emitted U8

Run: `PYTHONPATH=src .venv/bin/python scripts/make_threshold_vectors.py`
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_models.flight_reference import emissions      # noqa: E402
from sentinel_models.telemanom import Config                # noqa: E402

VECTORS = ROOT / "flight" / "test" / "vectors"
MAGIC = b"SNTD"
VERSION = 1
STEPS = 70 * 40          # 40 whole segments, past the 2,170 solve window


def smoothed(tier: str, channels: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed-error series, which is what the threshold consumes."""
    out = np.abs(rng.normal(0.0, 1.0, (STEPS, channels)))
    if tier == "d1":
        out[900:1000, 0] *= 14.0                    # a long sustained excursion
        out[1500:1504, 1] *= 9.0                    # a short sharp one
        # channel 2 is left as noise: its sweep must find nothing
    else:
        out[:, 0] *= 30.0                           # a loud channel throughout
        out[1200:1400, 1] *= 12.0                   # sustained
        out[2000:2006, 2] *= 8.0                    # sharp
        out[600:640, 5] *= 5.0                      # modest and wide
        out[:, 7] = 0.5                             # dead: sigma is exactly zero
    return np.ascontiguousarray(out, dtype=np.float32)


def reference(values: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    steps, channels = values.shape
    eps = np.zeros((steps, channels), dtype=np.float64)
    emitted = np.zeros(steps, dtype=bool)
    for c in range(channels):
        em, held = emissions(values[:, c], config, forward_only=True)
        eps[:, c] = held
        emitted |= em
    return eps, emitted


def write(path: Path, values, eps, emitted, config: Config) -> int:
    steps, channels = values.shape
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<HHIIII", VERSION, channels, steps,
                        config.error_window, config.stride,
                        config.error_window + config.stride)
    for t in range(steps):
        blob += values[t].astype("<f4").tobytes()
        blob += eps[t].astype("<f8").tobytes()
        blob += struct.pack("<B", 1 if emitted[t] else 0)
    path.write_bytes(bytes(blob))
    return len(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=VECTORS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = Config()
    manifest = {}
    for tier, channels, seed in (("d1", 3, 21), ("d2", 12, 22)):
        rng = np.random.default_rng(seed)
        values = smoothed(tier, channels, rng)
        eps, emitted = reference(values, config)
        size = write(args.out / f"{tier}.dvec", values, eps, emitted, config)
        manifest[tier] = {
            "n_channels": channels, "steps": STEPS, "seed": seed,
            "span": config.error_window, "stride": config.stride,
            "solve_window": config.error_window + config.stride,
            "emissions": int(emitted.sum()), "vector_bytes": size,
            "dilation": "forward only (docs/MODELS.md 39.5)",
            "provenance": "scripts/make_threshold_vectors.py, reference "
                          "src/sentinel_models/flight_reference.py",
        }
        print(f"  {tier}: {channels:2d} ch x {STEPS} steps  "
              f"{int(emitted.sum()):3d} emissions  {size:>9,} B")

    (args.out / "threshold_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  total {sum(m['vector_bytes'] for m in manifest.values()):,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
