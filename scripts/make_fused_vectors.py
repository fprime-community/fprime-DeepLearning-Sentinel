"""Generate the D68 flight-configuration vectors, tier P1.

D68 adopts the fused rule: `emitted` comes from `max(z_residual, z_derivative)`
against one calibrated cut -- D65's arm 2, the one that reached **EVAL 17 of 19
where the frozen arm reached 4**, reproduced on two independent reads. The PARAMS
block that carries that cut is **`param_version` 2**; the byte layout is
identical to version 1 and only the statistic the threshold cuts is different, so
`format_version` stays 1 and D30's freeze is untouched.

**This tier is the end-to-end evidence for that.** It writes a real `model.bin`
at `param_version` 2 and pins what a `Detector` loaded from it emits, step by
step -- not the streams in isolation, which `t*.tvec`, `d*.dvec` and `f*.fvec`
already cover, but the whole path from bytes on disk to a flag.

It also pins the thing that makes the version safe: a **version-1** file and a
**version-2** file with the *same threshold value* produce **different** flags,
because they cut different statistics. If that ever stopped being true the
version would be decoration.

Format, little-endian:

    off  size          field
      0     4          magic 'SNTP'
      4     2          version U16 = 1
      6     2          n_channels U16
      8     4          steps U32
     12     8          threshold F64      the fused cut in the paired model.bin
    ...    per step:   values F32[C], fused F64, crossing U8, emitted U8

Run: `PYTHONPATH=src .venv/bin/python scripts/make_fused_vectors.py`
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_export import format as fmt          # noqa: E402
from sentinel_export.writer import write_model     # noqa: E402


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


GV = _load("make_golden_vectors", "scripts/make_golden_vectors.py")
ARMS = _load("decision_layer_arms", "scripts/decision_layer_arms.py")

VECTORS = ROOT / "flight" / "test" / "vectors"
MAGIC = b"SNTP"
VERSION = 1
SPAN = 2100
STEPS = 3200            # 2,350 of warm-up, then 850 steps where the flag can
                        # actually be false or true. At 2,400 only five steps
                        # cleared the warm-up and 'exact on the flag' would have
                        # been an assertion about almost nothing.
WARMUP = 2350           # window 250 + error_window 2100, as the format has it


def fused_trace(weights, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(smoothed, fused)` for the whole run, as the core computes them.

    One continuous stream, not `build_trace`'s three-phase scenario: that one is
    fixed at 72 steps to exercise a chunk boundary and a reset, and this tier
    needs 2,400 to reach past the 2,100 standardising window. A `Detector`
    stepped 2,400 times without a reset is exactly this.
    """
    _hidden, _head, predictions, _state = GV.run_stream(weights, x)
    _forecast, _residual, smoothed, _score = GV.decide(x, predictions)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    steps, channels = smoothed.shape
    fused = np.zeros((steps, channels), dtype=np.float64)
    for c in range(channels):
        z_r = ARMS.zstat(smoothed[:, c], SPAN)
        z_d = ARMS.zstat(np.abs(np.diff(x[:, c].astype(np.float64),
                                        prepend=float(x[0, c]))), SPAN)
        fused[:, c] = np.maximum(z_r, z_d)
    return smoothed, fused


def build_tier(out: Path, label: str, channels: int, seed: int) -> dict:
    """One tier. `p1` is the fixture's full width; `p2` is `n_channels` 1, which
    40's T1 asks for and which no tier covered until 59 (MW8)."""
    hidden, predictions = [4, 4], 4
    weights = GV.seeded_weights(channels, hidden, predictions, seed=seed)
    x = GV.seeded_input(STEPS, channels, seed=seed)

    _, fused = fused_trace(weights, x)
    score = fused.max(axis=1)

    # A cut that the settled run crosses on some steps and not others, so
    # "exactly on the flag" is a real assertion rather than a constant.
    settled = score[WARMUP:]
    threshold = float(np.quantile(settled, 0.90))

    crossing = score >= threshold
    emitted = crossing & (np.arange(STEPS) >= WARMUP)

    # `param_version` is an argument rather than a patch after the fact: the
    # writer defaults it to 1, and a fused z-score cut in a version-1 file is
    # the silent drift `docs/MODEL_FILE.md` 6.2 forbids (`docs/MODELS.md` 40.6).
    spec = GV.to_spec(weights, threshold, WARMUP,
                      f"D68 fused max(z_residual, z_derivative); seeded tier {label.upper()}",
                      param_version=fmt.PARAM_VERSION_FUSED)
    (out / f"{label}.bin").write_bytes(write_model(spec))

    blob = bytearray(MAGIC + struct.pack("<HHI", VERSION, channels, STEPS)
                     + struct.pack("<d", threshold))
    for t in range(STEPS):
        blob += x[t].astype("<f4").tobytes()
        blob += struct.pack("<d", float(score[t]))
        blob += struct.pack("<BB", int(crossing[t]), int(emitted[t]))
    (out / f"{label}.pvec").write_bytes(bytes(blob))

    entry = {
        "n_channels": channels, "steps": STEPS, "span": SPAN, "seed": seed,
        "param_version": fmt.PARAM_VERSION_FUSED, "threshold": threshold,
        "warmup_steps": WARMUP, "crossings": int(crossing.sum()),
        "emitted": int(emitted.sum()), "vector_bytes": len(blob),
        "model_bytes": len((out / f"{label}.bin").read_bytes()),
        "provenance": "scripts/make_fused_vectors.py; D68's flight configuration",
    }
    print(f"  {label}: {channels} ch x {STEPS} steps  threshold {threshold:.6f}  "
          f"{int(crossing.sum())} crossings, {int(emitted.sum())} emitted  "
          f"{len(blob):,} B + {entry['model_bytes']:,} B model")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=VECTORS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "p1": build_tier(args.out, "p1", 3, 68),
        "p2": build_tier(args.out, "p2", 1, 69),
    }
    (args.out / "fused_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
