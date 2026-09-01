#!/usr/bin/env python3
"""Generate the Level 1 baseline golden vectors, tiers B1 to B4.

The work item 8 discipline applied to the baseline (`docs/MODELS.md` 20.7):
seeded inputs so a fresh clone regenerates them byte-identically, a committed
`.vec` per tier, a manifest recording what each tier is, and a C++ binary held to
the Python at 1e-5 with the crossing and emitted flags exact.

Vectors come from `baseline_reference.StreamingBaseline`, which is the
authoritative statement of the flight arithmetic -- deliberately NOT from
`baselines.RollingStd`, whose float32 prefix sums lose the statistic (D37).

    PYTHONPATH=src .venv/bin/python scripts/make_baseline_vectors.py

Zero bucket operations: seeded generators and the offline fixture only.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_models import baseline_reference as br  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "flight" / "test" / "vectors"

MAGIC = b"SNTB"
VERSION = 1
STEPS = 400

#: Not the flight recipe. `detector.py:127-132` calibrates at the 99.9th
#: percentile, which over 400 steps leaves at most one crossing and would pin the
#: flags weakly. These vectors exist to pin arithmetic, so the cut is set where
#: roughly a tenth of the steps cross and both branches are exercised.
VECTOR_QUANTILE = 0.90


def fixture_values(channels: int, steps: int) -> np.ndarray:
    """The offline synthetic fixture. Zero bucket operations."""
    from sentinel_eval import bundle as bundle_mod
    from sentinel_eval import synthetic, tasks
    from sentinel_eval.catalog import Catalog
    from sentinel_eval.labels import LabelSet
    from sentinel_eval.read import read_annotation

    bucket = synthetic.build(seed=0, n=20_000)
    catalog = Catalog.load(bucket)
    labels = LabelSet.from_table(read_annotation(bucket, catalog, "labels"))
    task = tasks.get("synthetic")
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=task.selection.resolve(catalog))
    return np.ascontiguousarray(loaded.values[:steps, :channels])


def build_tier(name: str):
    """Returns (values float32[T, C], valid bool[T], provenance)."""
    if name == "b1":
        rng = np.random.default_rng(11)
        values = rng.standard_normal((STEPS, 3)).astype(np.float32)
        return values, np.ones(STEPS, dtype=bool), "seeded B1, rng(11), unit normal"

    if name == "b2":
        rng = np.random.default_rng(12)
        values = rng.standard_normal((STEPS, 12)).astype(np.float32)
        values[rng.random(values.shape) < 0.02] = np.nan
        valid = np.ones(STEPS, dtype=bool)
        valid[[137, 138, 200, 321]] = False       # pins the -infinity branch
        return values, valid, "seeded B2, rng(12), unit normal, 2% NaN, 4 invalid ticks"

    if name == "b3":
        rng = np.random.default_rng(13)
        values = (rng.standard_normal((STEPS, 12)) * 3.0 + 1000.0).astype(np.float32)
        return values, np.ones(STEPS, dtype=bool), (
            "seeded B3, rng(13), N(1000, 3) -- the regime that exposes the "
            "float32 cancellation D37 records")

    if name == "b4":
        values = fixture_values(7, STEPS)
        return values, np.ones(STEPS, dtype=bool), (
            "the offline synthetic fixture, synthetic.build(seed=0), first "
            f"{STEPS} steps of 7 channels")

    raise ValueError(name)


def calibrate(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-channel scale and a threshold, by the harness's own recipes.

    Scale follows `baselines.py:105` -- the population standard deviation of the
    spread series itself, per channel. In flight this is the `BASELINE_SCALE`
    parameter (D34); here it is computed so the vectors are realistic.
    """
    spread = br.spread(values)
    warm = spread[br.WINDOW:]
    with np.errstate(invalid="ignore"):
        scale = np.nanstd(warm, axis=0)
    scale = np.maximum(scale, br.EPSILON)

    combined = (warm / np.maximum(scale, br.EPSILON)).max(axis=1)
    finite = combined[np.isfinite(combined)]
    threshold = float(np.quantile(finite, VECTOR_QUANTILE))
    return scale.astype(np.float64), threshold


def emit(path: Path, values, valid, scale, threshold) -> dict:
    steps, channels = values.shape
    baseline = br.StreamingBaseline(channels, scale, threshold)

    body = bytearray()
    crossings = 0
    emitted = 0
    for t in range(steps):
        baseline.step(values[t], bool(valid[t]))
        crossings += int(baseline.crossing)
        emitted += int(baseline.emitted)
        body += struct.pack(f"<{channels}f", *values[t].astype(np.float32))
        body += struct.pack(f"<{channels}d", *baseline.channel_scores)
        body += struct.pack("<d", baseline.score)
        body += struct.pack("<4B", int(valid[t]), int(baseline.crossing),
                            int(baseline.emitted), int(baseline.peak_channel))

    header = bytearray()
    header += MAGIC
    header += struct.pack("<HH", VERSION, channels)
    header += struct.pack("<II", steps, br.WINDOW)
    header += struct.pack("<d", threshold)
    header += struct.pack(f"<{channels}d", *scale)

    path.write_bytes(bytes(header) + bytes(body))
    return {"crossings": crossings, "emitted": emitted, "bytes": len(header) + len(body)}


def main() -> int:
    VECTORS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ("b1", "b2", "b3", "b4"):
        values, valid, provenance = build_tier(name)
        scale, threshold = calibrate(values)
        stats = emit(VECTORS / f"{name}.bvec", values, valid, scale, threshold)
        row = {
            "tier": name,
            "n_channels": int(values.shape[1]),
            "steps": int(values.shape[0]),
            "window": br.WINDOW,
            "threshold": threshold,
            "invalid_ticks": int((~valid).sum()),
            "nan_inputs": int(np.isnan(values).sum()),
            "crossings": stats["crossings"],
            "emitted": stats["emitted"],
            "vector_bytes": stats["bytes"],
            "provenance": provenance,
        }
        manifest.append(row)
        print(f"  {name}: {values.shape[0]} steps x {values.shape[1]} ch, "
              f"threshold {threshold:.12g}, {stats['crossings']} crossings, "
              f"{stats['emitted']} emitted, {stats['bytes']} bytes")

    (VECTORS / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest: {len(manifest)} tiers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
