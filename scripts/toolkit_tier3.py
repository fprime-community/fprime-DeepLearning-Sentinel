"""Tier 3 of the toolkit's acceptance ladder: one real mission, end to end.

`docs/MODELS.md` 40.9's third rung and 40.13's compute gate. The gate was
registered because no wall clock for a mission-scale fit had ever been measured;
it is measured here, on ESA-ADB Mission 1's twelve-channel gate set, at the flown
forecaster shape.

**Cost, projected before it is spent** (`docs/DATA.md` 4):

    1 Class B   the ops ledger        `ops.load` fetches it before anything else
    1 Class B   the manifest          `Catalog.load`
    1 Class B   the labels table      window selection, see below
    N Class B   one whole-object GET per channel object, sharded channels more
    1 Class A   the ledger write back, after the artifact

**(!) The labels are read, and they are a fixture choice rather than a
calibration input.** A deploying mission knows which of its own telemetry is
healthy; ESA-ADB is an archive and this project does not, so the annotation table
picks the window. Nothing it says reaches `fit_model`, which is handed an array
and nothing else -- `tests/test_toolkit.py`'s T5 asserts structurally that the
calibrating modules cannot take a label. The same relationship the generated
fixture's selftest already has with its own truth.

    python scripts/toolkit_tier3.py --limit 2 --smoke     # 5 Class B
    python scripts/toolkit_tier3.py                       # 15 Class B + 1 Class A
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                # noqa: E402
from sentinel_eval import bundle as bundle_mod, ops, read, tasks     # noqa: E402
from sentinel_eval.catalog import Catalog                            # noqa: E402
from sentinel_eval.labels import LabelSet                            # noqa: E402
from sentinel_toolkit.fit import fit_model                           # noqa: E402
from sentinel_toolkit.limits import FLIGHT_LIMITS                    # noqa: E402
from sentinel_export import Status, read_model                       # noqa: E402

TASK = "m1-g8.9.10"
RUNS = ROOT / "runs" / "esa-adb" / "_toolkit"


def longest_clean_run(values: np.ndarray, anomaly: np.ndarray) -> tuple[int, int]:
    """The longest contiguous healthy span. Contiguous because the derivative
    stream is `|x[t] - x[t-1]|`: stitching non-adjacent segments would invent a
    step change at every join and calibrate the detector against its own splicing.
    """
    clean = ~anomaly & np.isfinite(values).all(axis=1)
    edges = np.diff(np.concatenate([[0], clean.view(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not runs:
        raise SystemExit("  REFUSED: no contiguous healthy run in this selection")
    return max(runs, key=lambda r: r[1] - r[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="load only the first N channels; the smoke uses 2")
    ap.add_argument("--smoke", action="store_true",
                    help="load and report the cost, fit nothing")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--epochs", type=int, default=35)
    a = ap.parse_args(argv)

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    print(f"  month so far: {ledger.month_class_a} Class A, {ledger.month_class_b} Class B "
          f"of {C.OPS_CEILING['class_a']:,}")

    source = read.R2Source(client, cfg.bucket)
    catalog = Catalog.load(source)
    task = tasks.get(TASK)
    ids = task.selection.resolve(catalog)

    # Shard counts come from the manifest, which is already fetched, so the
    # projection below costs nothing and is exact rather than assumed. 39.13.4:
    # the one smoke that diverged did so because a fetch was left out of one.
    objects = {cid: len(catalog.channel(task.mission, cid).objects) for cid in ids}
    chosen = ids[: a.limit] if a.limit else ids
    projected = 1 + 1 + 1 + sum(objects[c] for c in chosen)
    print(f"  selection: {len(ids)} channels, objects per channel "
          f"{sorted(set(objects.values()))}")
    print(f"  loading {len(chosen)}: projected {projected} Class B "
          f"(1 ledger + 1 manifest + 1 labels + {sum(objects[c] for c in chosen)} arrays)")

    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    t0 = time.time()
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=chosen, telecommands=None)
    fetch_wall = time.time() - t0
    actual = budget.class_b
    print(f"  actual {actual} Class B, projected {projected}: "
          f"{'MATCH' if actual == projected else '(!) DIVERGED'}")
    print(f"  grid: {len(loaded.grid):,} timesteps x {loaded.values.shape[1]} channels, "
          f"{fetch_wall:.1f}s")

    if actual != projected:
        print("  STOPPING: the projection was wrong, which is what a smoke is for.")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 1

    if a.smoke:
        print("  smoke only; nothing fitted.")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 0

    lo, hi = longest_clean_run(loaded.values, loaded.truth.anomaly)
    healthy = loaded.values[lo:hi]
    print(f"  healthy window: [{lo:,}, {hi:,}) = {len(healthy):,} timesteps "
          f"({100 * len(healthy) / len(loaded.grid):.1f}% of the grid)")

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    result = fit_model(healthy, channel_names=list(chosen), mission=task.mission,
                       max_epochs=a.epochs, seed=0, log=print)
    wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824

    status, spec = read_model(result.model_bytes, FLIGHT_LIMITS)
    cal = result.calibration
    print(f"\n  fit+calibrate  {wall:.1f}s   peak RSS {peak:.2f} GiB "
          f"(+{(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1073741824:.2f})")
    print(f"  cut            {cal.cut:.6f}  at q={cal.quantile:g}, span {cal.span:,}")
    print(f"  sanity rate    {cal.holdout_alarms:,}/{cal.holdout_steps:,} = "
          f"{100 * cal.holdout_rate:.4f}%   (fit half {100 * cal.fit_rate:.4f}%, "
          f"{cal.ratio:.2f}x)")
    print(f"  model.bin      {len(result.model_bytes):,} B, {status.name}, "
          f"param_version {spec['params']['param_version']}")

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    artifact = RUNS / f"{stamp}-tier3-{task.mission}.json"
    artifact.write_text(json.dumps({
        "task": TASK, "mission": task.mission, "channels": list(chosen),
        "grid_steps": len(loaded.grid),
        "healthy_window": [int(lo), int(hi)], "healthy_steps": int(len(healthy)),
        "train_steps": result.train_steps, "scored_steps": result.scored_steps,
        "warmup_steps": result.warmup_steps,
        "epochs_max": a.epochs,
        "wall_seconds": round(wall, 1), "fetch_seconds": round(fetch_wall, 1),
        "peak_rss_gib": round(peak, 2),
        "cut": cal.cut, "quantile": cal.quantile, "span": cal.span,
        "fit_alarms": cal.fit_alarms, "fit_steps": cal.fit_steps,
        "holdout_alarms": cal.holdout_alarms, "holdout_steps": cal.holdout_steps,
        "ratio": round(cal.ratio, 4),
        # (!) The swept curve, which the first run did not record. It is the
        # sensitivity half of the calibration report -- the cut is derived and
        # the curve is REPORTED around it -- and leaving it out of the artifact
        # meant the report could be printed once and never redrawn from a
        # record. A figure that exists only in stdout is not a figure this
        # project has. `docs/HARNESS.md` 1 is why the curve is reported at all.
        "curve": [{"quantile": q, "cut": cut,
                   "fit_alarms": fa, "holdout_alarms": ha,
                   "fit_rate": fa / cal.fit_steps if cal.fit_steps else None,
                   "holdout_rate": ha / cal.holdout_steps if cal.holdout_steps else None}
                  for q, cut, fa, ha in cal.curve],
        "provenance": cal.provenance(),
        "model_bytes": len(result.model_bytes),
        "param_version": spec["params"]["param_version"],
        "operations": budget.as_dict(),
        "producer": "scripts/toolkit_tier3.py",
    }, indent=2) + "\n")
    print(f"  artifact       {artifact.relative_to(ROOT)}")

    if not a.no_ledger:
        ops.commit(client, cfg.bucket, ledger, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
