"""Tier 1 of the acceptance ladder: the generated fixture, end to end, zero ops.

`docs/MODELS.md` 40.9 registers three rungs -- minimal fixture, single channel,
full mission. This is the first, and the second is the same path at
`n_channels` 1. The third is a real fit whose wall clock is measured before it
is started (40.13), and it is **not run from here**.
"""
from __future__ import annotations

import numpy as np

from .errors import ToolkitError
from .fit import fit_model
from .limits import FLIGHT_LIMITS

#: Small deliberately. The window, hidden width and horizon are model-file
#: fields, so the flight component honours whatever is written; only the
#: trailing span is fixed in the component, and that is not negotiable here.
FIXTURE = dict(window=50, hidden=(16, 16), n_predictions=5, max_epochs=2)


def healthy_run(seed: int = 0, steps: int = 120_000) -> tuple[np.ndarray, tuple[str, ...]]:
    """The longest contiguous anomaly-free run of the generated fixture.

    Contiguous because the derivative stream is `|x[t] - x[t-1]|`: stitching
    non-adjacent healthy segments together would invent a step change at every
    join and calibrate the detector against its own splicing.
    """
    from sentinel_eval import bundle as bundle_mod, read, synthetic, tasks
    from sentinel_eval.catalog import Catalog
    from sentinel_eval.labels import LabelSet

    source = synthetic.build(seed=seed, n=steps)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    task = tasks.get("synthetic")
    ids = task.selection.resolve(catalog)
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=ids, telecommands=None)

    clean = ~loaded.truth.anomaly & np.isfinite(loaded.values).all(axis=1)
    edges = np.diff(np.concatenate([[0], clean.view(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not runs:
        raise ToolkitError("the fixture produced no anomaly-free run at all")
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return loaded.values[lo:hi], tuple(ids)


def run(seed: int = 0, steps: int = 120_000, log=print) -> int:
    from sentinel_export import Status, read_model

    values, names = healthy_run(seed=seed, steps=steps)
    log("  SELFTEST -- generated fixture, zero cloud operations\n")
    log(f"    fixture: longest healthy run {len(values):,} timesteps "
        f"x {values.shape[1]} channels")

    result = fit_model(values, channel_names=list(names), mission="fixture",
                       seed=seed, log=log, **FIXTURE)
    status, spec = read_model(result.model_bytes, FLIGHT_LIMITS)
    params = spec["params"] if spec else {}
    cal = result.calibration

    checks = [
        ("the model loads through the flight-mirroring reader",
         status is Status.OK, status.name),
        ("it is a version-2 file, so the fused rule is the one flown",
         params.get("param_version") == 2, f"param_version {params.get('param_version')}"),
        ("the tier is written rather than inherited",
         params.get("tier") == 3, f"tier {params.get('tier')}"),
        ("the threshold survives the round trip exactly",
         params.get("threshold") == cal.cut, f"{params.get('threshold'):.6f}"),
        ("provenance is recorded, not left blank",
         bool(params.get("provenance", "").strip()), repr(params.get("provenance", ""))),
        ("the warm-up covers the trailing span the component compiles in",
         params.get("warmup_steps", 0) >= 2100, f"{params.get('warmup_steps', 0):,}"),
        ("the cut was derived on data the forecaster never trained on",
         cal.fit_steps > 0 and cal.holdout_steps > 0,
         f"{cal.fit_steps:,} fit / {cal.holdout_steps:,} held out"),
        ("the held-out rate is within 2x of the calibration half (T6's band)",
         0.5 <= cal.ratio <= 2.0,
         f"{100 * cal.fit_rate:.4f}% -> {100 * cal.holdout_rate:.4f}%, "
         f"{cal.ratio:.2f}x"),
    ]
    log("")
    for label, ok, detail in checks:
        log(f"    {'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    failures = sum(1 for _, ok, _ in checks if not ok)
    log(f"\n  {len(checks) - failures}/{len(checks)} checks passed")

    from .report import render
    log("")
    log(render(cal, channels=values.shape[1], train_steps=result.train_steps,
               window=FIXTURE["window"], hidden=FIXTURE["hidden"],
               n_predictions=FIXTURE["n_predictions"],
               model_bytes=len(result.model_bytes), param_version=2, tier=3))
    return 1 if failures else 0
