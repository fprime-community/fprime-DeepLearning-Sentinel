"""Would a limit check have seen it at all? A proxy, and it errs against us.

Lead time in this project is measured against the **labelled event start** -- a
hindsight annotation written by an operations engineer after the fact, not a
limit trip (Objective.md 1.1). The quantity a mission actually cares about is the
gap between our warning and the moment a value becomes visible to the limit
checker it already has, and ESA-ADB carries no dictionary limits, so that cannot
be computed directly.

What can be computed is a **proxy**: for each anomaly the detector catches,
compare our honest emission point against the first moment any watched channel
leaves its own historical nominal envelope. Three outcomes:

1. **we fire first** -- measured advance, in timesteps
2. **the envelope breaks first** -- we are late even by this proxy
3. **the value never leaves its envelope at all** -- the event is invisible to
   any envelope test, so it is caught by us or by nothing

**(!) Outcome 3 is conservative, and that is what makes it quotable.** A real
RED/YELLOW limit sits **outside** a channel's historical operating range, so
"never left its historical envelope" is a **weaker** bar than "never tripped a
limit". Every event in outcome 3 is therefore invisible to any real limit check
*a fortiori*, and the count **undercounts** the invisible class rather than
flattering it. A proxy that errs against the thing it is testing is the only kind
worth putting in front of a review board.

**The envelope is built from fitting data only.** Per channel, over the fold's
fitting window with `train_mask` applied -- the same normal-only construction the
forecaster itself is fitted on, with annotated anomalies, gaps and invalid
segments removed. No test-window sample and no test-side label reaches it, which
is the standard the calibration pool is held to (`docs/NARRATIVE.md` section 6).

Two readings, as agreed. The **0.1 / 99.9 quantile pair** is the primary: a hard
min/max is brittle to a single outlier, and building the proxy on a brittleness
we criticise in limit checking elsewhere would be incoherent. The **hard min/max**
is reported beside it as the literal historical range, and **where the two
disagree that disagreement is itself the finding**.

At C=12 the envelope is 24 floats for the quantile pair and 24 more for min/max
-- against 91,640 model parameters, it is free, and a mission that ships it gets
this proxy for nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import bundle as bundle_mod                         # noqa: E402
from sentinel_eval import ops, read, splits, tasks                     # noqa: E402
from sentinel_eval.catalog import Catalog                              # noqa: E402
from sentinel_eval.detector import Context                             # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402

ANOMALY = "Anomaly"
LOW, HIGH = 0.001, 0.999          # the 0.1 / 99.9 quantile pair


def load(task, args):
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    source = read.R2Source(client, cfg.bucket)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=task.selection.resolve(catalog))
    return loaded, labels, catalog, client, budget, (cfg, ledger)


def members(primary, catalog, labels, loaded):
    out = []
    for task_id in tasks.paired_with(primary.id):
        task = tasks.get(task_id)
        if task_id != primary.id:
            task = type(task)(**{**vars(task), "split": primary.split,
                                 "split_params": primary.split_params})
        view = loaded if task_id == primary.id else loaded.subset(
            task.selection.resolve(catalog), labels)
        out.append((task, view))
    return out


def envelope(values: np.ndarray, usable: np.ndarray) -> dict:
    """Per-channel nominal envelope, from the fitting window's usable rows only.

    ``usable`` is `train_mask` intersected with what is observed: annotated
    anomalies, communication gaps and invalid segments are already out. Nothing
    from the test window and no test-side label reaches this.
    """
    rows = np.asarray(values)[np.asarray(usable, dtype=bool)]
    finite = np.isfinite(rows).all(axis=1)
    rows = rows[finite]
    if rows.shape[0] < 100:
        raise SystemExit("too few nominal rows to build an envelope")
    return {"low": np.quantile(rows, LOW, axis=0),
            "high": np.quantile(rows, HIGH, axis=0),
            "min": rows.min(axis=0), "max": rows.max(axis=0),
            "rows": int(rows.shape[0])}


def first_break(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int | None:
    """First timestep at which any channel sits outside ``[lo, hi]``."""
    outside = (values < lo) | (values > hi)
    outside &= np.isfinite(values)
    any_channel = outside.any(axis=1)
    hits = np.flatnonzero(any_channel)
    return int(hits[0]) if hits.size else None


def context_for(view, fold, window):
    return Context(mission=view.mission, channels=view.channel_ids, groups=view.groups,
                   period_seconds=view.provenance["grid_period_seconds"],
                   fold=fold.index, window=window, commands=None,
                   command_ids=view.command_ids)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--detector", default="lstm-whitened")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    print(f"  Envelope proxy -- {primary.id} / {args.detector}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    from sentinel_models import registry
    results, envelopes = {}, {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        rows, env_rows = [], []
        for fold in split.folds:
            train_lo, train_hi = fold.train
            test_lo, test_hi = fold.test
            usable = train_mask(fold, view.truth)[train_lo:train_hi]
            fitting = context_for(view, fold, fold.train)

            env = envelope(view.values[train_lo:train_hi], usable)
            env_rows.append({
                "fold": fold.index, "fitting_window": [int(train_lo), int(train_hi)],
                "nominal_rows": env["rows"],
                "low": [float(v) for v in env["low"]],
                "high": [float(v) for v in env["high"]],
                "min": [float(v) for v in env["min"]],
                "max": [float(v) for v in env["max"]],
            })

            det = registry.build(args.detector)
            det.fit(view.values[train_lo:train_hi], usable, fitting)
            det.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi],
                      fitting)

            warmup = min(det.warmup_steps, test_lo)
            window_lo = test_lo - warmup
            scored = context_for(view, fold, (test_lo, test_hi))
            values = view.values[window_lo:test_hi]
            raw = det.score(values, view.valid[window_lo:test_hi], scored)
            predicted = np.asarray(raw[warmup:] >= det.threshold_from(None), dtype=bool)
            emission = getattr(det, "last_emission", None)
            emission = (np.asarray(emission, dtype=bool)[warmup:]
                        if emission is not None else predicted)

            test = np.asarray(view.values[test_lo:test_hi])
            scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
            predicted = predicted & scorable
            ranges = mask_to_ranges(predicted)

            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a, b = lo - test_lo, min(hi - test_lo, test.shape[0])
                if b <= a:
                    continue
                hits = [(x, y) for x, y in ranges if not (y <= a or x >= b)]
                if not hits:
                    continue                       # not caught; nothing to compare
                start = min(x for x, _ in hits)
                inside = np.flatnonzero(emission[start:])
                emit = start + int(inside[0]) if inside.size else start

                span = test[a:b]
                out = {}
                for name, low, high in (("quantile", env["low"], env["high"]),
                                        ("minmax", env["min"], env["max"])):
                    brk = first_break(span, low, high)
                    if brk is None:
                        out[name] = {"outcome": 3, "gap": None}
                    else:
                        gap = (a + brk) - emit      # >0 we fired first
                        out[name] = {"outcome": 1 if gap > 0 else 2, "gap": int(gap)}
                rows.append({
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "footprint": int(hi - lo), "emission": int(emit - a),
                    **{f"{k}_{f}": v[f] for k, v in out.items() for f in ("outcome", "gap")},
                })
            D.clear_caches()
        results[task.id] = rows
        envelopes[task.id] = env_rows
        for name in ("quantile", "minmax"):
            counts = {o: sum(1 for r in rows if r[f"{name}_outcome"] == o) for o in (1, 2, 3)}
            print(f"    {task.id:12s} {name:9s} caught {len(rows):3d}  "
                  f"we-first {counts[1]:3d}  envelope-first {counts[2]:3d}  "
                  f"NEVER-BREAKS {counts[3]:3d}", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-envelope-{args.detector}.json"
    path.write_text(json.dumps({
        "task": primary.id, "detector": args.detector,
        "envelope_quantiles": [LOW, HIGH],
        "envelope_provenance": ("per channel, from the fold's FITTING window with "
                                "train_mask applied -- annotated anomalies, gaps and "
                                "invalid segments removed. No test-window sample and "
                                "no test-side label reaches it."),
        "conservatism": ("A real RED/YELLOW limit sits OUTSIDE a channel's historical "
                         "range, so 'never left its envelope' is a weaker bar than "
                         "'never tripped a limit'. Outcome 3 therefore UNDERCOUNTS the "
                         "class invisible to limit checking, a fortiori."),
        "model_bin_cost_floats": {"quantile_pair": 2 * len(loaded.channels),
                                  "min_max": 2 * len(loaded.channels)},
        "operations": budget.as_dict() if budget else None,
        "envelopes": envelopes, "sets": results,
    }, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")

    try:
        if budget is not None:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
            print(); print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      the operations were spent and are NOT recorded; "
              "the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
