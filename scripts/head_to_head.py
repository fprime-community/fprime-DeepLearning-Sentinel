"""Which events does each of two detectors catch, and which only one of them.

`docs/DECISIONS.md` D24 froze `lstm-whitened` without ever scoring
`lstm-quantile` on post-fix weights -- that branch was closed in D13 on
one-epoch models, and its reasoning has since been withdrawn. Re-measured, the
two are close on every headline number, so the decision turns on what the
scorecard cannot show: **whether they catch the same events**.

Two detectors that score alike and catch the same events are interchangeable and
the choice is made on flight simplicity. Two that score alike and catch
*different* events are not interchangeable at all, and the overlap is the finding
rather than the totals.

Also answers the eleven-weak-events question directly: `lstm-whitened` loses
eleven anomalies `lstm-telemanom` catches, and whether `lstm-quantile` recovers
any of them is not deducible from a recall count.

Cached weights, no refit, both channel sets, all folds.
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
from sentinel_eval.detector import Context, reduce_scores              # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402

ANOMALY = "Anomaly"


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


def context_for(view, fold, window):
    return Context(mission=view.mission, channels=view.channel_ids, groups=view.groups,
                   period_seconds=view.provenance["grid_period_seconds"],
                   fold=fold.index, window=window, commands=None,
                   command_ids=view.command_ids)


def alarms(name, view, fold):
    """One detector's alarm mask over a fold's test window. The harness's own path."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(name)
    det.fit(view.values[train_lo:train_hi], usable, fitting)
    train_raw = det.score(view.values[train_lo:train_hi],
                          view.valid[train_lo:train_hi], fitting)
    train_scores, _ = reduce_scores(train_raw, train_hi - train_lo)
    threshold = det.threshold_from(train_scores[usable] if usable.any() else train_scores)

    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    raw = det.score(view.values[window_lo:test_hi],
                    view.valid[window_lo:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - window_lo)
    predicted = np.asarray(scores[warmup:] >= threshold, dtype=bool)
    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    D.clear_caches()
    return predicted & scorable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--a", default="lstm-whitened")
    parser.add_argument("--b", default="lstm-quantile")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    print(f"  Head to head -- {primary.id}:  {args.a}  vs  {args.b}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        rows = []
        for fold in split.folds:
            test_lo, test_hi = fold.test
            masks = {n: alarms(n, view, fold) for n in (args.a, args.b)}
            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a_, b_ = lo - test_lo, min(hi - test_lo, masks[args.a].shape[0])
                if b_ <= a_:
                    continue
                rows.append({
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "footprint": int(hi - lo),
                    args.a: bool(masks[args.a][a_:b_].any()),
                    args.b: bool(masks[args.b][a_:b_].any()),
                })
            print(f"    {task.id} fold {fold.index} done", flush=True)
        results[task.id] = rows
        both = [r for r in rows if r[args.a] and r[args.b]]
        only_a = [r for r in rows if r[args.a] and not r[args.b]]
        only_b = [r for r in rows if r[args.b] and not r[args.a]]
        neither = [r for r in rows if not r[args.a] and not r[args.b]]
        print(f"    {task.id:12s} n={len(rows)}  both {len(both)}  "
              f"only-{args.a} {len(only_a)}  only-{args.b} {len(only_b)}  "
              f"neither {len(neither)}", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-head-to-head.json"
    path.write_text(json.dumps({"task": primary.id, "a": args.a, "b": args.b,
                                "operations": budget.as_dict() if budget else None,
                                "sets": results}, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")
    try:
        if budget is not None:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
            print(); print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      operations spent and NOT recorded; the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
