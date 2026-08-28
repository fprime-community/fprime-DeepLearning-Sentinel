#!/usr/bin/env python3
"""Refit one detector on one fold with another seed, and score that fold the harness's way.

docs/MODELS.md section 15. Work item 5's headline -- seven weak events recovered
on fold 0 -- has two readings: the cell, or the one optimisation path the LSTM's
banked fold-0 fit took (epoch 14, best 3, forty times the MSE of its other
folds). One fit with the seed changed and nothing else separates them.

Everything here is the harness's own path: the `Context` and the fold as
`harness._score_fold` builds them, the same `train_mask`, the same threshold
rule. Reach per event is added beside the harness's `FoldResult`, computed
from the cached errors the score already produced. Nothing published moves:
the detector is built with `Hyper(seed=N)`, whose weight-cache key and
fingerprint differ from the banked seed-0 fit's, so the store gains one file
per set and nothing else is touched.

    python scripts/reseed_fold.py --task m1-g8.9.10 --detector lstm-quantile --seed 1 --fold 0 --both
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import bundle as bundle_mod                         # noqa: E402
from sentinel_eval import harness, ops, read, splits, tasks            # noqa: E402
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


def members(primary, catalog, labels, loaded, both: bool):
    out = []
    for task_id in tasks.paired_with(primary.id):
        if task_id != primary.id and not both:
            continue
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


def reach(scores, threshold, lo, hi):
    """Peak score over ``[lo, hi)`` as a fraction of the threshold; None if nothing scorable."""
    peak = scores[lo:hi]
    peak = peak[np.isfinite(peak)]
    if peak.size == 0 or threshold is None or threshold <= 0:
        return None
    return float(peak.max() / threshold)


def per_event(det, view, fold, threshold):
    """Every anomaly in the fold's test window: caught, and its reach. Reads the cached errors."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    raw = det.score(view.values[window_lo:test_hi], view.valid[window_lo:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - window_lo)
    scores = scores[warmup:]
    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    scores = np.where(scorable, scores, -np.inf)
    predicted = scores >= threshold
    rows = []
    for event in view.truth.events:
        if event.category != ANOMALY or event.event_id not in view.truth.spans:
            continue
        lo, hi = view.truth.spans[event.event_id]
        if not (test_lo <= lo < test_hi):
            continue
        a, b = lo - test_lo, min(hi - test_lo, scores.shape[0])
        if b <= a:
            continue
        rows.append({"event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                     "footprint": int(hi - lo), "caught": bool(predicted[a:b].any()),
                     "reach": reach(scores, threshold, a, b)})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--detector", default="lstm-quantile")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--both", action="store_true", help="also the paired set's same fold")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    hyper = replace(registry.build(args.detector).hyper, seed=args.seed)
    print(f"  Reseed -- {primary.id} fold {args.fold}: {args.detector} with seed {args.seed} "
          f"(cell {hyper.cell}); banked rows untouched")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded, args.both):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        fold = split.folds[args.fold]
        det = registry.build(args.detector, hyper=hyper)
        print(f"    [{task.id}] fold {fold.index}: fit {fold.train_steps:,} -> score "
              f"{fold.test_steps:,} ...", flush=True)
        result = harness._score_fold(view, fold, det, task, beta=0.5, sweep=False)
        events = per_event(det, view, fold, result.threshold)
        caught = sum(r["caught"] for r in events)
        print(f"      threshold {result.threshold:.5f}   training "
              f"{result.training['epochs_run']} epochs, best {result.training['best_epoch']}, "
              f"val MSE {result.training['best_validation_mse']:.3e}   "
              f"events caught {caught}/{len(events)}", flush=True)
        results[task.id] = {"fold": result.as_dict(), "fingerprint": det.fingerprint(),
                            "params": det.params, "events": events}
        D.clear_caches()

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    expected = before + len(results)
    if after != expected:
        print(f"  ABORT: weight store {before} -> {after}, expected {expected}."); return 6
    print(f"  weight store {before} -> {after}: one new fit per set, nothing else")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-reseed-{args.detector}-seed{args.seed}.json"
    path.write_text(json.dumps({"task": primary.id, "detector": args.detector,
                                "seed": args.seed, "fold": args.fold,
                                "hyper": hyper.as_dict(), "device": "cpu",
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
