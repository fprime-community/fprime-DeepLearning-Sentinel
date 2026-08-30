#!/usr/bin/env python3
"""Scope the two-forecaster combination on cached weights. Nothing here is a detector.

docs/MODELS.md section 17. Both incumbents scored through the harness's own
path from one bundle load; their scores normalised by their own calibrated
thresholds into reach series; three one-line rules -- OR, AND, MEAN -- scored
by the referee's own metric code on their alarm masks. Per anomaly and per
rare nominal event, each model's peak reach and the other model's reach at
that moment. Artifact before ledger; held-back sets refused; the weight store
must not change.

    python scripts/combination_scope.py --task m1-g8.9.10 --a lstm-quantile --b gru-quantile
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
from sentinel_eval.metrics import eventwise, falsealarm                # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402

ANOMALY = "Anomaly"
RARE = "Rare Event"


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


def reach_series(name, view, fold):
    """One detector's score over the test window divided by its own threshold: reach per step."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(name)
    det.fit(view.values[train_lo:train_hi], usable, fitting)
    train_raw = det.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi], fitting)
    train_scores, _ = reduce_scores(train_raw, train_hi - train_lo)
    threshold = det.threshold_from(train_scores[usable] if usable.any() else train_scores)
    warmup = min(det.warmup_steps, test_lo)
    scored = context_for(view, fold, (test_lo, test_hi))
    raw = det.score(view.values[test_lo - warmup:test_hi], view.valid[test_lo - warmup:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - (test_lo - warmup))
    scores = scores[warmup:]
    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    reach = np.where(scorable, scores / threshold, -np.inf)
    D.clear_caches()
    return reach, float(threshold), scorable


def peak_and_there(r_self, r_other, lo, hi):
    """This model's peak reach in [lo, hi) and the other's reach at that same step."""
    seg = r_self[lo:hi]
    if not np.isfinite(seg).any():
        return None, None
    t = int(np.argmax(np.where(np.isfinite(seg), seg, -np.inf)))
    other = r_other[lo + t]
    return float(seg[t]), (float(other) if np.isfinite(other) else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--a", default="lstm-quantile")
    parser.add_argument("--b", default="gru-quantile")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2
    print(f"  Combination scope -- {primary.id}: {args.a} + {args.b}; OR / AND / MEAN on reach")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    rules = {"OR": lambda a, b: np.maximum(a, b) >= 1.0,
             "AND": lambda a, b: np.minimum(a, b) >= 1.0,
             "MEAN": lambda a, b: (a + b) / 2.0 >= 1.0}
    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        folds_out, anomalies, rares = [], [], []
        for fold in split.folds:
            test_lo, test_hi = fold.test
            r_a, thr_a, scorable = reach_series(args.a, view, fold)
            r_b, thr_b, _ = reach_series(args.b, view, fold)
            truth = view.truth
            events, spans = [], {}
            for e in truth.events:
                if e.event_id in truth.spans:
                    lo, hi = truth.spans[e.event_id]
                    if test_lo <= lo < test_hi:
                        events.append(e); spans[e.event_id] = (lo - test_lo, min(hi, test_hi) - test_lo)
            anomaly = np.asarray(truth.anomaly[test_lo:test_hi], dtype=bool)
            rare = np.asarray(truth.rare_event[test_lo:test_hi], dtype=bool)
            masks = {args.a: r_a >= 1.0, args.b: r_b >= 1.0}
            masks.update({k: f(r_a, r_b) for k, f in rules.items()})
            per_rule = {}
            for k, m in masks.items():
                m = np.asarray(m, dtype=bool) & scorable
                ev = eventwise.score(events, spans, m, scorable, beta=0.5) if task.scores_recall else None
                fa = falsealarm.score(events, spans, m, anomaly, rare, scorable)
                per_rule[k] = {"events": ev.as_dict() if ev is not None else None,
                               "false_alarms": fa.as_dict(), "alarm_ranges": len(mask_to_ranges(m))}
            for e in events:
                lo, hi = spans[e.event_id]
                pa, ba_at = peak_and_there(r_a, r_b, lo, hi)
                pb, ab_at = peak_and_there(r_b, r_a, lo, hi)
                row = {"event_id": e.event_id, "fold": fold.index, "category": e.category,
                       "cell": e.cell, "footprint": int(hi - lo),
                       f"{args.a}_peak": pa, f"{args.b}_at_{args.a}_peak": ba_at,
                       f"{args.b}_peak": pb, f"{args.a}_at_{args.b}_peak": ab_at}
                for k, m in masks.items():
                    row[f"caught_{k}"] = bool((np.asarray(m, dtype=bool) & scorable)[lo:hi].any())
                (anomalies if e.category == ANOMALY else rares if e.category == RARE else []).append(row)
            folds_out.append({"fold": fold.index, "thresholds": {args.a: thr_a, args.b: thr_b}, "rules": per_rule})
            print(f"    {task.id} fold {fold.index}: " + "  ".join(
                f"{k} recall {per_rule[k]['events']['event_recall']['k']}/{per_rule[k]['events']['event_recall']['n']} rare {per_rule[k]['false_alarms']['rare_event_false_alarms']['k']}/{per_rule[k]['false_alarms']['rare_event_false_alarms']['n']} nom {per_rule[k]['false_alarms']['nominal_step_false_alarms']['k']}"
                for k in ("OR", "AND", "MEAN")), flush=True)
        results[task.id] = {"folds": folds_out, "anomalies": anomalies, "rare_events": rares}

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-combination-scope.json"
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
