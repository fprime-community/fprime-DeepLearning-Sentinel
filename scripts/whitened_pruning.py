"""Does pruning explain the events `lstm-whitened` loses? Measured, not suspected.

`lstm-whitened` catches nothing `lstm-telemanom` misses -- `whitened-only` is 0 on
both channel sets -- so it is a **filter** on telemanom's detections rather than a
different view of the data. Eleven anomalies on the gate set, seven of them
headline-cell, are detected by one and not the other, and per-event forensics
found nothing that separates them: not channel count, not footprint, and not a
single one confined to one channel.

`telemanom.prune` is the remaining suspect and it was named without proof. It
sorts the sequence peaks, walks down the normalised step decreases, and discards
everything below the last drop of more than ``p``. That ladder was derived
against **telemanom's per-channel absolute errors**. `lstm-whitened` feeds it
something else entirely -- one joint whitened length whose distribution has no
reason to share the shape ``p = 0.13`` was chosen for.

So this scores the identical residuals twice, changing **only** the pruning, and
asks the two questions that decide it:

* do the lost events come back, and
* what happens to the rare-event false-alarm rate, currently 2/48

``pruning_p = 0.0`` disables the filter exactly: the ladder's relative steps are
non-negative, so no sequence is ever dropped.

One forward pass per fold per channel set. Both alarm shapings come off the same
residuals, so the comparison differs in the filter alone.
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
from sentinel_eval.harness import _local_spans                         # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.metrics import falsealarm                           # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import whiten                                     # noqa: E402

ANOMALY = "Anomaly"
SETTINGS = (0.13, 0.0)          # published, and disabled


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    print(f"  Whitened pruning isolation -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        events_out, alarms_out = [], {p: {"rare_k": 0, "rare_n": 0, "ranges": 0}
                                      for p in SETTINGS}
        for fold in split.folds:
            train_lo, train_hi = fold.train
            test_lo, test_hi = fold.test
            usable = train_mask(fold, view.truth)[train_lo:train_hi]
            fitting = context_for(view, fold, fold.train)

            det = D.WhitenedDetector()
            det.fit(view.values[train_lo:train_hi], usable, fitting)
            det.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi],
                      fitting)

            warmup = min(det.warmup_steps, test_lo)
            window_lo = test_lo - warmup
            scored = context_for(view, fold, (test_lo, test_hi))
            values = view.values[window_lo:test_hi]

            pieces = [b for b in det._signed_blocks(values, scored)]
            signed = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            del pieces

            observed = np.isfinite(values).all(axis=1)
            observed &= np.asarray(view.valid[window_lo:test_hi], dtype=bool).all(axis=1)
            scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
            anomaly = np.asarray(view.truth.anomaly[test_lo:test_hi], dtype=bool)
            rare = np.asarray(view.truth.rare_event[test_lo:test_hi], dtype=bool)
            local_events, local_spans = _local_spans(list(view.truth.events),
                                                     view.truth.spans, test_lo, test_hi)

            masks = {}
            for p in SETTINGS:
                cfg = whiten.Config(**{**vars(det.config), "pruning_p": p})
                out, _ = whiten.ratios(signed, cfg, det.whitening)
                m = (out >= 1.0)[warmup:] & observed[warmup:] & scorable
                masks[p] = m
                fa = falsealarm.score(local_events, local_spans, m, anomaly, rare,
                                      scorable)
                alarms_out[p]["rare_k"] += fa.rare_events.k
                alarms_out[p]["rare_n"] += fa.rare_events.n
                alarms_out[p]["ranges"] += len(mask_to_ranges(m))
            del signed

            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a, b = lo - test_lo, min(hi - test_lo, masks[SETTINGS[0]].shape[0])
                if b <= a:
                    continue
                events_out.append({
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "footprint": int(hi - lo),
                    **{f"caught_p{p}": bool(masks[p][a:b].any()) for p in SETTINGS},
                })
            D.clear_caches()
        results[task.id] = {"events": events_out, "alarms": alarms_out}
        for p in SETTINGS:
            n = alarms_out[p]
            caught = sum(1 for e in events_out if e[f"caught_p{p}"])
            mvgs = sum(1 for e in events_out
                       if e["cell"] == "Multivariate/Global/Subsequence" and e[f"caught_p{p}"])
            mvgs_n = sum(1 for e in events_out
                         if e["cell"] == "Multivariate/Global/Subsequence")
            print(f"    {task.id:12s} pruning_p={p:<5} recall {caught}/{len(events_out)}"
                  f"  MVGS {mvgs}/{mvgs_n}  rare-FA {n['rare_k']}/{n['rare_n']}"
                  f"  ranges {n['ranges']:,}", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-pruning.json"
    path.write_text(json.dumps({"task": primary.id, "settings": list(SETTINGS),
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
        print("      the operations were spent and are NOT recorded; "
              "the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
