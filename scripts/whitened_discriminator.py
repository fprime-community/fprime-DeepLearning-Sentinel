"""What actually separates the events `lstm-whitened` loses from the ones it keeps.

Three hypotheses have now been measured and refuted. Not channel count and not
footprint (`docs/RESULTS.md` 6d): lost and kept events have the same values for
both. Not single-channel blindness: every lost event touches 5 to 12 of the
channels in view. And not pruning: disabling it entirely leaves `m1-g8.9.10`
unchanged at 27/46 and 21/32.

The remaining candidate is mechanical, and it is the uncomfortable one -- the
complement of the design's own strength rather than a defect in it.

`lstm-whitened` scores ``sqrt(z' P z)`` where ``P`` is the inverse nominal
correlation. That **divides out** movement along directions the residual normally
travels. It is why a commanded manoeuvre is suppressed: a manoeuvre moves
channels together in the pattern the forecaster learned. But an **anomaly** that
moves channels together in that same pattern, merely at abnormal magnitude, is
suppressed by exactly the same arithmetic -- while telemanom's per-channel
magnitude test sees it plainly.

Two lengths per timestep decide it:

    raw    = ||z||              standardised, ignoring the correlation
    whit   = sqrt(z' P z)       whitened, dividing the correlation out
    ratio  = whit / raw         LOW means in-pattern, HIGH means off-pattern

If the lost events are **large in `raw` and low in `ratio`**, they are in-pattern
excursions and whitening removes their signal by construction. That is a
structural property of the rule, not a threshold to loosen, and it would mean the
two detectors are complementary rather than one dominating the other.

If they are simply small in both, the events are weak in every view and the loss
is ordinary sensitivity.

One forward pass per fold per channel set. Cached weights, nothing refitted.
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
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import whiten                                     # noqa: E402

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    print(f"  Whitened discriminator -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        rows = []
        for fold in split.folds:
            train_lo, train_hi = fold.train
            test_lo, test_hi = fold.test
            usable = train_mask(fold, view.truth)[train_lo:train_hi]
            fitting = context_for(view, fold, fold.train)

            det = D.WhitenedDetector()
            det.fit(view.values[train_lo:train_hi], usable, fitting)
            det.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi],
                      fitting)
            w = det.whitening

            warmup = min(det.warmup_steps, test_lo)
            window_lo = test_lo - warmup
            scored = context_for(view, fold, (test_lo, test_hi))
            values = view.values[window_lo:test_hi]
            pieces = [b for b in det._signed_blocks(values, scored)]
            signed = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            del pieces

            z = (np.asarray(signed, dtype=np.float64) - w.mean) / w.scale
            del signed
            raw = np.sqrt(np.einsum("ij,ij->i", z, z))
            whit = np.sqrt(np.maximum(np.einsum("ij,ij->i", z @ w.precision, z), 0.0))
            del z
            raw, whit = raw[warmup:], whit[warmup:]
            caught = whit >= w.threshold

            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a, b = lo - test_lo, min(hi - test_lo, whit.shape[0])
                if b <= a:
                    continue
                pr, pw = float(raw[a:b].max()), float(whit[a:b].max())
                rows.append({
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "footprint": int(hi - lo),
                    "peak_raw": pr, "peak_whitened": pw,
                    "ratio": (pw / pr if pr > 0 else None),
                    "threshold": float(w.threshold),
                    "reach": pw / float(w.threshold),
                    "caught": bool(caught[a:b].any()),
                })
            del raw, whit, caught
            D.clear_caches()
        results[task.id] = rows
        lost = [r for r in rows if not r["caught"]]
        kept = [r for r in rows if r["caught"]]
        for label, grp in (("kept", kept), ("lost", lost)):
            if not grp:
                continue
            print(f"    {task.id:12s} {label:4s} n={len(grp):<3} "
                  f"peak_raw med {np.median([r['peak_raw'] for r in grp]):8.2f}  "
                  f"peak_whit med {np.median([r['peak_whitened'] for r in grp]):8.2f}  "
                  f"ratio med {np.median([r['ratio'] for r in grp if r['ratio']]):6.3f}  "
                  f"reach med {np.median([r['reach'] for r in grp]):6.3f}", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-discriminator.json"
    path.write_text(json.dumps({"task": primary.id,
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
