#!/usr/bin/env python3
"""Fit every fold and bank the weights. No scoring. Built for a rented GPU.

Fitting is the only expensive part of a run and the only part a GPU helps with:
scoring goes through `sentinel_models.reference`, the plain-NumPy forward pass
that Phase 2's C++ is transcribed from, and no GPU touches it. So the work
splits -- **fit on the pod, score on the Mac** -- and weights travel between them
as plain arrays with no tie to the fitting machine.

**What makes that legitimate is an assertion, not an environment lockfile.**
`tests/test_reference_equivalence.py` holds the torch model and the NumPy
reference to 1e-5. Run it on the fitting box and the weights it produces are
certified for the scoring box, whatever the Python, torch or CUDA versions are at
either end. The environments do not need to match; only the arithmetic does.

**Correctness here is self-verifying.** The fold windows and `Context` are built
exactly as `harness._score_fold` builds them, and if they are not, the
content-keyed weight cache simply misses when the Mac scores and the Mac refits.
That is a performance failure, never a wrong number -- so the worst outcome of a
mistake in this file is a slow evening, and the local run asserts zero new weight
files as proof the hit happened.

    python scripts/fit_folds.py --task m1-g8.9.10 --detector lstm-telemanom \\
                               --detector lstm-commanded --device cuda
    python scripts/fit_folds.py --task m1-g8.9.10 --detector gru-quantile --device cuda

**The report is written before the ledger is committed.** The weights are on
disk the moment each fit ends, so a ledger failure never costs a fit; but a
ledger failure used to cost the timing report, which is the one record of what
the pod actually did. Artifact first, bookkeeping second, the same order every
analysis script keeps (commit 75cc846).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                   # noqa: E402

from sentinel_data import config as C                                # noqa: E402
from sentinel_eval import bundle as bundle_mod                       # noqa: E402
from sentinel_eval import ops, read, splits, tasks                   # noqa: E402
from sentinel_eval.catalog import Catalog                            # noqa: E402
from sentinel_eval.detector import Context                           # noqa: E402
from sentinel_eval.labels import LabelSet                            # noqa: E402
from sentinel_eval.splits import train_mask                          # noqa: E402
from sentinel_models import detectors as D                           # noqa: E402
from sentinel_models import lstm as L                                # noqa: E402


def load(task, priority: int | None):
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    source = read.R2Source(client, cfg.bucket)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=task.selection.resolve(catalog),
                             telecommands=priority, log=print)
    return loaded, labels, catalog, client, budget, (cfg, ledger)


def members(primary, catalog, labels, loaded):
    """The paired sets, carrying the primary's split -- as `cli.py` builds them."""
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


def fit_fold(view, fold, detector) -> dict:
    """Exactly the fit `harness._score_fold` performs, and nothing after it."""
    train_lo, train_hi = fold.train
    context = Context(
        mission=view.mission, channels=view.channel_ids, groups=view.groups,
        period_seconds=view.provenance["grid_period_seconds"], fold=fold.index,
        window=(train_lo, train_hi),
        commands=None if view.commands is None else view.commands[train_lo:train_hi],
        command_ids=view.command_ids,
    )
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    started = time.perf_counter()
    detector.fit(view.values[train_lo:train_hi], usable, context)
    return {"seconds": time.perf_counter() - started, **(detector.report or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--telecommand-priority", type=int, default=3)
    parser.add_argument("--threads", type=int, default=L.THREADS)
    args = parser.parse_args()

    from sentinel_models import registry
    built = [registry.build(name) for name in args.detector]
    wants = any(getattr(d, "wants_commands", False) for d in built)

    L.DEVICE = args.device
    L.THREADS = args.threads
    print(f"  FIT ONLY   {args.task}   device {args.device}   "
          f"detectors {args.detector}"
          f"{'   +telecommands' if wants else ''}")

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2
    loaded, labels, catalog, client, budget, state = load(
        primary, args.telecommand_priority if wants else None)
    print(loaded.describe())

    report: dict = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        for detector, name in zip(built, args.detector):
            for fold in split.folds:
                print(f"    [{task.id}] {name} fold {fold.index}: "
                      f"fit {fold.train_steps:,} ...", flush=True)
                info = fit_fold(view, fold, detector)
                info["device"] = args.device
                info["cell"] = detector.hyper.cell
                report[f"{task.id}/{name}/{fold.index}"] = info
                print(f"      {info['seconds'] / 60:.1f} min   "
                      f"{info.get('epochs_run', 0)} epochs   "
                      f"best epoch {info.get('best_epoch', -1)}   "
                      f"val MSE {info.get('best_validation_mse', float('nan')):.3e}",
                      flush=True)

    total = sum(v["seconds"] for v in report.values())
    print(f"\n  {len(report)} fits in {total / 60:.1f} min")
    kept_first = [k for k, v in report.items() if v.get("best_epoch") == 0]
    print(f"  fits that kept their first epoch: {len(kept_first)}"
          f"{'  <-- THE DEFECT IS BACK' if kept_first else '   (none, as it should be)'}")

    out = C.PROJECT_ROOT / "runs" / "_weights" / "fit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"  wrote {out}")

    try:
        cfg, ledger = state
        ops.commit(client, cfg.bucket, ledger, budget)
        print(); print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      operations spent and NOT recorded; the weights and the report above are safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
