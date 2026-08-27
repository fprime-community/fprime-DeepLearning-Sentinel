#!/usr/bin/env python3
"""The detection threshold, swept against the residuals of the model it will run on.

`z` counts standard deviations of the smoothed error, so the constants audit
called it scale-free -- correctly, about units. It is not shape-free. Correcting
the `min_delta` defect improved the forecast about fortyfold, and telemanom's
published floor of 2.5 then produced **3,548 alarm ranges where it had produced
182**, while every detection figure improved: 38 of 46 events caught against 37,
headline cell held, lead time unchanged at +26. With a poor forecast the residual
is dominated by model bias and 2.5 sigma is a genuine excursion; with a good one
it is dominated by noise and 2.5 sigma sits on the floor.

So the threshold is **fitted against the residuals it will see**, and this is the
measurement that fits it (docs/DECISIONS.md, the amendment on dimensionless
constants).

**Lead time is the constraint, not F0.5.** A higher floor takes longer to cross,
so raising it spends the +26 timestep budget that is the only reason this
component exists. What we are looking for is the point where the alarm count
returns to something sane while headline-cell recall and lead time survive -- not
the highest F0.5.

    python scripts/threshold_sweep.py                     # z floor, then pruning
    python scripts/threshold_sweep.py --p 0.05 0.13 0.3   # the pruning axis


"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import bundle as bundle_mod                         # noqa: E402
from sentinel_eval import harness, ops, read, splits, synthetic, tasks  # noqa: E402
from sentinel_eval.catalog import Catalog                              # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.metrics.counts import Count                         # noqa: E402
from sentinel_eval.metrics.leadtime import LeadTimeScore               # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import telemanom                                  # noqa: E402

#: The floor of the z sweep. 2.5 is telemanom's published value and the control:
#: every number this project has published was measured there.
Z_FLOORS = (2.5, 5.0, 8.0, 12.0, 16.0, 22.0, 30.0)

#: Pruning: the normalised step decrease a sequence must stand clear by. It reads
#: the same residual ladder the threshold does, so it may have shifted for the
#: same reason. 0.13 is published and is the control.
PRUNING = (0.13,)

#: Held at the control values throughout, so what moves is the threshold alone.
PERSISTENCE, AGREEMENT = 1, 1


def load(task, args):
    """One bundle, held for the whole grid. Costs what a single run costs."""
    if task.id == "synthetic":
        source, client, budget, state = synthetic.build(seed=0, n=args.steps), None, None, None
    else:
        cfg = C.load_r2_config()
        client, budget = ops.connect(cfg)
        ledger = ops.load(client, cfg.bucket)
        budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
        source, state = read.R2Source(client, cfg.bucket), (cfg, ledger)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=task.selection.resolve(catalog))
    return loaded, labels, catalog, client, budget, state


def members(primary, catalog, labels, loaded):
    """The paired sets, as `cli.py` builds them: the primary's split, so only
    channels differ. Reporting both is not optional (docs/HARNESS.md section 2)."""
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


def cell(view, task, z_floor, pruning_p, mode) -> dict:
    """One point of the sweep. Weights come from the cache; the threshold does not."""
    task = type(task)(**{**vars(task), "persistence": PERSISTENCE})
    split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
    config = telemanom.Config(z_floor=z_floor,
                              z_ceiling=max(telemanom.Z_CEILING, z_floor + 8.0),
                              pruning_p=pruning_p)
    build = D.TelemanomQuantile if mode == "quantile" else D.ForecastDetector
    detector = build(agreement=AGREEMENT, config=config)
    pooled = harness.evaluate(view, split, [detector], task,
                              sweep=False).scorecards[0].pooled

    lead = pooled.get("lead_time")
    return {
        "z_floor": z_floor, "pruning_p": pruning_p,
        "f0.5": pooled.get("event_f0.5"),
        "recall": pooled["event_recall"],
        "precision": pooled["event_precision"],
        "mvgs": pooled["headline_cell_recall"],
        "point": pooled["point_recall"],
        "rare_fa": pooled["rare_event_false_alarms"],
        "alarms_per_1k": pooled.get("alarms_per_1000_nominal_timesteps"),
        "vus_pr": pooled.get("vus_pr"),
        "lead": lead if isinstance(lead, LeadTimeScore) else None,
    }


def render(task_id: str, rows: list[dict]) -> str:
    head = (f"  {'z':>5} {'p':>5} | {'F0.5':>6} {'recall':>13} {'precision':>15} "
            f"{'MVGS':>13} {'point':>12} | {'rare-event FA':>15} {'alarms/1k':>9} "
            f"| {'lead med':>9} {'p25':>8} {'p75':>8}")
    lines = [f"", f"  === {task_id} " + "=" * (len(head) - len(task_id) - 8), head,
             "  " + "-" * (len(head) - 2)]
    for r in rows:
        f = "  n/a " if r["f0.5"] is None else f"{r['f0.5']:>6.3f}"
        lead = r["lead"].summary() if r["lead"] else {"median": None, "p25": None, "p75": None}
        def q(key):
            return "        -" if lead[key] is None else f"{lead[key]:>9,.0f}"
        lines.append(
            f"  {r['z_floor']:>5.1f} {r['pruning_p']:>5.2f} | {f} "
            f"{r['recall'].brief():>13} {r['precision'].brief():>15} "
            f"{r['mvgs'].brief():>13} {r['point'].brief():>12} | "
            f"{r['rare_fa'].brief():>15} "
            f"{(r['alarms_per_1k'] or 0):>9.3f} | {q('median')} {q('p25')[1:]} {q('p75')[1:]}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--mode", choices=("ndt", "quantile"), default="ndt",
                        help="which thresholding rule to sweep. quantile mode holds "
                             "the best F0.5 and the best false-alarm rate in the "
                             "project and is disqualified on lead time; the sweep "
                             "asks whether any k restores it")
    parser.add_argument("--steps", type=int, default=60_000, help="synthetic fixture length")
    parser.add_argument("--no-cache", action="store_true",
                        help="refit from cold; required before publishing")
    parser.add_argument("--z", type=float, nargs="*", default=list(Z_FLOORS))
    parser.add_argument("--p", type=float, nargs="*", default=list(PRUNING))
    args = parser.parse_args()

    D.set_caching(not args.no_cache)
    if args.no_cache:
        print("  --no-cache: refitting from cold, persisted weights ignored")

    primary = tasks.get(args.task)
    print(f"  DECISION GRID   {primary.id}   mode {args.mode}   "
          f"z floor {args.z} x pruning {args.p}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    print(loaded.describe())

    results, grids = {}, []
    for task, view in members(primary, catalog, labels, loaded):
        rows = []
        for pruning_p in args.p:
            for z_floor in args.z:
                print(f"    [{task.id}] z={z_floor} p={pruning_p} ...", flush=True)
                rows.append(cell(view, task, z_floor, pruning_p, args.mode))
        rows.sort(key=lambda r: (r["z_floor"], r["pruning_p"]))
        results[task.id] = rows
        grids.append(render(task.id, rows))

    print("\n".join(grids))
    print("\n  Every cell is reported. No configuration is selected here: choosing one "
          "and\n  publishing only its numbers would hide the trade-off the grid exists "
          "to show.")
    print("  A cell with a negative median lead time is NOT a candidate, whatever its "
          "F0.5\n  -- docs/HARNESS.md section 1. Lead time is the constraint here: a "
          "higher\n  floor takes longer to cross, and +26 timesteps is the whole budget.")

    if budget is not None:
        cfg, ledger = state
        ops.commit(client, cfg.bucket, ledger, budget)
        print()
        print(budget.report())

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_threshold"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-{args.mode}.json"
    path.write_text(json.dumps({
        "task": primary.id, "cold": args.no_cache, "mode": args.mode,
        "z_floors": args.z, "pruning": args.p,
        "operations": budget.as_dict() if budget else None,
        "sets": {t: [{**r,
                      **{key: r[key].as_dict() for key in
                         ("recall", "precision", "mvgs", "point", "rare_fa")},
                      "lead": r["lead"].as_dict() if r["lead"] else None}
                     for r in rows] for t, rows in results.items()},
    }, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
