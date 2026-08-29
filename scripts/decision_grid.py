#!/usr/bin/env python3
"""The decision layer, swept as a grid rather than chosen and reported.

Work item 4 established that the forecaster works and the decision rule does
not: `lstm-quantile` and `lstm-telemanom` are the *same* weights with different
thresholding, and they scored 0.421 against 0.269 with 1/48 rare-event false
alarms against 22/48. Two mechanisms this project designed were switched off
while that was measured.

    persistence N   Objective.md 7. A signal must survive N consecutive cycles.
                    Ran at N=1, which is off. Harness-side, already implemented.
    agreement   k   How many channels must exceed threshold at once. Ran at k=1,
                    the maximum over channels -- twelve independent chances to be
                    wrong on twelve channels.

They are not independent: both suppress alarms, so one can make the other
redundant. Hence a grid, and hence the whole grid is reported rather than the
cell somebody liked. Picking a configuration and publishing only its numbers
would hide the trade-off the grid exists to show.

**What this costs.** Nothing upstream of the reduction changes across the grid,
so weights are fitted once (persisted under `runs/_weights`, Rule 1 as stated in
docs/HARNESS.md) and the per-channel ratios are computed once per fold and held.
Every grid point after the first is a reduction and a threshold comparison.

    python scripts/decision_grid.py                    # the full grid
    python scripts/decision_grid.py --no-cache         # refit from cold
    python scripts/decision_grid.py --task synthetic   # zero R2 operations
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

#: Persistence values swept. 1 is off, and is the control -- every published
#: number so far was measured there.
PERSISTENCE = (1, 5, 20, 60)

#: Channels required to agree. 1 is the control: the maximum over channels.
AGREEMENT = (1, 2, 3)


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


def cell(view, task, n, k, mode) -> dict:
    """One grid point. Weights and per-channel reductions come from the cache."""
    task = type(task)(**{**vars(task), "persistence": n})
    split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
    build = D.TelemanomQuantile if mode == "quantile" else D.ForecastDetector
    detector = build(agreement=k)
    pooled = harness.evaluate(view, split, [detector], task,
                              sweep=False).scorecards[0].pooled

    lead = pooled.get("lead_time")
    return {
        "persistence": n, "agreement": k,
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
    head = (f"  {'N':>3} {'k':>2} | {'F0.5':>6} {'recall':>13} {'precision':>15} "
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
            f"  {r['persistence']:>3} {r['agreement']:>2} | {f} "
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
    parser.add_argument("--persistence", type=int, nargs="*", default=list(PERSISTENCE))
    parser.add_argument("--agreement", type=int, nargs="*", default=list(AGREEMENT))
    args = parser.parse_args()

    D.set_caching(not args.no_cache)
    if args.no_cache:
        print("  --no-cache: refitting from cold, persisted weights ignored")

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2
    print(f"  DECISION GRID   {primary.id}   mode {args.mode}   "
          f"persistence {args.persistence} x agreement {args.agreement}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    print(loaded.describe())

    results, grids = {}, []
    for task, view in members(primary, catalog, labels, loaded):
        rows = []
        for k in args.agreement:
            for n in args.persistence:
                print(f"    [{task.id}] N={n} k={k} ...", flush=True)
                rows.append(cell(view, task, n, k, args.mode))
        rows.sort(key=lambda r: (r["persistence"], r["agreement"]))
        results[task.id] = rows
        grids.append(render(task.id, rows))

    print("\n".join(grids))
    print("\n  Every cell is reported. No configuration is selected here: choosing one "
          "and\n  publishing only its numbers would hide the trade-off the grid exists "
          "to show.")
    print("  A cell with a negative median lead time is NOT a candidate, whatever its "
          "F0.5\n  -- docs/HARNESS.md section 1.")


    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_grid"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-{args.mode}.json"
    path.write_text(json.dumps({
        "task": primary.id, "cold": args.no_cache, "mode": args.mode,
        "persistence": args.persistence, "agreement": args.agreement,
        "operations": budget.as_dict() if budget else None,
        "sets": {t: [{**r,
                      **{key: r[key].as_dict() for key in
                         ("recall", "precision", "mvgs", "point", "rare_fa")},
                      "lead": r["lead"].as_dict() if r["lead"] else None}
                     for r in rows] for t, rows in results.items()},
    }, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")

    # Artifact first, ledger second, and the order is load-bearing: a
    # transient RequestTimeTooSkewed on the ledger PutObject once destroyed
    # twenty-five minutes of completed analysis held in memory. The result is
    # the expensive thing; the ledger is bookkeeping and can be retried.
    try:
        if budget is not None:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
            print()
            print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      the operations were spent and are NOT recorded; "
              "the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
