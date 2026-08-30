"""The admission-rate curve for the order-statistic rule. docs/MODELS.md 10.8.

The first `lstm-oscfar` run failed on an arithmetic error rather than on the
design: the two terms were calibrated independently, each to admit the target
rate, and then combined with ``max()``. The maximum of two thresholds each
admitting ``r`` admits far less than ``r``, so the rule was far quieter than its
budget asked, the floor dominated every window, and a dominant global term is the
global rule -- `lstm-quantile` rediscovered, identical on recall, headline cell
and point recall. This retests the design as intended and preserves the fit that
failed rather than replacing it.

**The admission rate is swept, never chosen.** An earlier version fixed 0.1% and
called it label-free. It is label-free, and label-free is not derived: nobody
derived 0.1%, and relabelling an arbitrary threshold as an arbitrary budget does
not remove the arbitrary constant. What a mission can absorb is an operational
input -- a two-person CubeSat team answers differently from a control room -- so
what is reported is the curve, and no operating point is recommended from it.
Choosing the best-scoring cell would be the oracle sweep docs/MODELS.md section 7
refuses, and it stays refused.

Cached weights, no refit. Both channel sets, every cell. `error_buffer` held at
100 so a smoothing parameter and a threshold rule do not move together
(docs/DECISIONS.md D21). Held-back sets are not loaded.
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
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import oscfar                                     # noqa: E402

#: Admission rates, as shares of nominal timesteps. Three decades, so the shape
#: of the response is visible rather than a single point being defended.
RATES = (0.0001, 0.001, 0.01)

#: Held at the control values, so what moves is the threshold rule alone.
PERSISTENCE, AGREEMENT = 1, 1


def load(task, args):
    """One bundle for the whole curve. Costs what a single scored run costs."""
    if task.id == "synthetic":
        source, client, budget, state = synthetic.build(seed=0, n=args.steps), None, None, None
    else:
        cfg = C.load_r2_config()
        client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
        ledger = ops.load(client, cfg.bucket)
        budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
        source, state = read.R2Source(client, cfg.bucket), (cfg, ledger)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=task.selection.resolve(catalog))
    return loaded, labels, catalog, client, budget, state


def members(primary, catalog, labels, loaded):
    """The paired sets, as `cli.py` builds them. Both, always -- HARNESS.md 2."""
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


def cell(view, task, mode: str, rate: float) -> dict:
    """One point of the curve. Weights come from the cache; the rule does not."""
    task = type(task)(**{**vars(task), "persistence": PERSISTENCE})
    split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
    config = oscfar.Config(calibration=mode, admission_rate=rate)
    detector = D.OSCFARDetector(agreement=AGREEMENT, config=config)
    card = harness.evaluate(view, split, [detector], task, sweep=False).scorecards[0]

    folds = []
    for fold in card.folds:
        events = fold.events
        training = fold.training or {}
        lead = (fold.lead_time.as_dict() if fold.lead_time else None)
        folds.append({
            "fold": fold.fold,
            "binding_rate": training.get("binding_rate"),
            "calibration": training.get("calibration"),
            "recall": events.recall.as_dict() if events else None,
            "precision": events.precision.as_dict() if events else None,
            "mvgs": events.headline_cell.as_dict() if events else None,
            "point": events.point.as_dict() if events else None,
            "alarm_ranges": events.precision.n if events else None,
            "rare_fa": fold.false_alarms.rare_events.as_dict(),
            "lead": (lead or {}).get("overall"),
        })
    pooled = card.pooled
    return {
        "calibration": mode, "admission_rate": rate,
        "binding_rate": [f["binding_rate"] for f in folds],
        "f0.5": pooled.get("event_f0.5"),
        "recall": pooled["event_recall"].as_dict(),
        "precision": pooled["event_precision"].as_dict(),
        "mvgs": pooled["headline_cell_recall"].as_dict(),
        "point": pooled["point_recall"].as_dict(),
        "rare_fa": pooled["rare_event_false_alarms"].as_dict(),
        "alarm_ranges": pooled["event_precision"].n,
        "lead": (pooled["lead_time"].as_dict().get("overall")
                 if pooled.get("lead_time") else None),
        "folds": folds,
    }


def render(set_id: str, rows: list[dict]) -> str:
    lines = ["", f"  {set_id}", "",
             "  BINDING RATE LEADS. Below 0.10 the floor is doing the work and the",
             "  design has collapsed to lstm-quantile -- docs/MODELS.md 10.3.",
             "  Alarms are counted; no operating point is recommended from these numbers.",
             "",
             f"  {'calibration':12s} {'rate':>8} {'bind f0/f1/f2':>18} {'F0.5':>7} "
             f"{'lead':>8} {'recall':>9} {'precision':>13} {'MVGS':>8} {'rare-FA':>8}"]
    for r in rows:
        bind = "/".join("-" if b is None else f"{b:.2f}" for b in r["binding_rate"])
        lead = r["lead"] or {}
        med = lead.get("median")
        f05 = r["f0.5"]
        lines.append(
            f"  {r['calibration']:12s} {r['admission_rate']:>8.2%} {bind:>18} "
            f"{('-' if f05 is None else format(f05, '.3f')):>7} "
            f"{('-' if med is None else format(med, '+.1f')):>8} "
            f"{r['recall']['k']}/{r['recall']['n']:<7} "
            f"{r['precision']['k']}/{r['precision']['n']:<11} "
            f"{r['mvgs']['k']}/{r['mvgs']['n']:<6} "
            f"{r['rare_fa']['k']}/{r['rare_fa']['n']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--modes", nargs="*", default=list(oscfar.CALIBRATIONS))
    parser.add_argument("--rates", nargs="*", type=float, default=list(RATES))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    parser.add_argument("--tag", default="curve")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back -- run once, at the end, settings "
              f"frozen (docs/MODELS.md section 6).")
        return 2
    for mode in args.modes:
        if mode not in oscfar.CALIBRATIONS:
            print(f"  unknown calibration {mode!r}; expected {oscfar.CALIBRATIONS}")
            return 2

    print(f"  OS-CFAR admission-rate curve -- {primary.id}")
    print(f"  modes {args.modes}  rates {args.rates}")
    loaded, labels, catalog, client, budget, state = load(primary, args)

    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz") if store.is_dir() else 0

    results: dict[str, list[dict]] = {}
    for task, view in members(primary, catalog, labels, loaded):
        rows = results.setdefault(task.id, [])
        for mode in args.modes:
            for rate in args.rates:
                print(f"    {task.id:12s} {mode:12s} rate {rate:.2%}", flush=True)
                rows.append(cell(view, task, mode, rate))
        print(render(task.id, rows), flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz") if store.is_dir() else 0
    if after != before:
        print(f"  ABORT: the weight store went from {before} to {after}. Something "
              f"trained; a cache miss is silent (docs/DECISIONS.md D16).")
        return 6
    print(f"\n  weight store unchanged at {after} files -- nothing was fitted")


    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_curve"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-{args.tag}.json"
    path.write_text(json.dumps({
        "task": primary.id, "modes": args.modes, "rates": args.rates,
        "persistence": PERSISTENCE, "agreement": AGREEMENT,
        "error_buffer": oscfar.Config().error_buffer,
        "operations": budget.as_dict() if budget else None,
        "sets": results,
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
