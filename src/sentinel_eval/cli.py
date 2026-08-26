"""Command line for the harness. The composition root, and nothing more.

    list-tasks               what can be scored
    describe <task>          channels, denominators and split coverage, before any run
    run <task> --detector..  fit, score and record. Several detectors, one bundle
    selftest                 the whole pipeline on generated data, zero R2 operations

This is the only module importing both `sentinel_eval` and `sentinel_models`. The
harness must not know what a GRU is, so turning ``--detector gru`` into an object
happens here and the dependency stays one-way.

**Several detectors per run is the point, not a convenience.** A cold run costs
ten Class B operations whether it scores one detector or five, because the bundle
is fetched once and held. Scoring LSTM, GRU and TCN separately would cost thirty
and would approach the thousand-operation tripwire over a month of work; scoring
them together costs ten, and is how the architecture comparison runs anyway.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sentinel_data import config as C

from . import bundle as bundle_mod
from . import ops, read, splits, synthetic, tasks
from .catalog import Catalog
from .errors import HarnessError
from .labels import LabelSet
from .metrics import diagnostics
from .scorecard import GateRecord

RUNS = C.PROJECT_ROOT / "runs"
SYNTHETIC_TASK = "synthetic"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _open_source(task, args):
    """Pick the bucket. The synthetic task never touches R2 or credentials."""
    if task.id == SYNTHETIC_TASK:
        return synthetic.build(seed=args.seed, n=args.steps), None, None, None
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    return read.R2Source(client, cfg.bucket), client, budget, (cfg, ledger)


def _load(task, args, log=print, telecommands: int | None = None):
    source, client, budget, ledger_state = _open_source(task, args)
    catalog = Catalog.load(source)
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    channel_ids = task.selection.resolve(catalog)
    log(f"  resolved {len(channel_ids)} channels from the manifest: {channel_ids}")
    if telecommands is not None:
        log(f"  telecommands: priority {telecommands}, as exogenous model inputs")
    loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                             channel_ids=channel_ids, telecommands=telecommands,
                             log=log if args.verbose else None)
    return loaded, labels, catalog, client, budget, ledger_state


def _split_for(task, n):
    builder = {"chronological": splits.chronological,
               "forward_chaining": splits.forward_chaining}[task.split]
    return builder(n, **task.split_kwargs)


# --------------------------------------------------------------------------
def cmd_list(args) -> int:
    print("tasks:")
    print(tasks.listing())
    from sentinel_models import registry
    print(f"\ndetectors: {', '.join(registry.available())}")
    return 0


def cmd_describe(args) -> int:
    task = tasks.get(args.task)
    print(f"TASK {task.id}   {task.headline}")
    print(f"  {task.note}")
    if task.limitation:
        print(f"  LIMITATION  {task.limitation}")
    loaded, labels, _catalog, client, budget, state = _load(task, args)
    print(loaded.describe())
    census = labels.census(list(loaded.truth.events))
    print(f"\n  events on this grid: {census}")
    split = _split_for(task, len(loaded.grid))
    print(f"\n  split {split.describe()}")
    print(splits.render(splits.coverage(split, loaded.truth)))
    if budget:
        print()
        print(budget.report())
    return 0


def _member(primary, secondary_id: str, persistence: int | None):
    """A paired set, carrying the primary's split so only channels differ.

    Scoring the pair over different splits would confound "which channels" with
    "which years"; the grid is shared for the same reason (`Bundle.subset`).
    """
    secondary = tasks.get(secondary_id)
    fields = {**vars(secondary), "split": primary.split,
              "split_params": primary.split_params}
    if persistence is not None:
        fields["persistence"] = persistence
    return type(secondary)(**fields)


def cmd_run(args) -> int:
    from sentinel_models import registry
    from .harness import evaluate

    # Weights persist under runs/ between processes (docs/HARNESS.md, Rule 1
    # stated precisely). Caching is for iteration; a published number is
    # reproduced from cold, which is what this refuses the cache for.
    registry.set_caching(not args.no_cache)
    if args.no_cache:
        print("  --no-cache: refitting from cold, persisted weights ignored")

    primary = tasks.get(args.task)
    if args.persistence is not None:
        primary = type(primary)(**{**vars(primary), "persistence": args.persistence})

    pair = tasks.paired_with(primary.id)
    if args.only:
        pair = (primary.id,)
    members = [primary] + [_member(primary, tid, args.persistence)
                           for tid in pair if tid != primary.id]
    partial = len(members) < len(tasks.paired_with(primary.id))

    print(f"TASK {primary.id}   {primary.headline}")
    if len(members) > 1:
        print(f"  paired sets, reported together always: "
              f"{', '.join(m.id for m in members)}")
    if partial:
        print("  --only: PARTIAL run, barred from docs/RESULTS.md")

    # Detectors first, deliberately: whether the load fetches telecommands is a
    # property of what is being scored, and asking after the load would mean
    # either a second read or a flag that can disagree with the detector.
    detectors = [registry.build(name) for name in args.detector]
    wants = any(getattr(d, "wants_commands", False) for d in detectors)

    # One load of the widest selection; the rest are column subsets of it.
    loaded, labels, catalog, client, budget, state = _load(
        primary, args, telecommands=args.telecommand_priority if wants else None)
    print(loaded.describe())
    print(f"\n  scoring {len(detectors)} detector(s) x {len(members)} set(s) "
          f"against one loaded bundle")

    gate = GateRecord(partial=partial)
    for member in members:
        view = loaded if member.id == primary.id else loaded.subset(
            member.selection.resolve(catalog), labels)
        split = _split_for(member, len(view.grid))
        rows = splits.check(split, view.truth) if member.scores_recall \
            else splits.coverage(split, view.truth)
        print(f"\n  [{member.id}] {len(view.channels)} channels, split {split.describe()}")
        print(splits.render(rows))
        gate.records.append(evaluate(view, split, detectors, member,
                                     sweep=not args.no_sweep,
                                     log=print if args.verbose else None))

    if budget is not None:
        if not args.no_ledger:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
        for record in gate.records:
            record.ops = {**budget.as_dict(), "report": budget.report()}

    print(gate.render())

    if args.diagnostics:
        print("\n" + diagnostics.BANNER)

    for path in _write(gate, primary, args):
        print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")
    return 0


def cmd_selftest(args) -> int:
    """Prove the referee before trusting it: the ceiling and the floor."""
    from sentinel_models import baselines

    args.task = SYNTHETIC_TASK
    task = tasks.get(SYNTHETIC_TASK)
    loaded, labels, _catalog, _client, _budget, _state = _load(task, args,
                                                              log=lambda *a, **k: None)
    split = _split_for(task, len(loaded.grid))
    from .harness import evaluate

    checks: list[tuple[str, bool, str]] = []
    oracle = evaluate(loaded, split, [baselines.Oracle(loaded.truth.anomaly)], task,
                      sweep=False).scorecards[0].pooled
    quiet = evaluate(loaded, split, [baselines.AlwaysQuiet()], task,
                     sweep=False).scorecards[0].pooled

    checks.append(("oracle reaches perfect recall",
                   oracle["event_recall"].rate == 1.0, str(oracle["event_recall"])))
    checks.append(("oracle reaches perfect precision",
                   oracle["event_precision"].rate == 1.0, str(oracle["event_precision"])))
    checks.append(("oracle F0.5 is 1.0", abs((oracle["event_f0.5"] or 0) - 1.0) < 1e-9,
                   f"{oracle['event_f0.5']:.6f}"))
    checks.append(("oracle VUS-PR is 1.0", abs((oracle["vus_pr"] or 0) - 1.0) < 1e-9,
                   f"{oracle['vus_pr']:.6f}"))
    checks.append(("oracle raises no false alarm on rare events",
                   oracle["rare_event_false_alarms"].k == 0,
                   str(oracle["rare_event_false_alarms"])))
    checks.append(("silent detector recalls nothing",
                   quiet["event_recall"].k == 0, str(quiet["event_recall"])))
    checks.append(("silent detector precision is undefined, not zero",
                   quiet["event_precision"].undefined, str(quiet["event_precision"])))
    checks.append(("every count carries a denominator",
                   all(getattr(v, "n", 1) is not None
                       for v in oracle.values() if hasattr(v, "n")), "k/n"))

    print("  SELFTEST -- generated data, zero R2 operations\n")
    for label, ok, detail in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label:<46} {detail}")
    failures = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n  {len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


def _write(gate, task, args) -> list[Path]:
    """One artifact per detector, carrying every paired set it was scored on."""
    written = []
    stamp = _utc_stamp()
    seen = {c.fingerprint: c.detector
            for record in gate.records for c in record.scorecards}
    for fingerprint, detector in seen.items():
        folder = RUNS / task.id / detector
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stamp}-{fingerprint}.json"
        path.write_text(json.dumps(gate.for_detector(fingerprint), indent=2,
                                   default=str) + "\n")
        written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="synthetic fixture seed")
    parser.add_argument("--steps", type=int, default=20_000, help="synthetic fixture length")
    parser.add_argument("--acknowledge-tripwire", action="store_true",
                        help="continue past the 1,000-operation per-run tripwire")
    parser.add_argument("--no-ledger", action="store_true",
                        help="do not write the ops ledger back (saves 1 Class A)")
    parser.add_argument("--telecommand-priority", type=int, default=3,
                        help="ESA's priority grade to supply as model inputs. 3 is "
                             "what ESA feeds its own baselines; Mission1 holds 11 of "
                             "them. Only fetched if a detector asks for commands")
    parser.add_argument("--no-cache", action="store_true",
                        help="refit from cold, ignoring persisted weights. Required "
                             "before any result enters docs/RESULTS.md")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tasks")
    sub.add_parser("selftest")
    describe = sub.add_parser("describe")
    describe.add_argument("task")
    run = sub.add_parser("run")
    run.add_argument("task")
    run.add_argument("--detector", action="append", required=True,
                     help="repeatable; all share one bundle load")
    run.add_argument("--persistence", type=int, default=None)
    run.add_argument("--diagnostics", action="store_true",
                     help="also compute quarantined point-adjusted figures")
    run.add_argument("--no-sweep", action="store_true",
                     help="skip the oracle best-F0.5 upper bound")
    run.add_argument("--only", action="store_true",
                     help="score this set alone -- stamps the artifact partial "
                          "and bars it from docs/RESULTS.md")

    args = parser.parse_args(argv)
    handlers = {"list-tasks": cmd_list, "describe": cmd_describe,
                "run": cmd_run, "selftest": cmd_selftest}
    try:
        return handlers[args.command](args)
    except HarnessError as exc:
        print(f"\n  REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
