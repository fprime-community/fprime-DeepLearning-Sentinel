"""Work item 9.7 studies A and B. Pre-registered in `docs/MODELS.md` 23, M1-M9.

One bundle load for both studies, because two scripts would be two loads
(`docs/MODELS.md` 22's budget note). Cached weights, nothing refitted.

A -- the matched-operating-point comparison. Both detectors' thresholds are
     swept and the full recall-versus-nominal-alarm-rate curve is reported.
     Two points are named in advance (23.3): matched-quiet, where `rstd` is
     raised until its nominal-step rate equals `gru-quantile`'s, and
     matched-loud, the reverse. The threshold is swept, never chosen
     (`scripts/oscfar_curve.py`'s rule, section 7's standing refusal).

B -- the reduction study, D23 re-derived. Under the frozen calibration recipe
     -- the same 99.9th quantile of the same anomaly-masked fitting window --
     max-across-channels is replaced by L2, sum and k-of-n at k=2 and k=3.
     All five come from ONE forecast pass over `_smoothed_errors`, so no
     detector class is added and `gru-quantile` is untouched: the `max` arm IS
     the flying detector and reproducing its published counts is the gate this
     script must pass before it reports anything new.

     23.2's third correction stands: every arm here is downstream of the
     `np.abs` that removes the sign, so all four alternatives are sign-blind.
     A gain is a gain in co-occurrence sensitivity, not evidence for the
     relationship claim.

    PYTHONPATH=src .venv/bin/python scripts/reduction_and_curve.py
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
from sentinel_eval.detector import Context, Detector, reduce_scores    # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402

ANOMALY = "Anomaly"
GRU = "gru-quantile"
RSTD = "rstd"
MVGS = "Multivariate/Global/Subsequence"
ARMS = ("max", "l2", "sum", "k2", "k3")
#: 23.3's sweep. Multipliers of each detector's own calibrated threshold; 1.0 is
#: the frozen operating point and must reproduce the published scorecard.
#: What the `max` arm and `rstd` must reproduce before anything new is believed.
#: docs/RESULTS.md 6l, from the two named scorecard artifacts.
PUBLISHED = {
    "m1-g8.9.10": {"max": (27, 22), RSTD: (34, 25), "events": 46, "mvgs": 32},
    "m1-ss5":     {"max": (26, 21), RSTD: (34, 25), "events": 42, "mvgs": 31},
}
GRID = np.unique(np.concatenate([
    np.geomspace(0.02, 1.0, 60), np.geomspace(1.0, 50.0, 40), [1.0]]))


def reduce_arms(smoothed: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    """The five reductions, from one (T, C) matrix of smoothed per-channel errors.

    `max` is what `ForecastDetector.score` computes in quantile mode, so the
    `max` arm is `gru-quantile` and not a reimplementation of it. The other four
    are the alternatives D23 asked for and never got.
    """
    x = np.asarray(smoothed, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        srt = np.sort(np.nan_to_num(x, nan=-np.inf), axis=1)[:, ::-1]
        out = {
            "max": srt[:, 0],
            "k2": srt[:, 1] if x.shape[1] > 1 else srt[:, 0],
            "k3": srt[:, 2] if x.shape[1] > 2 else srt[:, -1],
            "l2": np.sqrt(np.nansum(x * x, axis=1)),
            "sum": np.nansum(x, axis=1),
        }
    for name in out:
        out[name] = out[name].astype(np.float64)
        out[name][~observed] = -np.inf     # nothing measured is never an alarm
    return out


def observed_mask(values, valid) -> np.ndarray:
    obs = np.isfinite(np.asarray(values, dtype=np.float32)).all(axis=1)
    if valid is not None:
        obs &= np.asarray(valid, dtype=bool).all(axis=1)
    return obs


def load(task, args):
    """The bundle, once. The synthetic fixture never touches R2 or credentials.

    `sentinel_eval.cli._open_source` makes the same split for the same reason: a
    smoke run must be able to exercise every line of this script at zero bucket
    operations, and it cannot do that through a path that opens a client first.
    """
    if task.id == "synthetic":
        from sentinel_eval import synthetic
        source = synthetic.build(seed=0, n=20_000)
        catalog = Catalog.load(source)
        labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
        loaded = bundle_mod.load(source, catalog, labels, mission=task.mission,
                                 channel_ids=task.selection.resolve(catalog))
        return loaded, labels, catalog, None, None, None
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


def gru_arms(view, fold, name=GRU):
    """Fit `gru-quantile` from cache, then every arm from one forecast pass."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(name)
    det.fit(view.values[train_lo:train_hi], usable, fitting)

    tr_vals, tr_valid = view.values[train_lo:train_hi], view.valid[train_lo:train_hi]
    tr = reduce_arms(det._smoothed_errors(tr_vals, fitting), observed_mask(tr_vals, tr_valid))

    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    te_vals, te_valid = view.values[window_lo:test_hi], view.valid[window_lo:test_hi]
    te_all = reduce_arms(det._smoothed_errors(te_vals, scored), observed_mask(te_vals, te_valid))
    te = {k: v[warmup:] for k, v in te_all.items()}

    thresholds = {k: float(Detector.threshold_from(det, tr[k][usable] if usable.any() else tr[k]))
                  for k in ARMS}
    D.clear_caches()
    return te, thresholds, usable


def rstd_scores(view, fold):
    """The floor, through the harness's own path, exactly as `floor_audit` does."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(RSTD)
    det.fit(view.values[train_lo:train_hi], usable, fitting)
    train_raw = det.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi], fitting)
    train_scores, _ = reduce_scores(train_raw, train_hi - train_lo)
    threshold = float(det.threshold_from(train_scores[usable] if usable.any() else train_scores))
    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    raw = det.score(view.values[window_lo:test_hi], view.valid[window_lo:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - window_lo)
    D.clear_caches()
    return scores[warmup:], threshold


def local_events(view, test_lo, test_hi):
    out = []
    for event in view.truth.events:
        if event.category != ANOMALY or event.event_id not in view.truth.spans:
            continue
        lo, hi = view.truth.spans[event.event_id]
        if test_lo <= lo < test_hi:
            out.append((event, lo - test_lo, min(hi - test_lo, test_hi - test_lo)))
    return out


def curve_point(scores, threshold, mult, events, clean):
    """Recall, headline-cell recall and nominal-step alarm rate at one multiplier."""
    fired = scores >= threshold * mult
    caught = [e for e, a_, b_ in events if b_ > a_ and fired[a_:b_].any()]
    return {
        "multiplier": float(mult),
        "caught": len(caught),
        "caught_mvgs": sum(1 for e in caught if e.cell == MVGS),
        "nominal_flagged": int((fired & clean).sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    # The offline fixture has no cached weights, so a smoke run must be allowed
    # to fit. Never passed against a real task: on Mission 1 a changed weight
    # store means something refitted, and that aborts.
    parser.add_argument("--detector", default=GRU)
    parser.add_argument("--allow-fit", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back (D29)."); return 2

    print(f"  Work item 9.7, studies A and B -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    out = {"task": primary.id, "pre_registration": "docs/MODELS.md 23",
           "arms": list(ARMS), "sets": {}}

    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        per_event, curves, totals = [], {}, {"events": 0, "mvgs": 0, "clean": 0}
        arm_hits = {a: {"events": 0, "mvgs": 0, "nominal": 0} for a in ARMS}
        arm_hits[RSTD] = {"events": 0, "mvgs": 0, "nominal": 0}
        grid_acc = {name: {i: {"caught": 0, "caught_mvgs": 0, "nominal_flagged": 0}
                           for i in range(len(GRID))} for name in ("max", RSTD)}

        for fold in split.folds:
            test_lo, test_hi = fold.test
            te, thr, _ = gru_arms(view, fold, args.detector)
            r_scores, r_thr = rstd_scores(view, fold)

            scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
            anomaly = np.asarray(view.truth.anomaly[test_lo:test_hi], dtype=bool)
            rare = np.asarray(view.truth.rare_event[test_lo:test_hi], dtype=bool)
            clean = scorable & ~anomaly & ~rare
            events = local_events(view, test_lo, test_hi)
            totals["clean"] += int(clean.sum())

            series = {a: (te[a], thr[a]) for a in ARMS}
            series[RSTD] = (r_scores, r_thr)
            masks = {n: (s >= t) & scorable for n, (s, t) in series.items()}

            for name in ("max", RSTD):
                s, t = series[name]
                for i, m in enumerate(GRID):
                    p = curve_point(np.where(scorable, s, -np.inf), t, m, events, clean)
                    for k in ("caught", "caught_mvgs", "nominal_flagged"):
                        grid_acc[name][i][k] += p[k]

            for name in series:
                fired = masks[name]
                arm_hits[name]["nominal"] += int((fired & clean).sum())

            for event, a_, b_ in events:
                if b_ <= a_:
                    continue
                totals["events"] += 1
                totals["mvgs"] += int(event.cell == MVGS)
                row = {"event_id": event.event_id, "fold": fold.index,
                       "cell": event.cell, "footprint": int(b_ - a_)}
                for name, (s, t) in series.items():
                    hit = bool(masks[name][a_:b_].any())
                    span = np.asarray(s[a_:b_], dtype=np.float64)
                    reach = float(np.max(span) / t) if np.isfinite(span).any() else None
                    row[name] = hit
                    row[f"{name}_reach"] = reach
                    if hit:
                        arm_hits[name]["events"] += 1
                        arm_hits[name]["mvgs"] += int(event.cell == MVGS)
                per_event.append(row)

        for name in ("max", RSTD):
            curves[name] = [{"multiplier": float(GRID[i]), **grid_acc[name][i],
                             "nominal_rate": grid_acc[name][i]["nominal_flagged"] / max(1, totals["clean"])}
                            for i in range(len(GRID))]
        # The gate: the `max` arm IS gru-quantile and `rstd` IS the floor. An
        # audit that cannot reproduce the scorecards it is extending is not
        # worth reading (docs/MODELS.md 22.6's rule, applied to this script).
        checks = []
        want = PUBLISHED.get(task.id)
        if want:
            for name in ("max", RSTD):
                got = (arm_hits[name]["events"], arm_hits[name]["mvgs"])
                checks.append({"arm": name, "expected": list(want[name]),
                               "measured": list(got), "ok": got == tuple(want[name])})
            checks.append({"arm": "denominators",
                           "expected": [want["events"], want["mvgs"]],
                           "measured": [totals["events"], totals["mvgs"]],
                           "ok": (totals["events"], totals["mvgs"]) == (want["events"], want["mvgs"])})
        out["sets"][task.id] = {"totals": totals, "arms": arm_hits, "reproduces": checks,
                                "curves": curves, "events": per_event}
        for c in checks:
            flag = "OK  " if c["ok"] else "FAIL"
            print(f"      reproduce {flag} {c['arm']:12s} expected {c['expected']} got {c['measured']}")
        print(f"    {task.id}: {totals['events']} events, {totals['mvgs']} MVGS, "
              f"{totals['clean']:,} clean nominal steps")
        for name in list(ARMS) + [RSTD]:
            h = arm_hits[name]
            print(f"      {name:5s} {h['events']:>3d}/{totals['events']:<3d} events   "
                  f"{h['mvgs']:>3d}/{totals['mvgs']:<3d} MVGS   "
                  f"nominal {h['nominal']:>7,} ({100*h['nominal']/max(1,totals['clean']):.4f}%)")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before and not args.allow_fit:
        print(f"  ABORT: weight store moved {before} -> {after}; nothing should have fitted.")
        return 6
    print(f"  weight store unchanged at {before} files")

    out["operations"] = budget.as_dict() if budget else {"class_a": 0, "class_b": 0,
                                                        "offline_fixture": True}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs") / primary.id / "_forensics" / f"{stamp}-reduction-and-curve.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"  artifact: {path}")

    # Artifact before ledger, always (commit 75cc846: a ledger failure once
    # destroyed 25 minutes of finished analysis).
    if state is None:
        print("  offline fixture: zero bucket operations, no ledger to commit")
        return 0
    cfg, ledger = state
    ops.commit(client, cfg.bucket, ledger, budget)
    print(f"  {budget.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
