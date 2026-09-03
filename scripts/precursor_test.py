"""Work item 9.9 study 2: are the forecaster's nominal-period alarms precursors?

Pre-registered in `docs/MODELS.md` 25, committed before any figure. S1 to S6.

D44 measured that a per-channel range check, widened until it is as noisy as the
forecaster, catches more on both Mission 1 sets and is never later. This tests the
forecaster's *other* output -- the alarms it raises in nominal periods, which every
scorecard here counts as false because an alarm outside a labelled span is a false
alarm by definition, and that definition has never been checked.

The population is in-range nominal time only: every watched channel strictly
inside its own training min/max (D43, strict rule per 24.9). That is the region a
limit check cannot see, so it is the only place such an alarm could be worth
anything D44's range check does not already deliver.

Primary, fixed in 25.2 and not movable afterwards: W = 10,000 timesteps, the
circular-shift null, the gate set, one-sided. A null result is the finding (S3).

    PYTHONPATH=src .venv/bin/python scripts/precursor_test.py
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
from sentinel_eval.detector import Context, Detector                   # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402

ANOMALY = "Anomaly"
GRU = "gru-quantile"
WINDOWS = (1_000, 10_000, 100_000)     # 25.2: primary is 10,000
PRIMARY_W = 10_000
DRAWS = 10_000
SEED = 0


def load(task, args):
    """One bundle load. The synthetic fixture never opens a client."""
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


def fold_state(view, fold):
    """Alarms, their attributed channel, and the eligible region. One forecast pass."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(GRU)
    det.fit(view.values[train_lo:train_hi], usable, fitting)

    tr_vals = view.values[train_lo:train_hi]
    tr_sm = np.asarray(det._smoothed_errors(tr_vals, fitting), dtype=np.float64)
    tr_max = np.nanmax(np.where(np.isfinite(tr_sm), tr_sm, -np.inf), axis=1)
    threshold = float(Detector.threshold_from(det, tr_max[usable] if usable.any() else tr_max))

    warmup = min(det.warmup_steps, test_lo)
    scored = context_for(view, fold, (test_lo, test_hi))
    te_sm = np.asarray(det._smoothed_errors(view.values[test_lo - warmup:test_hi], scored),
                       dtype=np.float64)[warmup:]
    finite = np.isfinite(te_sm)
    te_max = np.where(finite, te_sm, -np.inf).max(axis=1)
    attribution = np.where(finite, te_sm, -np.inf).argmax(axis=1)

    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    anomaly = np.asarray(view.truth.anomaly[test_lo:test_hi], dtype=bool)
    rare = np.asarray(view.truth.rare_event[test_lo:test_hi], dtype=bool)

    # 25.2's eligible region: nominal AND every channel strictly inside its own
    # training min/max -- the region a range check cannot see (D43).
    raw_rows = np.asarray(tr_vals, dtype=np.float64)
    raw_rows = raw_rows[usable] if usable.any() else raw_rows
    with np.errstate(invalid="ignore"):
        lo_c, hi_c = np.nanmin(raw_rows, axis=0), np.nanmax(raw_rows, axis=0)
        centre = (lo_c + hi_c) / 2.0
        half = np.maximum((hi_c - lo_c) / 2.0, np.finfo(np.float64).tiny)
        raw_test = np.asarray(view.values[test_lo:test_hi], dtype=np.float64)
        dev = np.abs(raw_test - centre) / half
        env = np.where(np.isfinite(dev), dev, -np.inf).max(axis=1)
    in_range = env <= 1.0                       # strict: touching is not leaving
    eligible = scorable & ~anomaly & ~rare & in_range

    alarm = (te_max >= threshold) & scorable
    starts = np.array([lo for lo, _ in mask_to_ranges(alarm)], dtype=np.int64)
    D.clear_caches()
    return {"starts": starts, "groups_at": attribution, "eligible": eligible,
            "T": int(test_hi - test_lo), "threshold": threshold,
            "nominal": int((scorable & ~anomaly & ~rare).sum()),
            "eligible_n": int(eligible.sum())}


def group_of(view):
    """channel_id -> ESA Group, the physical grouping HARNESS 6a rests on."""
    return {ch.channel_id: ch.group for ch in view.channels}


def targets(view, fold, state, W, chan_group):
    """Per group, the eligible steps inside W before a matching event's start.

    Windows are merged by construction -- a boolean array cannot count a step
    twice -- so an alarm preceding two events is counted once, as 25.2 requires.
    """
    test_lo, test_hi = fold.test
    T = state["T"]
    out = {}
    for event in view.truth.events:
        if event.category != ANOMALY or event.event_id not in view.truth.spans:
            continue
        lo, _ = view.truth.spans[event.event_id]
        if not (test_lo <= lo < test_hi):
            continue
        a = lo - test_lo
        groups = {chan_group[c] for c in event.channels if c in chan_group}
        for g in groups:
            m = out.setdefault(g, np.zeros(T, dtype=bool))
            m[max(0, a - W):a] = True
    for g in out:
        out[g] &= state["eligible"]
    return out


def group_selectors(starts, groups_at, chan_index_group, target):
    """Which alarm starts belong to which group. Fixed under a shift, so hoisted
    out of the 10,000-draw loop rather than recomputed inside it."""
    if starts.size == 0:
        return {}
    owner = np.array([chan_index_group[int(groups_at[s])] for s in starts])
    return {g: (owner == g) for g in target}


def statistic(starts, selectors, target, T, shift=0):
    """Alarm starts landing in their own group's target set, after a torus shift."""
    if starts.size == 0:
        return 0
    pos = (starts + shift) % T
    return int(sum(int(target[g][pos[sel]].sum()) for g, sel in selectors.items() if sel.any()))


def permute(starts, groups_at, chan_index_group, target, T, rng, eligible):
    """Null 1 circular shift (primary) and null 2 uniform placement (secondary)."""
    selectors = group_selectors(starts, groups_at, chan_index_group, target)
    obs = statistic(starts, selectors, target, T)
    shifts = rng.integers(0, T, size=DRAWS)
    null1 = np.array([statistic(starts, selectors, target, T, int(s)) for s in shifts],
                     dtype=np.int64)

    pool = np.flatnonzero(eligible)
    null2 = np.zeros(DRAWS, dtype=np.int64)
    if pool.size and starts.size:
        # Vectorised over draws: one (DRAWS, n) index matrix per group.
        for g, sel in selectors.items():
            n = int(sel.sum())
            if not n:
                continue
            picks = rng.choice(pool, size=(DRAWS, n), replace=True)
            null2 += target[g][picks].sum(axis=1).astype(np.int64)
    return obs, null1, null2


def summarise(obs, null):
    mean = float(null.mean()) if null.size else 0.0
    p = float((1 + int((null >= obs).sum())) / (1 + null.size)) if null.size else None
    return {"observed": int(obs), "null_mean": mean,
            "rate_ratio": (float(obs) / mean) if mean > 0 else None, "p": p}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    parser.add_argument("--allow-fit", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back (D29)."); return 2

    print(f"  Work item 9.9 study 2, the precursor test -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    rng = np.random.default_rng(SEED)

    out = {"task": primary.id, "pre_registration": "docs/MODELS.md 25",
           "primary": {"W": PRIMARY_W, "null": "circular shift", "set": primary.id,
                       "sided": "one", "draws": DRAWS, "seed": SEED},
           "windows": list(WINDOWS), "sets": {}}

    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        chan_group = group_of(view)
        idx_group = {i: chan_group[cid] for i, cid in enumerate(view.channel_ids)}
        per_w = {W: {"observed": 0, "null1": np.zeros(DRAWS, dtype=np.int64),
                     "null2": np.zeros(DRAWS, dtype=np.int64)} for W in WINDOWS}
        starts_total = nominal_total = eligible_total = 0

        for fold in split.folds:
            st = fold_state(view, fold)
            elig_starts = st["starts"][st["eligible"][st["starts"]]] if st["starts"].size else st["starts"]
            starts_total += int(elig_starts.size)
            nominal_total += st["nominal"]; eligible_total += st["eligible_n"]
            for W in WINDOWS:
                tgt = targets(view, fold, st, W, chan_group)
                o, n1, n2 = permute(st["starts"], st["groups_at"], idx_group, tgt,
                                    st["T"], rng, st["eligible"])
                per_w[W]["observed"] += o
                per_w[W]["null1"] += n1
                per_w[W]["null2"] += n2

        rows = {}
        for W in WINDOWS:
            d = per_w[W]
            rows[str(W)] = {"circular_shift": summarise(d["observed"], d["null1"]),
                            "uniform": summarise(d["observed"], d["null2"])}
        elig_frac = eligible_total / max(1, nominal_total)
        out["sets"][task.id] = {
            "nominal_steps": nominal_total, "eligible_steps": eligible_total,
            "eligible_fraction": elig_frac, "eligible_alarm_starts": starts_total,
            "underpowered": starts_total < 20, "windows": rows}

        print(f"    {task.id}: nominal {nominal_total:,}  eligible {eligible_total:,} "
              f"({100*elig_frac:.1f}%)  eligible alarm starts {starts_total}"
              f"{'   [UNDERPOWERED n<20]' if starts_total < 20 else ''}")
        for W in WINDOWS:
            cs, un = rows[str(W)]["circular_shift"], rows[str(W)]["uniform"]
            star = "  <- PRIMARY" if W == PRIMARY_W and task.id == primary.id else ""
            rr = f"{cs['rate_ratio']:.2f}x" if cs["rate_ratio"] else "-"
            rr2 = f"{un['rate_ratio']:.2f}x" if un["rate_ratio"] else "-"
            print(f"      W={W:>7,}  obs {cs['observed']:>4}   shift null {cs['null_mean']:8.2f} "
                  f"({rr:>7}, p={cs['p']:.4f})   uniform null {un['null_mean']:8.2f} "
                  f"({rr2:>7}, p={un['p']:.4f}){star}")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before and not args.allow_fit:
        print(f"  ABORT: weight store moved {before} -> {after}."); return 6
    print(f"  weight store unchanged at {before} files")

    out["operations"] = budget.as_dict() if budget else {"class_a": 0, "class_b": 0,
                                                         "offline_fixture": True}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs") / primary.id / "_forensics" / f"{stamp}-precursor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"  artifact: {path}")

    if state is None:
        print("  offline fixture: zero bucket operations, no ledger to commit")
        return 0
    cfg, ledger = state
    ops.commit(client, cfg.bucket, ledger, budget)
    print(f"  {budget.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
