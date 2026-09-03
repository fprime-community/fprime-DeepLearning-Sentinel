"""Work item 9.7 studies A and B (M1-M9) and 9.8 parts 1-4 (N1-N8).

Pre-registered in `docs/MODELS.md` 23 and 24, both committed before any figure.

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

9.8 adds, from the same single load and the same cached forecast pass:

    part 1  min/max-contextual, which is the w = 1.0 case of part 2's envelope
    part 2  a per-channel range check widened until it is as noisy as we are,
            and the per-event lead of our emission over its first break
    part 3  the raw channel's z beside the residual's, at our first crossing
    part 4  each channel against its OWN 99.9th percentile, then max, cut at 1.0

and closes work item 9.7's M4 and M9 by scoring event-wise F0.5 along the sweep,
which 23.15 could not do because alarm ranges were never classified.

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
from sentinel_eval.harness import _local_spans                         # noqa: E402
from sentinel_eval.metrics import eventwise                            # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402

ANOMALY = "Anomaly"
GRU = "gru-quantile"
RSTD = "rstd"
MVGS = "Multivariate/Global/Subsequence"
ARMS = ("max", "l2", "sum", "k2", "k3", "perchan")
#: Series whose whole threshold curve is swept. `envelope` is work item 9.8
#: part 2's per-channel range check; `perchan` is part 4's per-channel bar.
SWEPT = ("max", "perchan", "envelope", "rstd")
ENVELOPE = "envelope"
#: Series whose alarm rule is STRICT. An event that touches a channel's historical
#: extreme has not *left* the envelope, and ESA-ADB is min-max scaled per group, so
#: values sitting exactly on a training min or max are common rather than rare --
#: 14 of 42 events on `m1-ss5`. `scripts/envelope_proxy.py:115` uses
#: `(values < lo) | (values > hi)`, `scripts/floor_audit.py` uses the same, and
#: D39, D40 and 22.11 are all published on that reading. A non-strict comparison
#: here silently redefined "contextual" and disagreed with the floor audit on 2
#: gate-set and 14 subset events, every one of them at a reach of exactly 1.000.
STRICT = (ENVELOPE,)


def fires(scores, level, name):
    """Alarm rule. Strict for the envelope, `>=` for every score-based series."""
    return scores > level if name in STRICT else scores >= level
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


def per_channel_bar(smoothed: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Work item 9.8 part 4: D25's recipe applied per column, not a new recipe.

    `Detector.threshold_from` cannot do this. Its `np.isfinite` mask flattens a
    (T, C) matrix and it returns one scalar pooled over every channel and step
    (`src/sentinel_eval/detector.py:127-132`), which is precisely the global bar
    part 4 exists to question. So the 0.999 quantile is taken with `axis=0`
    explicitly, on the anomaly-masked fitting window, per channel.

    This normalises *residuals*, not input values. The per-channel scaler ban
    (`sentinel_eval/normalisation.py`, `tests/test_no_per_channel_scaler.py`, D2)
    forbids rescaling inputs because it would erase the amplitude ratios between
    related channels -- the information the cross-channel claim rests on. Those
    ratios have already been used by the time a residual exists, and
    `telemanom.py` and `oscfar.py` both already normalise residuals per channel.
    """
    rows = smoothed[usable] if usable.any() else smoothed
    with np.errstate(invalid="ignore"):
        bar = np.nanquantile(np.asarray(rows, dtype=np.float64), 0.999, axis=0)
    return np.maximum(bar, np.finfo(np.float64).tiny)


def envelope_score(values: np.ndarray, centre: np.ndarray, half: np.ndarray
                   ) -> np.ndarray:
    """Part 2's range check, expressed so a threshold sweep widens the envelope.

    `max_c |x_c - centre_c| / half_c >= w` is exactly "some channel is outside
    the training min/max widened by `w`", so sweeping the multiplier on a
    threshold of 1.0 sweeps the envelope width and the existing curve machinery
    applies unchanged. The alarm rule is `scripts/envelope_proxy.py:113-119`'s
    `first_break`, evaluated over the whole test window rather than only inside
    event spans -- which is what gives it a nominal-step rate to be matched on,
    and is the reason that script could not be reused as it stands.
    """
    x = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs(x - centre) / half
        finite = np.isfinite(dev)
        out = np.where(finite, dev, -np.inf).max(axis=1)
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
    tr_smoothed = det._smoothed_errors(tr_vals, fitting)
    tr = reduce_arms(tr_smoothed, observed_mask(tr_vals, tr_valid))

    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    te_vals, te_valid = view.values[window_lo:test_hi], view.valid[window_lo:test_hi]
    te_smoothed = det._smoothed_errors(te_vals, scored)
    te_all = reduce_arms(te_smoothed, observed_mask(te_vals, te_valid))
    te = {k: v[warmup:] for k, v in te_all.items()}

    thresholds = {k: float(Detector.threshold_from(det, tr[k][usable] if usable.any() else tr[k]))
                  for k in ARMS if k != "perchan"}

    # -- part 4: each channel against its own bar, then max, cut at 1.0 -----
    bar = per_channel_bar(tr_smoothed, usable)
    te_matrix = np.asarray(te_smoothed, dtype=np.float64)[warmup:]
    with np.errstate(invalid="ignore"):
        ratio = te_matrix / bar
        seen = np.isfinite(ratio)
        te["perchan"] = np.where(seen, ratio, -np.inf).max(axis=1)
    thresholds["perchan"] = 1.0        # the cut IS 1.0; 24.2 fixes it

    # -- parts 2 and 3: the fitting window's own statistics, per channel ----
    raw_rows = np.asarray(tr_vals, dtype=np.float64)
    raw_rows = raw_rows[usable] if usable.any() else raw_rows
    res_rows = np.asarray(tr_smoothed, dtype=np.float64)
    res_rows = res_rows[usable] if usable.any() else res_rows
    with np.errstate(invalid="ignore"):
        lo_c, hi_c = np.nanmin(raw_rows, axis=0), np.nanmax(raw_rows, axis=0)
        stats = {
            "env_centre": (lo_c + hi_c) / 2.0,
            "env_half": np.maximum((hi_c - lo_c) / 2.0, np.finfo(np.float64).tiny),
            "raw_mean": np.nanmean(raw_rows, axis=0),
            "raw_std": np.maximum(np.nanstd(raw_rows, axis=0), np.finfo(np.float64).tiny),
            "res_std": np.maximum(np.nanstd(res_rows, axis=0), np.finfo(np.float64).tiny),
            "chan_bar": bar,
        }
    D.clear_caches()
    return te, thresholds, usable, te_matrix, stats


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


def first_reach_index(series, level, strict=False):
    """First index at which `series` reaches `level`, or None. O(log n).

    A running maximum is non-decreasing, so `searchsorted` finds the crossing
    without one pass per level. `strict` selects `>` over `>=` -- see STRICT.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.size == 0:
        return None
    run = np.maximum.accumulate(np.where(np.isfinite(x), x, -np.inf))
    i = int(np.searchsorted(run, level, side="right" if strict else "left"))
    if i >= run.size:
        return None
    return i if (run[i] > level if strict else run[i] >= level) else None


def amplitude_at(cross, a_, te_matrix, raw_test, stats):
    """Part 3: the raw channel's z beside the residual's, at the first crossing.

    The channel is the one attaining the maximum smoothed residual at that step
    -- the channel that actually raised the alarm. If the residual crosses while
    the raw value is still ordinary, the alarm is not an amplitude excursion; if
    it does not, this detector is an expensive magnitude detector and 24.1 part 3
    says so in those words.
    """
    if cross is None:
        return {"cross_channel": None, "raw_z": None, "residual_z": None}
    t = a_ + int(cross)
    if t >= te_matrix.shape[0] or t >= raw_test.shape[0]:
        return {"cross_channel": None, "raw_z": None, "residual_z": None}
    row = te_matrix[t]
    if not np.isfinite(row).any():
        return {"cross_channel": None, "raw_z": None, "residual_z": None}
    c = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))
    raw = raw_test[t, c]
    with np.errstate(invalid="ignore"):
        raw_z = (raw - stats["raw_mean"][c]) / stats["raw_std"][c]
        res_z = row[c] / stats["res_std"][c]
    return {"cross_channel": c,
            "raw_z": float(raw_z) if np.isfinite(raw_z) else None,
            "residual_z": float(res_z) if np.isfinite(res_z) else None}


def local_events(view, test_lo, test_hi):
    out = []
    for event in view.truth.events:
        if event.category != ANOMALY or event.event_id not in view.truth.spans:
            continue
        lo, hi = view.truth.spans[event.event_id]
        if test_lo <= lo < test_hi:
            out.append((event, lo - test_lo, min(hi - test_lo, test_hi - test_lo)))
    return out


def curve_point(scores, threshold, mult, events, clean, objs, spans, scorable,
                name="max"):
    """Recall, headline-cell recall, nominal rate -- and event-wise F0.5.

    F0.5 and precision are what work item 9.7's M4 and M9 needed and 23.15 could
    not answer, because that sweep recorded events caught and nominal steps
    flagged and never classified an alarm range. They come from the referee's own
    `eventwise.score`, so a curve point is directly comparable to a scorecard.
    """
    fired = fires(scores, threshold * mult, name)
    caught = [e for e, a_, b_ in events if b_ > a_ and fired[a_:b_].any()]
    ev = eventwise.score(objs, spans, fired, scorable, beta=0.5)
    row = ev.as_dict() if hasattr(ev, "as_dict") else {}
    return {
        "multiplier": float(mult),
        "caught": len(caught),
        "caught_mvgs": sum(1 for e in caught if e.cell == MVGS),
        "nominal_flagged": int((fired & clean).sum()),
        "eventwise": row,
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

    print(f"  Work items 9.7 A/B and 9.8 parts 1-4 -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    out = {"task": primary.id, "pre_registration": ["docs/MODELS.md 23", "docs/MODELS.md 24"],
           "arms": list(ARMS), "swept": list(SWEPT), "sets": {}}

    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        per_event, curves, totals = [], {}, {"events": 0, "mvgs": 0, "clean": 0}
        arm_hits = {a: {"events": 0, "mvgs": 0, "nominal": 0} for a in ARMS}
        for extra in (RSTD, ENVELOPE):
            arm_hits[extra] = {"events": 0, "mvgs": 0, "nominal": 0}
        grid_acc = {name: {i: {"caught": 0, "caught_mvgs": 0, "nominal_flagged": 0,
                               "tp": 0, "pred": 0, "truth": 0}
                           for i in range(len(GRID))} for name in SWEPT}

        for fold in split.folds:
            test_lo, test_hi = fold.test
            te, thr, _, te_matrix, stats = gru_arms(view, fold, args.detector)
            r_scores, r_thr = rstd_scores(view, fold)

            scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
            anomaly = np.asarray(view.truth.anomaly[test_lo:test_hi], dtype=bool)
            rare = np.asarray(view.truth.rare_event[test_lo:test_hi], dtype=bool)
            clean = scorable & ~anomaly & ~rare
            events = local_events(view, test_lo, test_hi)
            objs, spans = _local_spans(list(view.truth.events), view.truth.spans,
                                       test_lo, test_hi)
            totals["clean"] += int(clean.sum())

            # -- part 2: the per-channel range check over the whole window ---
            raw_test = np.asarray(view.values[test_lo:test_hi], dtype=np.float64)
            env = envelope_score(raw_test, stats["env_centre"], stats["env_half"])

            series = {a: (te[a], thr[a]) for a in ARMS}
            series[RSTD] = (r_scores, r_thr)
            series[ENVELOPE] = (env, 1.0)
            masks = {n: fires(s, t, n) & scorable for n, (s, t) in series.items()}

            for name in SWEPT:
                s, t = series[name]
                masked = np.where(scorable, s, -np.inf)
                for i, m in enumerate(GRID):
                    pt = curve_point(masked, t, m, events, clean, objs, spans,
                                     scorable, name)
                    for k in ("caught", "caught_mvgs", "nominal_flagged"):
                        grid_acc[name][i][k] += pt[k]
                    # Pooled the way `harness.py` pools folds: sum k and n, then
                    # recompute the rate. Each event is scored in exactly one fold.
                    ew = pt["eventwise"]
                    rec, pre = ew.get("event_recall", {}), ew.get("event_precision", {})
                    grid_acc[name][i]["tp"] += int(rec.get("k") or 0)
                    grid_acc[name][i]["truth"] += int(rec.get("n") or 0)
                    grid_acc[name][i]["hit"] = grid_acc[name][i].get("hit", 0) + int(pre.get("k") or 0)
                    grid_acc[name][i]["pred"] += int(pre.get("n") or 0)

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
                    if hit and name in ARMS + (RSTD, ENVELOPE):
                        arm_hits[name]["events"] += 1
                        arm_hits[name]["mvgs"] += int(event.cell == MVGS)

                # -- part 2: first crossing / first envelope break, per multiplier
                #
                # A running maximum is non-decreasing, so `searchsorted` gives the
                # first step at which a series reaches a level in O(log n) instead
                # of one pass per level. Storing the whole grid rather than one
                # operating point means the lead can be read at any matched rate
                # afterwards, including one this run does not name.
                row["gru_first_cross"] = first_reach_index(te["max"][a_:b_], thr["max"])
                env_span = env[a_:b_]
                row["env_first_break"] = [first_reach_index(env_span, w, strict=True)
                                          for w in GRID]

                # -- part 1: min/max-contextual is the w = 1.0 case of the same
                # envelope, so it costs nothing and cross-checks the floor audit.
                one = int(np.searchsorted(GRID, 1.0))
                row["minmax_contextual"] = row["env_first_break"][one] is None

                # -- part 3: the amplitude mechanism at our first crossing -------
                row.update(amplitude_at(row["gru_first_cross"], a_, te_matrix,
                                        raw_test, stats))
                per_event.append(row)

        for name in SWEPT:
            rows = []
            for i in range(len(GRID)):
                g = grid_acc[name][i]
                r = g["tp"] / g["truth"] if g["truth"] else None
                pr = g.get("hit", 0) / g["pred"] if g["pred"] else None
                f = None
                if r is not None and pr is not None and (0.25 * pr + r) > 0:
                    f = (1 + 0.25) * pr * r / (0.25 * pr + r)
                rows.append({"multiplier": float(GRID[i]), **g,
                             "nominal_rate": g["nominal_flagged"] / max(1, totals["clean"]),
                             "event_recall": r, "event_precision": pr, "event_f0.5": f})
            curves[name] = rows
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
                                "grid": [float(x) for x in GRID],
                                "curves": curves, "events": per_event}
        for c in checks:
            flag = "OK  " if c["ok"] else "FAIL"
            print(f"      reproduce {flag} {c['arm']:12s} expected {c['expected']} got {c['measured']}")
        print(f"    {task.id}: {totals['events']} events, {totals['mvgs']} MVGS, "
              f"{totals['clean']:,} clean nominal steps")
        for name in list(ARMS) + [RSTD, ENVELOPE]:
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
