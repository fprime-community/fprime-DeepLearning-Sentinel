"""Which events each decision rule catches, event by event, and what they are.

`lstm-whitened` trades the adoption number for cross-channel recall: rare-event
false alarms 30/48 to 2/48, headline cell 28/32 to 21/32 on the gate set and
29/31 to 15/31 on the subset (`docs/RESULTS.md` 6d). The scorecard says *how
many* were lost and nothing about *which*, and the two possible causes want
opposite responses:

* **relationship breaks the rule was too strict to catch** -- loosen it, and find
  where recall returns before the false alarms do;
* **single-channel excursions** -- a whitened residual is a statement about the
  *joint* direction, so an event confined to one channel of the view has no
  relationship to break and the rule cannot see it in principle. That needs a
  second path, not a looser threshold.

`Event.segments_on(selected)` decides it. ESA labels an event `Multivariate` over
the whole 76-channel mission; what matters here is how many of **our** channels
it touches, and an event that reaches only one of them is single-channel *to this
detector* whatever the taxonomy says.

**It also answers the 6-versus-12 question in the same pass.** If the losses
cluster on events touching few channels, then `lstm-whitened` improves with more
channels, `m1-ss5`'s 15/31 is the pessimistic end of its range rather than its
typical behaviour, and 21/32 is closer to what a wider instance would do.

**And it measures the `error_buffer` lead-time inflation** (`docs/DECISIONS.md`
D21). Every telemanom-path alarm range is dilated by +/-99 timesteps before lead
time is measured from its start, so up to 99 timesteps of a reported +26 may be
the buffer. Both readings are taken per event: from the dilated range, and from
the first actual exceedance inside it. Lead time is a gate metric that
disqualifies configurations, so it has to be honest before architectures are
compared.

Cached weights, no refit. Both channel sets. Held-back sets are not loaded.
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
from sentinel_eval.metrics.ranges import mask_to_ranges                # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import telemanom, whiten                          # noqa: E402

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
                   fold=fold.index, window=window,
                   commands=None, command_ids=view.command_ids)


def telemanom_masks(e_s: np.ndarray, config: telemanom.Config):
    """`channel_ratios`' loop, keeping the dilated alarm AND the raw exceedance.

    The alarm is what the harness scores. The exceedance is the timestep that
    actually crossed the threshold, before `error_buffer` widened it by +/-99.
    The difference between where those two start is the buffer's contribution to
    every lead-time figure this project reports.
    """
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)
    alarm = np.zeros(steps, dtype=bool)
    exceed = np.zeros(steps, dtype=bool)

    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - span)
        window = e_s[reference_lo:seg_hi]
        offset = seg_lo - reference_lo
        eps, sequences = telemanom.dynamic_threshold(window, config)
        if not sequences:
            continue
        keeps = telemanom.prune(window, sequences, eps, config.pruning_p)
        for keep, (lo, hi) in zip(keeps, sequences):
            lo, hi = max(lo, offset), min(hi, window.shape[0])
            if keep and hi > lo:
                alarm[reference_lo + lo:reference_lo + hi] = True
        # The undilated crossings inside the judged segment only.
        crossed = window[offset:] >= eps
        if crossed.any():
            exceed[seg_lo:seg_hi] = crossed
    opening = min(telemanom.EWMA_SETTLE * config.smoothing_window, steps)
    alarm[:opening] = False
    exceed[:opening] = False
    return alarm, exceed


def whitened_mask(detector, values, context):
    pieces = [b for b in detector._signed_blocks(values, context)]
    signed = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    scored, _ = whiten.ratios(signed, detector.config, detector.whitening)
    return scored >= 1.0


def lead_of(ranges, exceed, start):
    """`(lead from the dilated range, lead from the first real exceedance)`."""
    overlapping = [(lo, hi) for lo, hi in ranges if lo < start or lo <= start]
    hit = [(lo, hi) for lo, hi in ranges if not (hi <= start or lo >= start + 1)]
    return hit, overlapping


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    print(f"  Event forensics -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        selected = set(view.channel_ids)
        rows = []
        for fold in split.folds:
            train_lo, train_hi = fold.train
            test_lo, test_hi = fold.test
            usable = train_mask(fold, view.truth)[train_lo:train_hi]
            fitting = context_for(view, fold, fold.train)

            tele = D.ForecastDetector()
            tele.fit(view.values[train_lo:train_hi], usable, fitting)
            whit = D.WhitenedDetector()
            whit.fit(view.values[train_lo:train_hi], usable, fitting)
            whit.score(view.values[train_lo:train_hi], view.valid[train_lo:train_hi],
                       fitting)                       # calibrates on the clean pool

            warmup = min(tele.warmup_steps, test_lo)
            window_lo = test_lo - warmup
            scored = context_for(view, fold, (test_lo, test_hi))
            values = view.values[window_lo:test_hi]

            smoothed = tele._smoothed_errors(values, scored)
            t_alarm = np.zeros(smoothed.shape[0], dtype=bool)
            t_exceed = np.zeros(smoothed.shape[0], dtype=bool)
            for c in range(smoothed.shape[1]):
                a, e = telemanom_masks(smoothed[:, c], tele.config)
                t_alarm |= a
                t_exceed |= e
            del smoothed
            w_alarm = whitened_mask(whit, values, scored)

            observed = np.isfinite(values).all(axis=1)
            observed &= np.asarray(view.valid[window_lo:test_hi], dtype=bool).all(axis=1)
            scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
            keep = lambda m: (m[warmup:] & observed[warmup:] & scorable)
            t_alarm, t_exceed, w_alarm = keep(t_alarm), keep(t_exceed), keep(w_alarm)
            t_ranges = mask_to_ranges(t_alarm)

            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a, b = lo - test_lo, min(hi - test_lo, t_alarm.shape[0])
                if b <= a:
                    continue
                on = sorted({s.channel for s in event.segments_on(selected)})
                caught_t = bool(t_alarm[a:b].any())
                caught_w = bool(w_alarm[a:b].any())
                lead_dilated = lead_raw = None
                if caught_t:
                    hits = [(x, y) for x, y in t_ranges if not (y <= a or x >= b)]
                    if hits:
                        first = min(x for x, _ in hits)
                        lead_dilated = int(a - first)
                        inside = np.flatnonzero(t_exceed[first:b])
                        if inside.size:
                            lead_raw = int(a - (first + int(inside[0])))
                rows.append({
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "dimensionality": event.dimensionality, "length": event.length,
                    "channels_in_view": len(on), "channels": on,
                    "channels_all": len(event.channels),
                    "footprint": int(hi - lo),
                    "telemanom": caught_t, "whitened": caught_w,
                    "lead_dilated": lead_dilated, "lead_undilated": lead_raw,
                })
            D.clear_caches()
        results[task.id] = rows
        print(f"    {task.id}: {len(rows)} scorable anomalies examined", flush=True)

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}; something trained."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    # The artifact is written BEFORE the ledger is committed, and the order is
    # load-bearing. It used to be the other way round -- the house pattern from
    # `decision_grid.py` -- and a transient `RequestTimeTooSkewed` on the ledger
    # PutObject destroyed twenty-five minutes of completed analysis that was
    # sitting in memory. The result is the expensive, unrepeatable thing; the
    # ledger is bookkeeping and can be retried. Never lose the first to the second.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-events.json"
    path.write_text(json.dumps({"task": primary.id,
                                "operations": budget.as_dict() if budget else None,
                                "sets": results}, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")

    if budget is not None:
        cfg, ledger = state
        try:
            ops.commit(client, cfg.bucket, ledger, budget)
        except Exception as failure:
            # Reported loudly and not swallowed: the ops actually happened, so a
            # ledger that does not record them understates real usage against a
            # hard ceiling (docs/DATA.md rule 4).
            print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
            print(f"      {budget.class_b} Class B and {budget.class_a} Class A were "
                  f"spent and are NOT recorded. The artifact above is safe.")
        else:
            print(); print(budget.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
