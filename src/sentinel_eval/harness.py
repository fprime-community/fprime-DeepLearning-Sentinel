"""The scoring loop. One bundle, many detectors, many folds, identical treatment.

This is the referee. It fits each detector on a window stripped of annotated
anomalies, scores the fold that follows, turns scores into alarms at a
**label-free** threshold, and measures the result. Nothing here knows what kind
of model it is holding, which is the point: the architecture gate becomes a table
rather than an argument.

Four details that decide whether the numbers mean anything:

**Warm-up.** A window-based detector scoring a fold from its first timestep has
no history behind it. It is handed a prefix reaching back ``warmup_steps`` and
those scores are discarded, so a fold boundary cannot masquerade as an anomaly.

**An event is scored in exactly one fold** -- the one its span starts in --
matching the coverage census, so pooled ``k/n`` is a plain sum over disjoint sets
and no event is double-counted.

**Two thresholds, never conflated.** The detector picks its operating point from
*training* scores alone; that is the honest number. The harness separately sweeps
for the best achievable F0.5 and reports it explicitly labelled as an oracle
upper bound, because it saw the answers.

**Persistence.** Objective.md 7's filter: a signal must survive N consecutive
cycles before it counts as an alarm. Applied here rather than inside a detector,
so every candidate gets the same discipline. Default 1, meaning off, and always
recorded.
"""
from __future__ import annotations

import numpy as np

from .bundle import Bundle
from .detector import Context, Detector, reduce_scores
from .labels import ANOMALY, Event
from .metrics import eventwise, falsealarm, leadtime, vus
from .metrics.counts import Count, f_beta
from .metrics.ranges import Range, merge
from .scorecard import FoldResult, RunRecord, Scorecard
from .splits import Split, coverage, train_mask
from .tasks import Task

SWEEP_POINTS = 200


def apply_persistence(mask: np.ndarray, n: int) -> np.ndarray:
    """Keep only alarms that survive ``n`` consecutive timesteps.

    Objective.md 7: a single odd sample is noise. The filter is deliberately
    causal -- a timestep is confirmed by the ``n-1`` before it, never by what
    comes after -- because the flight component cannot see the future either.
    """
    if n <= 1:
        return np.asarray(mask, dtype=bool)
    flat = np.asarray(mask, dtype=bool).astype(np.int16)
    run = np.zeros_like(flat)
    total = 0
    for i, value in enumerate(flat):
        total = total + 1 if value else 0
        run[i] = total
    return run >= n


def _local_spans(events: list[Event], spans: dict[str, Range], lo: int, hi: int
                 ) -> tuple[list[Event], dict[str, Range]]:
    """Events starting inside ``[lo, hi)``, with spans rebased on the window."""
    kept, local = [], {}
    for event in events:
        span = spans.get(event.event_id)
        if span is None or not (lo <= span[0] < hi):
            continue
        kept.append(event)
        local[event.event_id] = (max(0, span[0] - lo), min(hi - lo, span[1] - lo))
    return kept, local


def _sweep_best_f_beta(scores: np.ndarray, events, spans, scorable, beta: float
                       ) -> tuple[float | None, float | None]:
    """Best F0.5 any threshold could have reached. An oracle: it saw the labels."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return None, None
    candidates = np.unique(np.quantile(finite, np.linspace(0.5, 1.0, SWEEP_POINTS)))
    best, best_at = None, None
    for threshold in candidates:
        result = eventwise.score(events, spans, scores >= threshold, scorable, beta=beta)
        value = result.f_beta
        if value is not None and (best is None or value > best):
            best, best_at = value, float(threshold)
    return best, best_at


def evaluate(bundle: Bundle, split: Split, detectors: list[Detector], task: Task,
             *, beta: float = 0.5, sweep: bool = True, log=None) -> RunRecord:
    """Score every detector on every fold of one already-loaded bundle."""
    truth = bundle.truth
    rows = coverage(split, truth)
    record = RunRecord(
        task=task.as_dict(),
        bundle=dict(bundle.provenance),
        coverage=[vars(r) for r in rows],
        limitation=task.limitation,
    )

    for detector in detectors:
        card = Scorecard(detector=detector.name, detector_params=detector.params,
                         fingerprint=detector.fingerprint())
        for fold in split.folds:
            if log:
                log(f"    {detector.name}  fold {fold.index}: "
                    f"fit {fold.train_steps:,} -> score {fold.test_steps:,}")
            card.folds.append(
                _score_fold(bundle, fold, detector, task, beta=beta, sweep=sweep)
            )
        card.pooled = _pool(card.folds, task, beta=beta)
        record.scorecards.append(card)
    return record


def _score_fold(bundle: Bundle, fold, detector: Detector, task: Task, *, beta: float,
                sweep: bool) -> FoldResult:
    truth = bundle.truth
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test

    def commands_for(lo: int, hi: int):
        """Telecommands sliced exactly as the values are, or None.

        Two different slices below -- the fit window, and the score window with
        its warm-up prefix -- so this is derived from the same bounds rather than
        carried on the context, where the two would drift apart silently.
        """
        return None if bundle.commands is None else bundle.commands[lo:hi]

    context = Context(
        mission=bundle.mission, channels=bundle.channel_ids, groups=bundle.groups,
        period_seconds=bundle.provenance["grid_period_seconds"], fold=fold.index,
        window=(train_lo, train_hi),
        commands=commands_for(train_lo, train_hi), command_ids=bundle.command_ids,
    )

    # -- fit on nominal data only ------------------------------------------
    usable = train_mask(fold, truth)[train_lo:train_hi]
    detector.fit(bundle.values[train_lo:train_hi], usable, context)

    # Whatever the detector recorded about its own fit. Optional by design: the
    # trivial baselines have nothing to say and their scorecards stay unchanged.
    # It is here because the defect that disabled training for every model in the
    # project announced itself only in this data -- eleven epochs with best_epoch
    # zero, every time -- and nothing was reading it.
    training = getattr(detector, "report", None)

    # -- a label-free operating point, from training scores ----------------
    train_raw = detector.score(bundle.values[train_lo:train_hi],
                               bundle.valid[train_lo:train_hi], context)
    train_scores, _ = reduce_scores(train_raw, train_hi - train_lo)
    threshold = detector.threshold_from(train_scores[usable] if usable.any() else train_scores)

    # -- score the fold, with warm-up history that is then discarded --------
    warmup = min(detector.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = Context(**{**vars(context), "window": (test_lo, test_hi),
                        "commands": commands_for(window_lo, test_hi)})
    raw = detector.score(bundle.values[window_lo:test_hi],
                         bundle.valid[window_lo:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - window_lo)
    scores = scores[warmup:]

    # -- alarms -------------------------------------------------------------
    predicted = np.zeros(scores.shape[0], dtype=bool) if threshold is None \
        else np.asarray(scores >= threshold)
    predicted = apply_persistence(predicted, task.persistence)

    events, spans = _local_spans(list(truth.events), truth.spans, test_lo, test_hi)
    scorable = truth.scorable[test_lo:test_hi]
    anomaly = truth.anomaly[test_lo:test_hi]
    rare = truth.rare_event[test_lo:test_hi]

    event_score = eventwise.score(events, spans, predicted, scorable, beta=beta) \
        if task.scores_recall else None
    alarms = falsealarm.score(events, spans, predicted, anomaly, rare, scorable)

    truth_ranges = merge([spans[e.event_id] for e in events
                          if e.category == ANOMALY and e.event_id in spans])
    volume, detail = (vus.vus_pr(np.where(scorable, scores, -np.inf), truth_ranges,
                                 scores.shape[0])
                      if (truth_ranges and task.scores_recall) else (None, {}))

    best, best_at = (_sweep_best_f_beta(scores, events, spans, scorable, beta)
                     if (sweep and task.scores_recall) else (None, None))

    # How early, not just whether -- Objective.md 4's claim is early warning and
    # nothing here had ever measured it. Additive: absent unless recall is scored,
    # so a scorecard that never had it is unchanged.
    lead = (leadtime.score(events, spans, predicted, scorable)
            if task.scores_recall else None)

    return FoldResult(
        fold=fold.index, window=(test_lo, test_hi), threshold=threshold,
        events=event_score, false_alarms=alarms, vus_pr=volume, vus_detail=detail,
        oracle_f_beta=best, oracle_threshold=best_at, lead_time=lead,
        training=training,
    )


def _pool(folds: list[FoldResult], task: Task, *, beta: float) -> dict:
    """Pool folds. Each event is scored once, so counts are a plain sum."""
    def total(pick) -> Count:
        out = Count(0, 0)
        for fold in folds:
            value = pick(fold)
            if value is not None:
                out = out + value
        return out

    pooled: dict = {
        "rare_event_false_alarms": total(lambda f: f.false_alarms.rare_events),
        "nominal_step_false_alarms": total(lambda f: f.false_alarms.nominal_steps),
        "persistence": task.persistence,
    }
    nominal = pooled["nominal_step_false_alarms"]
    alarms = sum(f.false_alarms.nominal_alarms for f in folds)
    pooled["alarms_per_1000_nominal_timesteps"] = (
        None if nominal.n == 0 else alarms * 1000 / nominal.n
    )

    if not task.scores_recall:
        pooled["recall"] = "disabled for this task by design"
        return pooled

    recall = total(lambda f: f.events.recall if f.events else None)
    precision = total(lambda f: f.events.precision if f.events else None)
    pooled.update({
        "event_recall": recall,
        "event_precision": precision,
        "event_f0.5": f_beta(precision, recall, beta),
        "contextual_recall": total(lambda f: f.events.contextual if f.events else None),
        "headline_cell_recall": total(lambda f: f.events.headline_cell if f.events else None),
        "point_recall": total(lambda f: f.events.point if f.events else None),
    })

    cells: dict[str, Count] = {}
    for fold in folds:
        if not fold.events:
            continue
        for cell, count in fold.events.by_cell.items():
            cells[cell] = cells.get(cell, Count(0, 0)) + count
    pooled["recall_by_cell"] = cells

    leads = [f.lead_time for f in folds if f.lead_time is not None]
    if leads:
        pooled["lead_time"] = leadtime.pool(leads)

    volumes = [f.vus_pr for f in folds if f.vus_pr is not None]
    pooled["vus_pr"] = float(np.mean(volumes)) if volumes else None
    bests = [f.oracle_f_beta for f in folds if f.oracle_f_beta is not None]
    pooled["oracle_best_f0.5"] = float(np.mean(bests)) if bests else None
    return pooled
