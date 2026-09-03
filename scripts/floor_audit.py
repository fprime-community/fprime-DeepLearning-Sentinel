"""Work item 9.6: is the corrected floor real? Pre-registered in `docs/MODELS.md` 22.

Work item 9.5 corrected `baselines._rolling` and the floor moved from F0.5 0.250
to 0.676 and from 3/32 to 25/32, falsifying this project's central claim. This
audits that result the way a hostile reviewer would, before the restatement is
allowed to stand: a result that overturns a thesis has to survive more scrutiny
than the thesis did.

Four questions, one bundle load, cached weights, nothing refitted:

1. leakage      L1-L6   is `rstd` seeing the future, or catching stray ticks?
2. overlap      L7-L10  which events does each detector catch? The only-GRU set
                        is the thesis's remaining evidence.
3. adaptation   L11-L12 does the GRU's residual collapse inside a sustained
                        event while `rstd`'s does not?
4. Claim B      L13-L14 how many headline-cell events are TRULY contextual --
                        no channel outside its own training envelope -- and does
                        `rstd` catch those, or only the ones with a per-channel
                        signature?

Every definition is `docs/MODELS.md` 22.2's, fixed before this ran.

    PYTHONPATH=src .venv/bin/python scripts/floor_audit.py
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
from sentinel_eval.detector import Context, reduce_scores              # noqa: E402
from sentinel_eval.labels import LabelSet                              # noqa: E402
from sentinel_eval.splits import train_mask                            # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402
from sentinel_models.baselines import _rolling                         # noqa: E402

ANOMALY = "Anomaly"
LOW, HIGH = 0.001, 0.999          # 22.2's envelope, envelope_proxy.py's pair
RSTD_WINDOW = 120
EWMA_SPAN = 105
GRU = "gru-quantile"
RSTD = "rstd"


# ----------------------------------------------------------------------------
# 1. Leakage, the parts that need no bundle (L1, L3, L4)
# ----------------------------------------------------------------------------

def leakage_offline() -> dict:
    """L1 by perturbation, L3 and L4 by reading the code path."""
    import inspect
    from sentinel_eval.detector import Detector
    from sentinel_models.baselines import RollingStd

    rng = np.random.default_rng(19)
    values = rng.standard_normal((600, 4)).astype(np.float32)
    base = _rolling(values, RSTD_WINDOW, "std")

    # L1: change a future sample; every row at or before it must be untouched.
    worst_before = 0.0
    for t in (200, 350, 500):
        poked = values.copy()
        poked[t] += 1000.0
        after = _rolling(poked, RSTD_WINDOW, "std")
        worst_before = max(worst_before, float(np.abs(after[:t] - base[:t]).max()))
    # and the perturbed row itself MUST move, or the test proves nothing
    poked = values.copy(); poked[300] += 1000.0
    moved = float(np.abs(_rolling(poked, RSTD_WINDOW, "std")[300] - base[300]).max())

    return {
        "L1_future_leak_max_abs_change_before_t": worst_before,
        "L1_control_perturbed_row_did_move": moved,
        "L3_score_uses_fallback_scale_when_unfitted":
            "np.nanstd(spread" in inspect.getsource(RollingStd.score),
        "L4_threshold_from_is_base_class":
            RollingStd.threshold_from is Detector.threshold_from,
        "L4_threshold_source": inspect.getsource(Detector.threshold_from).strip(),
    }


# ----------------------------------------------------------------------------
# shared machinery, the harness's own path step for step
# ----------------------------------------------------------------------------

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
                   fold=fold.index, window=window, commands=None,
                   command_ids=view.command_ids)


def run_detector(name, view, fold):
    """Mask, combined scores, threshold AND per-channel scores over the test window."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, view.truth)[train_lo:train_hi]
    fitting = context_for(view, fold, fold.train)
    det = registry.build(name)
    det.fit(view.values[train_lo:train_hi], usable, fitting)
    train_raw = det.score(view.values[train_lo:train_hi],
                          view.valid[train_lo:train_hi], fitting)
    train_scores, _ = reduce_scores(train_raw, train_hi - train_lo)
    threshold = det.threshold_from(train_scores[usable] if usable.any() else train_scores)

    warmup = min(det.warmup_steps, test_lo)
    window_lo = test_lo - warmup
    scored = context_for(view, fold, (test_lo, test_hi))
    raw = det.score(view.values[window_lo:test_hi],
                    view.valid[window_lo:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - window_lo)
    scores = scores[warmup:]
    per_channel = np.asarray(raw)[warmup:] if np.asarray(raw).ndim == 2 else None
    scorable = np.asarray(view.truth.scorable[test_lo:test_hi], dtype=bool)
    D.clear_caches()
    return {"mask": np.asarray(scores >= threshold, dtype=bool) & scorable,
            "scores": np.where(scorable, scores, -np.inf),
            "threshold": float(threshold), "per_channel": per_channel,
            "usable": usable}


def runs_of(mask):
    """(count, longest, median length) of consecutive True runs."""
    if mask.size == 0 or not mask.any():
        return 0, 0, 0.0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    lengths = ends - starts
    return int(lengths.size), int(lengths.max()), float(np.median(lengths))


def half_life(scores, lo, hi):
    """Steps from the event's first scorable step to the first step at half peak."""
    seg = scores[lo:hi]
    seg = seg[np.isfinite(seg)]
    if seg.size < 3:
        return None
    peak_at = int(np.argmax(seg))
    peak = float(seg[peak_at])
    if not np.isfinite(peak) or peak <= 0:
        return None
    after = seg[peak_at:]
    below = np.where(after <= peak / 2.0)[0]
    return int(peak_at + below[0]) if below.size else None


def lead_of(mask, lo, hi):
    """event_start - first alarm step overlapping the event; 0 means at the boundary."""
    seg = mask[lo:hi]
    hit = np.where(seg)[0]
    return -int(hit[0]) if hit.size else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="m1-g8.9.10")
    parser.add_argument("--acknowledge-tripwire", action="store_true")
    args = parser.parse_args(argv)

    primary = tasks.get(args.task)
    if primary.id in ("m1-g3", "m2-ss1"):
        print(f"  REFUSED: {primary.id} is held back."); return 2

    offline = leakage_offline()
    print("  == leakage, offline ==")
    print(f"    L1 future perturbation, max change before t : "
          f"{offline['L1_future_leak_max_abs_change_before_t']:.3e}  "
          f"(control, the row itself moved by {offline['L1_control_perturbed_row_did_move']:.3e})")
    print(f"    L4 threshold_from is the base class          : {offline['L4_threshold_from_is_base_class']}")

    print(f"\n  Floor audit -- {primary.id}")
    loaded, labels, catalog, client, budget, state = load(primary, args)
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    results = {}
    for task, view in members(primary, catalog, labels, loaded):
        split = splits.forward_chaining(len(view.grid), **task.split_kwargs)
        rows = []
        for fold in split.folds:
            train_lo, train_hi = fold.train
            test_lo, test_hi = fold.test
            got = {n: run_detector(n, view, fold) for n in (GRU, RSTD)}

            # 22.2's envelope and per-channel variance threshold, fitting data only
            usable = got[RSTD]["usable"]
            fit_values = view.values[train_lo:train_hi]
            rows_ok = fit_values[usable] if usable.any() else fit_values
            with np.errstate(invalid="ignore"):
                env_low = np.nanquantile(rows_ok, LOW, axis=0)
                env_high = np.nanquantile(rows_ok, HIGH, axis=0)
                env_min = np.nanmin(rows_ok, axis=0)
                env_max = np.nanmax(rows_ok, axis=0)
                fit_spread = _rolling(fit_values, RSTD_WINDOW, "std")
                chan_var_cut = np.nanquantile(
                    fit_spread[usable] if usable.any() else fit_spread, HIGH, axis=0)

            test_values = view.values[test_lo:test_hi]
            test_spread = _rolling(view.values[max(0, test_lo - RSTD_WINDOW):test_hi],
                                   RSTD_WINDOW, "std")[min(test_lo, RSTD_WINDOW):]

            for event in view.truth.events:
                if event.category != ANOMALY or event.event_id not in view.truth.spans:
                    continue
                lo, hi = view.truth.spans[event.event_id]
                if not (test_lo <= lo < test_hi):
                    continue
                a_, b_ = lo - test_lo, min(hi - test_lo, got[GRU]["mask"].shape[0])
                if b_ <= a_:
                    continue

                span = test_values[a_:b_]
                spread_span = test_spread[a_:b_]
                with np.errstate(invalid="ignore"):
                    breach_q = bool(np.any(
                        (span < env_low) | (span > env_high))) if span.size else False
                    breach_m = bool(np.any(
                        (span < env_min) | (span > env_max))) if span.size else False
                    var_sig = bool(np.any(spread_span > chan_var_cut)) if spread_span.size else False
                    breach_channels = int(np.sum(
                        ((span < env_low) | (span > env_high)).any(axis=0))) if span.size else 0

                r_runs = runs_of(got[RSTD]["mask"][a_:b_])
                row = {
                    "event_id": event.event_id, "fold": fold.index, "cell": event.cell,
                    "footprint": int(hi - lo),
                    "gru": bool(got[GRU]["mask"][a_:b_].any()),
                    "rstd": bool(got[RSTD]["mask"][a_:b_].any()),
                    "gru_reach": float(np.max(got[GRU]["scores"][a_:b_]) / got[GRU]["threshold"])
                        if np.isfinite(got[GRU]["scores"][a_:b_]).any() else None,
                    "rstd_reach": float(np.max(got[RSTD]["scores"][a_:b_]) / got[RSTD]["threshold"])
                        if np.isfinite(got[RSTD]["scores"][a_:b_]).any() else None,
                    "rstd_runs": r_runs[0], "rstd_longest_run": r_runs[1],
                    "rstd_median_run": r_runs[2],
                    "gru_lead": lead_of(got[GRU]["mask"], a_, b_),
                    "rstd_lead": lead_of(got[RSTD]["mask"], a_, b_),
                    "gru_half_life": half_life(got[GRU]["scores"], a_, b_),
                    "rstd_half_life": half_life(got[RSTD]["scores"], a_, b_),
                    "envelope_breach_quantile": breach_q,
                    "envelope_breach_minmax": breach_m,
                    "channels_breaching": breach_channels,
                    "variance_signature": var_sig,
                }
                rows.append(row)
            print(f"    {task.id} fold {fold.index} done", flush=True)
        results[task.id] = rows

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    if after != before:
        print(f"  ABORT: weight store {before} -> {after}."); return 6
    print(f"  weight store unchanged at {after} -- nothing was fitted")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = C.PROJECT_ROOT / "runs" / primary.id / "_forensics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}-floor-audit.json"
    path.write_text(json.dumps({
        "task": primary.id, "pre_registration": "docs/MODELS.md 22",
        "leakage_offline": offline,
        "definitions": {"envelope_quantiles": [LOW, HIGH], "rstd_window": RSTD_WINDOW,
                        "ewma_span": EWMA_SPAN},
        "operations": budget.as_dict() if budget else None,
        "sets": results}, indent=2, default=str) + "\n")
    print(f"\n  wrote {path.relative_to(C.PROJECT_ROOT)}")
    try:
        if budget is not None:
            cfg, ledger = state
            ops.commit(client, cfg.bucket, ledger, budget)
            print(); print(budget.report())
    except Exception as failure:
        print(f"\n  (!) LEDGER NOT COMMITTED: {failure}")
        print("      operations spent and NOT recorded; the artifact above is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
