"""Per-event forensics on the 38 in-range contextual anomalies. DIAGNOSIS ONLY.

**Nothing here is pre-registered as an improvement and nothing here proposes one.**
D62 froze the pipeline on stage 4's configuration at 10 of 38, and nine arms failed
to beat it. This asks the prior question those arms skipped: *for each of the 38,
what actually went wrong?*

The classification it produces is the point, and the rule is fixed here:

    caught                 the frozen arm caught it
    lost in the decision   an ORACLE per-channel threshold, set at the frozen
      layer                arm's own pooled quiet rate, would have caught it
    invisible in the       the oracle would not have caught it either -- the
      residual             signal is not in the residual to be thresholded

That trichotomy decides the next work: the first two say the alarm rule, the third
says the network's inputs.

**One read, cached weights, no fits.** Only channels carrying one of the 38 are
loaded, so the read is smaller than a full study.

    PYTHONPATH=src .venv/bin/python scripts/smap_forensics_38.py \
        --visibility runs/smap-msl/_forensics/<stage1>.json \
        --stage4     runs/smap-msl/_forensics/<stage4>.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                   # noqa: E402
from sentinel_eval import ops                                           # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402
import torch                                                            # noqa: E402

#: The study script is the source of the frozen arm's own construction. Importing
#: it rather than re-deriving means the forensics cannot drift from the arm.
_spec = importlib.util.spec_from_file_location("smap_rungs",
                                               ROOT / "scripts" / "smap_rungs.py")
SR = importlib.util.module_from_spec(_spec)
sys.modules["smap_rungs"] = SR
_spec.loader.exec_module(SR)

#: 35.4's rule, unchanged and restated: the share of total variation carried by
#: the largest 1% of |dx| on the training split, cut at the median ACROSS the
#: channels measured here. Fixed before any event is examined.
STEP_CUT_SOURCE = "median of step_likeness across the channels in this run"


def horizon_predictions(det, values, chunk=512):
    """`(T, n_pred)` -- the ten predictions OF each timestep, un-aggregated.

    `aggregate_predictions` collapses these to one forecast per step; the spread
    across them is the quantity the goal calls "disagreement across the 10
    predicted horizons", and it cannot be recovered after aggregation.
    """
    hyper = det.hyper
    model = SR._to_torch(det._weights, hyper, 1, 0)
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    steps, w, npred = len(v), hyper.window, hyper.n_predictions
    made = np.full((steps, npred), np.nan, dtype=np.float64)
    starts = np.arange(w, steps)
    for lo in range(0, len(starts), chunk):
        idx = starts[lo:lo + chunk]
        hist = np.stack([v[t - w:t, 0] for t in idx])[:, :, None]
        with torch.no_grad():
            y = model(torch.from_numpy(hist.astype(np.float32)))
        made[idx] = np.asarray(y)[:, :, 0]          # prediction of t+1..t+npred
    out = np.full((steps, npred), np.nan, dtype=np.float64)
    for j in range(npred):                          # the prediction OF t made j+1 back
        src = np.arange(steps) - (j + 1)
        ok = src >= 0
        out[ok, j] = made[src[ok], j]
    return out


def attribute(e_s, cfg, lo, hi, warmup):
    """Which stage of the FROZEN path this event did not survive.

    The frozen path is `src/sentinel_models/telemanom.py`'s `channel_ratios`:
    a trailing window every `stride`, `dynamic_threshold` inside it, then
    `prune`. **It has no magnitude conjunct, no whole-window bail-out and no
    coverage or sequence cap** -- those exist only in the vendored source's
    `errors.py` and therefore in the port, not here (29.4 measured them not
    firing on MSL in any case). Reported as `not-in-this-path` rather than
    silently omitted.
    """
    if hi < warmup:
        return "warm-up", {}
    span, stride = cfg.error_window, max(1, cfg.stride)
    best = {"eps": np.inf, "candidate": False, "survived": False, "peak": 0.0}
    for seg_lo in range(0, len(e_s), stride):
        seg_hi = min(seg_lo + stride, len(e_s))
        if seg_hi <= lo or seg_lo > hi:
            continue
        ref = max(0, seg_lo - span)
        window = np.asarray(e_s[ref:seg_hi], dtype=np.float32)
        if window.size == 0:
            continue
        eps, seqs = telemanom.dynamic_threshold(window, cfg)
        peak = float(np.max(e_s[lo:hi + 1]))
        best["peak"] = peak
        best["eps"] = min(best["eps"], float(eps))
        if peak >= eps:
            best["candidate"] = True
            keep = telemanom.prune(window, seqs, eps, telemanom.PRUNING_P) if seqs else []
            for (a, b), k in zip(seqs, keep):
                if not (b + ref < lo or a + ref > hi) and k:
                    best["survived"] = True
    if best["survived"]:
        return "survived", best
    if best["candidate"]:
        return "pruning", best
    return "below-threshold", best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visibility", required=True)
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    vis = json.loads(Path(args.visibility).read_text())
    st4 = json.loads(Path(args.stage4).read_text())
    mult = float(st4["arms"]["gru"]["multiplier"])
    base_rate = float(st4["arms"]["gru"]["nominal_rate"])
    print(f"  frozen arm: multiplier {mult:.6f}, pooled nominal {100*base_rate:.4f}% "
          f"(read from {Path(args.stage4).name})")

    target = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
              if v["class"] == "contextual" and v["in_range"]]
    wanted = sorted({c for c, _, _ in target})
    print(f"  {len(target)} in-range contextual sequences across {len(wanted)} channels")

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(SR.fetch(client, cfg.bucket, SR.MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        SR.fetch(client, cfg.bucket, man["labels_key"]).decode())))
    spans, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen:
            continue
        seen.add(cid)
        spans[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]),
                              [x.strip() for x in r["class"].strip("[]").split(",")]))

    chans = [ch for ch in man["channels"] if ch["channel_id"] in wanted]
    if args.limit:
        chans = chans[: args.limit]
    store = SR.D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    rows, per_channel = [], {}
    for i, ch in enumerate(chans, 1):
        cid = ch["channel_id"]
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(SR.fetch(client, cfg.bucket, keys["train"])))
        test = np.load(io.BytesIO(SR.fetch(client, cfg.bucket, keys["test"])))
        try:
            det = SR.registry.build("gru-telemanom")
            det.config = SR.proportional_config(len(test))
            v_tr = train[:, :1].astype(np.float32)
            v_te = test[:, :1].astype(np.float32)
            det.fit(v_tr, np.ones(len(v_tr), dtype=bool), SR.ctx_for(cid, len(v_tr)))
            s_ctx = SR.ctx_for(cid, len(v_te))
            e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
            raw = np.asarray(SR.raw_errors(det, v_te, s_ctx), dtype=np.float64).ravel()
            as_src, _ = SR.reduce_scores(det.score(v_te, None, s_ctx), len(v_te))
            hz = horizon_predictions(det, v_te[:, 0])
        except Exception as exc:
            print(f"    {cid}: SKIP -- {type(exc).__name__}: {str(exc).splitlines()[0]}")
            continue
        x = v_te[:, 0].astype(np.float64)
        anom = np.zeros(len(x), dtype=bool)
        for (a, b), _cls in spans.get(cid, []):
            anom[a:b + 1] = True
        nominal = ~anom
        warm = int(det.warmup_steps)
        nominal[:warm] = False
        per_channel[cid] = dict(
            e_s=e_s, raw=raw, x=x, as_src=np.asarray(as_src, dtype=np.float64),
            hz=hz, warm=warm, nominal=nominal, cfg=det.config,
            step=SR.step_likeness(train[:, 0]),
            dx=np.abs(np.diff(x, prepend=x[0])),
            disagree=np.nanstd(hz, axis=1),
        )
        SR.D.clear_caches()
        print(f"    [{i}/{len(chans)}] {cid}")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"

    steps = sorted(v["step"] for v in per_channel.values())
    step_cut = float(np.median(steps)) if steps else 0.0

    for cid, start, end in target:
        p = per_channel.get(cid)
        if p is None:
            continue
        e_s, nom = p["e_s"], p["nominal"]
        lo, hi = start, min(end, len(e_s) - 1)
        if hi < lo:
            continue
        seg = slice(lo, hi + 1)
        nom_e = e_s[nom]
        peak = float(np.max(e_s[seg]))
        mu, sd = float(np.mean(nom_e)), float(np.std(nom_e))
        z = (peak - mu) / sd if sd > 0 else np.inf
        pct = float((nom_e < peak).mean() * 100.0) if nom_e.size else np.nan
        caught = bool((p["as_src"][seg] >= mult).any())
        stage, detail = attribute(e_s, p["cfg"], lo, hi, p["warm"])
        # the ORACLE: this channel's own threshold at the frozen arm's quiet rate
        oracle_thr = (float(np.quantile(nom_e, 1.0 - base_rate))
                      if nom_e.size else np.inf)
        oracle = bool(peak >= oracle_thr)
        # EWMA attribution: a raw peak that the smoothing flattened
        rz = ((float(np.max(p["raw"][seg])) - float(np.mean(p["raw"][nom])))
              / (float(np.std(p["raw"][nom])) or np.inf))
        dz = ((float(np.max(p["dx"][seg])) - float(np.mean(p["dx"][nom])))
              / (float(np.std(p["dx"][nom])) or np.inf))
        gz = ((float(np.nanmax(p["disagree"][seg])) - float(np.nanmean(p["disagree"][nom])))
              / (float(np.nanstd(p["disagree"][nom])) or np.inf))
        if caught:
            klass = "caught"
        elif oracle:
            klass = "lost-in-decision-layer"
        else:
            klass = "invisible-in-residual"
        rows.append(dict(
            channel=cid, start=start, end=end, length=hi - lo + 1,
            spacecraft=next(c["spacecraft"] for c in man["channels"]
                            if c["channel_id"] == cid),
            caught=caught, classification=klass, stage=stage,
            residual_peak=peak, residual_z=float(z), residual_percentile=pct,
            raw_residual_z=float(rz), derivative_z=float(dz), disagreement_z=float(gz),
            oracle_threshold=oracle_thr, oracle_would_catch=oracle,
            step_likeness=p["step"], step_like=bool(p["step"] >= step_cut),
            eps_seen=float(detail.get("eps", np.nan)),
            value_mean_in=float(np.mean(p["x"][seg])),
            value_mean_out=float(np.mean(p["x"][nom])) if nom.any() else np.nan,
            value_std_in=float(np.std(p["x"][seg])),
            value_std_out=float(np.std(p["x"][nom])) if nom.any() else np.nan,
        ))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = ROOT / "runs" / "smap-msl" / "_forensics" / f"{stamp}-forensics38.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(generated_utc=stamp, diagnosis_only=True,
               frozen_arm=dict(multiplier=mult, pooled_nominal_rate=base_rate,
                               source=Path(args.stage4).name),
               classification_rule=(
                   "caught / lost-in-decision-layer (an oracle per-channel threshold at "
                   "the frozen arm's own pooled quiet rate would catch it) / "
                   "invisible-in-residual (it would not)"),
               step_cut=step_cut, step_cut_source=STEP_CUT_SOURCE,
               stages_not_in_this_path=["magnitude floor", "whole-window bail-out",
                                        "coverage cap", "sequence cap"],
               events=rows, weight_store=dict(before=before, after=after))
    out["operations"] = budget.as_dict()
    path.write_text(json.dumps(out, indent=1))       # ARTIFACT BEFORE LEDGER
    print(f"\n  artifact {path.relative_to(ROOT)}")
    try:
        ops.commit(client, cfg.bucket, ledger, budget)
    except Exception as exc:
        print(f"  (!) LEDGER COMMIT FAILED: {exc}")
    print(budget.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
