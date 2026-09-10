#!/usr/bin/env python3
"""A1, A5 and A6: the pruning ladder, the alarm rate beside every figure, and lead time.

Pre-registered in `docs/MODELS.md` 37.7. **DIAGNOSIS ONLY.** Nothing here is an arm,
nothing is adjudicated, and D62's freeze is untouched by anything this script prints.

Three quantities the committed forensic could not answer, and why:

  A1  the smallest `p` that would have retained each of the 13 pruned events.
      `2026-09-09T230300Z-forensics38.json` keeps each event's peak and the threshold
      it was judged against, but **not the ladder** -- the other candidates' peaks and
      `non_anom_max` in the deciding window -- and 37.2 shows the retention rule is a
      property of the whole ladder: a sequence at rank `r` survives iff
      `max(drop_i : i >= r) >= p`. So this script records the ladder.
  A5  the pooled nominal alarm rate beside every recall figure (D41). That rate is over
      **all 81 channels** and the forensic loaded 26.
  A6  lead time in timesteps: emission step minus labelled onset. `attribute()` returned
      a stage and a peak and retained no timestep.

**The forecast is computed once.** Sweeping `p` afterwards re-decides pruning from the
cached ladders and touches no network, which is the argument 33.5 made and 33.7
confirmed: a sweep inside one load is compute, not operations.

**It also settles a question the vendored source cannot** (37.5): `channel.py:69-82`
loads the `.npy` arrays and never scales them, so whether the published `(-1,1)` map was
fitted once on the test split and applied to both arrays, or fitted separately on each,
is not answerable from telemanom's code. It is answerable from the arrays, and the
per-split extrema below answer it.

Cost: one bundle load, cached weights, **no fits**. 165 Class B and 1 Class A over 81
channels; `--limit 2` is the 7 Class B smoke. The weight store is asserted unmoved
rather than reported.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                   # noqa: E402
from sentinel_eval import ops                                           # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402

_spec = importlib.util.spec_from_file_location("smap_rungs",
                                               ROOT / "scripts" / "smap_rungs.py")
SR = importlib.util.module_from_spec(_spec)
sys.modules["smap_rungs"] = SR
_spec.loader.exec_module(SR)

#: The sweep grid for `p`. Published telemanom is 0.13; 0.0 is no pruning at all
#: (37.2: every normalised drop is >= 0, so nothing is ever accumulated for removal).
P_GRID = [round(x, 4) for x in np.arange(0.0, 0.3001, 0.005)]


def ladders(e_s, cfg):
    """Every deciding window's pruning ladder, and each sequence's survival bound.

    Mirrors `telemanom.channel_ratios`' trailing-window loop exactly -- same `span`,
    same `stride`, same guard-cell branch -- but records, per candidate sequence:
    its span on the timeline, its peak, and `M_r = max(drop_i : i >= r)`, the largest
    normalised drop at or below its rank. By 37.2 the sequence survives pruning for
    every `p <= M_r` and is deleted for every `p > M_r`, so one number per sequence
    replaces re-running `prune` at every grid point.
    """
    e_s = np.asarray(e_s, dtype=np.float32)
    steps = e_s.shape[0]
    span, stride = cfg.error_window, max(1, cfg.stride)
    out = []
    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        ref_lo = max(0, seg_lo - span)
        window = e_s[ref_lo:seg_hi]
        offset = seg_lo - ref_lo
        if cfg.guard_segment and offset > 0:
            eps, _ = telemanom.dynamic_threshold(window[:offset], cfg)
            seqs = telemanom.sequences_at(window, eps, cfg)
        else:
            eps, seqs = telemanom.dynamic_threshold(window, cfg)
        if not seqs:
            continue
        # the ladder, exactly as `telemanom.prune` builds it
        peaks = np.array([float(np.max(window[lo:hi])) for lo, hi in seqs])
        below = window[window < eps]
        normal_max = float(np.max(below)) if below.size else 0.0
        order = np.argsort(peaks)[::-1]
        rungs = np.append(peaks[order], normal_max)
        drops = [float((rungs[i] - rungs[i + 1]) / rungs[i]) if rungs[i] else 0.0
                 for i in range(len(rungs) - 1)]
        suffix_max = [0.0] * len(drops)
        run = 0.0
        for i in range(len(drops) - 1, -1, -1):        # M_r = max(drop_i : i >= r)
            run = max(run, drops[i])
            suffix_max[i] = run
        for rank, idx in enumerate(order):
            lo, hi = seqs[idx]
            lo_c, hi_c = max(lo, offset), min(hi, window.shape[0])
            if hi_c <= lo_c:
                continue                                # reference history, not judged here
            out.append(dict(seg_lo=int(seg_lo), seg_hi=int(seg_hi),
                            lo=int(ref_lo + lo_c), hi=int(ref_lo + hi_c),
                            peak=float(peaks[idx]), eps=float(eps),
                            normal_max=normal_max, survives_upto_p=float(suffix_max[rank]),
                            emits_at=int(seg_hi - 1),
                            emitted=bool(np.any(window[offset:] >= eps))))
    return out


def alarm_mask(lad, steps, p):
    """The alarm mask at pruning parameter `p`, from cached ladders. No network."""
    m = np.zeros(steps, dtype=bool)
    for s in lad:
        if s["survives_upto_p"] >= p:
            m[s["lo"]:s["hi"]] = True
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visibility", required=True)
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: number of channels")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args(argv)

    vis = json.loads(Path(args.visibility).read_text())
    st4 = json.loads(Path(args.stage4).read_text())
    mult = float(st4["arms"]["gru"]["multiplier"])
    base_rate = float(st4["arms"]["gru"]["nominal_rate"])
    print(f"  frozen arm: multiplier {mult:.6f}, pooled nominal {100*base_rate:.4f}%")

    target = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
              if v["class"] == "contextual" and v["in_range"]]
    print(f"  {len(target)} in-range contextual sequences on {len({c for c,_,_ in target})} channels")

    cfg_r2 = C.load_r2_config()
    client, budget = ops.connect(cfg_r2)
    ledger = ops.load(client, cfg_r2.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(SR.fetch(client, cfg_r2.bucket, SR.MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        SR.fetch(client, cfg_r2.bucket, man["labels_key"]).decode())))
    spans, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen:
            continue
        seen.add(cid)
        spans[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]),
                              [x.strip() for x in r["class"].strip("[]").split(",")]))

    chans = list(man["channels"])
    if args.limit:
        chans = chans[: args.limit]
    store = SR.D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")

    per, extrema, skipped = {}, [], {}
    t0 = time.time()
    for i, ch in enumerate(chans, 1):
        cid = ch["channel_id"]
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["train"])))
        test = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["test"])))
        # 37.5 / item 2: the per-split extrema of column 0, before anything else
        extrema.append(dict(channel=cid,
                            train_min=float(train[:, 0].min()), train_max=float(train[:, 0].max()),
                            test_min=float(test[:, 0].min()), test_max=float(test[:, 0].max())))
        t1 = time.time()
        try:
            det = SR.registry.build("gru-telemanom")
            det.config = SR.proportional_config(len(test))
            v_tr = train[:, :1].astype(np.float32)
            v_te = test[:, :1].astype(np.float32)
            det.fit(v_tr, np.ones(len(v_tr), dtype=bool), SR.ctx_for(cid, len(v_tr)))
            s_ctx = SR.ctx_for(cid, len(v_te))
            e_s = np.asarray(det._smoothed_errors(v_te, s_ctx), dtype=np.float64)[:, 0]
        except Exception as exc:
            skipped[cid] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"    [{i}/{len(chans)}] {cid}: SKIP -- {skipped[cid][:70]}")
            continue
        anom = np.zeros(len(v_te), dtype=bool)
        for (a, b), _c in spans.get(cid, []):
            anom[a:b + 1] = True
        nominal = ~anom
        warm = int(det.warmup_steps)
        nominal[:warm] = False
        per[cid] = dict(lad=ladders(e_s, det.config), steps=len(e_s),
                        nominal=nominal, warm=warm)
        SR.D.clear_caches()
        print(f"    [{i}/{len(chans)}] {cid}  {len(per[cid]['lad'])} candidate seqs  "
              f"{time.time()-t1:.1f}s")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    wall = time.time() - t0

    # ---- A5 + A1: the recall-vs-alarm-rate curve over p ----------------------
    curve = []
    for p in P_GRID:
        alarms = nom_steps = 0
        caught = []
        masks = {}
        for cid, d in per.items():
            m = alarm_mask(d["lad"], d["steps"], p)
            masks[cid] = m
            alarms += int((m & d["nominal"]).sum())
            nom_steps += int(d["nominal"].sum())
        for cid, lo, hi in target:
            if cid in masks and masks[cid][lo:hi + 1].any():
                caught.append(f"{cid}[{lo}]")
        curve.append(dict(p=p, nominal_rate=alarms / nom_steps if nom_steps else float("nan"),
                          caught=len(caught), of=len(target), events=caught))

    # A1: the smallest p that retains each target event
    per_event = []
    for cid, lo, hi in target:
        d = per.get(cid)
        if d is None:
            per_event.append(dict(event=f"{cid}[{lo}]", channel=cid, start=lo,
                                  retained_upto_p=None, note="channel not scored"))
            continue
        touching = [s for s in d["lad"] if not (s["hi"] <= lo or s["lo"] > hi)]
        best = max((s["survives_upto_p"] for s in touching), default=None)
        per_event.append(dict(event=f"{cid}[{lo}]", channel=cid, start=lo,
                              retained_upto_p=best, n_candidate_seqs=len(touching),
                              emits_at=min((s["emits_at"] for s in touching), default=None)))

    # ---- A6: lead time in timesteps, at the frozen p -------------------------
    lead = []
    for cid, lo, hi in target:
        d = per.get(cid)
        if d is None:
            continue
        touch = [s for s in d["lad"]
                 if s["survives_upto_p"] >= telemanom.PRUNING_P
                 and not (s["hi"] <= lo or s["lo"] > hi) and s["emitted"]]
        if touch:
            lead.append(dict(event=f"{cid}[{lo}]", onset=lo,
                             emits_at=min(s["emits_at"] for s in touch),
                             lead_steps=lo - min(s["emits_at"] for s in touch)))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = dict(generated_utc=stamp, diagnosis_only=True, pre_registration="docs/MODELS.md 37.7",
               frozen_arm=dict(multiplier=mult, pooled_nominal_rate=base_rate,
                               source=Path(args.stage4).name),
               channels_scored=len(per), channels_skipped=skipped,
               wall_clock_s=round(wall, 1), p_grid=P_GRID,
               curve=curve, per_event=per_event, lead_time=lead, extrema=extrema,
               weight_store=dict(before=before, after=after),
               operations=budget.as_dict())
    path = ROOT / "runs" / "smap-msl" / "_forensics" / f"{stamp}-{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n  artifact {path.relative_to(ROOT)}")
    print(f"  {budget.report()}")
    try:
        ops.commit(client, cfg_r2.bucket, ledger, budget)
    except Exception as exc:                     # artifact before ledger, always
        print(f"  (!) LEDGER COMMIT FAILED -- the artifact is written: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
