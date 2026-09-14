#!/usr/bin/env python3
"""`docs/MODELS.md` 45: what is the flown rule's lead against the labelled onset?

**Pre-registered at 45 before this ran.** Every lead-time figure this project holds
was measured on the **frozen** arm, `telemanom.channel_ratios`, whose
``emits_at = seg_hi - 1``. **D68's rule crosses per tick** -- `Detector.cpp:199-210`
compares the fused score to the cut every tick -- so the flown rule's lead has never
been measured.

**The quantity, stated once and fixed in advance.** For each in-range contextual
event, the **first tick** at which ``max(z_residual, z_derivative) >= cut``, scanned
from ``onset - SCAN`` to the event's end, at the matched **0.6820%** on 37.8's
channel-disjoint split. Lead is ``onset - t``: **positive means early**.

**(!) `SCAN` IS 200 AND DOES NOT MOVE.** 45's LD3 makes a scan window chosen after
seeing the answer a stop condition, so it is a module constant here rather than a
flag, and changing it is a commit somebody makes on purpose.

**(!) This is a bounded lead and the bound is reported with it.** An alarm that
begins well before the onset for reasons of its own would show as a large positive
lead, which is the weakness `sentinel_eval.metrics.leadtime`'s own docstring names --
*"a very wide alarm range that happens to overlap inflates lead"*. The 200-step window
caps it, and the distance from the window's edge is reported per event so a reader can
see when a figure sits against the cap.

**Timesteps only, never wall clock.** SMAP/MSL's arrays carry no clock and its
timestamps are anonymised (`Objective.md` 1.1). **No early-warning claim follows from
any outcome here**: lead is measured against a **labelled onset**, a hindsight
annotation written after the fact, not a limit trip. That measurement is work item
11's, on a real clock, and stays there.

**Two reproduction gates, and this producer refuses to draw if either fails**, the
shape `scripts/dump_event_trace.py` and `scripts/derivative_only_ablation.py` use:

  1. the frozen arm, rebuilt, returns **0.6820% and 10 of 38**
  2. the fused arm, rebuilt, returns **TUNE 13/19, EVAL 17/19, 30/38** (D65)

**Cost.** One bundle load of all 81 channels, cached weights, **no fits**:

    smoke   1 ledger + 1 manifest + 1 labels +   6 arrays =   9 Class B + 1 Class A
    run     1 ledger + 1 manifest + 1 labels + 162 arrays = 165 Class B + 1 Class A
"""
from __future__ import annotations

import argparse, ast, csv, importlib.util, io, json, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Degrees of freedom")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sentinel_data import config as C                                   # noqa: E402
from sentinel_eval import ops                                           # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


SR = _load("smap_rungs", "scripts/smap_rungs.py")
ARMS = _load("decision_layer_arms", "scripts/decision_layer_arms.py")

OUT = ROOT / "runs/smap-msl/_forensics"
VIS = OUT / "2026-09-03T192723Z-visibility.json"
STAGE4 = OUT / "2026-09-03T212818Z-stage4-ndt.json"

#: (!) FIXED BY 45's LD3. Not a flag, not tunable, and moving it is a commit.
SCAN = 200

#: 37.7a's committed lead table, for LD1. The frozen arm's honest emission point
#: `emits_at = seg_hi - 1`, quoted rather than recomputed -- it is a record.
FROZEN_LEADS = {"T-8[870]": -17, "F-8[1950]": -33, "D-16[600]": -47, "E-11[5614]": -55,
                "G-7[3650]": -59, "E-10[5601]": -68, "G-7[7560]": -69, "T-13[1900]": -79,
                "M-3[1250]": -180, "A-8[4569]": -3620}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="lead")
    a = ap.parse_args(argv)
    limit = 3 if a.smoke else 0

    vis = json.loads(VIS.read_text()); st4 = json.loads(STAGE4.read_text())
    frozen_mult = float(st4["arms"]["gru"]["multiplier"])
    target = float(st4["arms"]["gru"]["nominal_rate"])
    ARMS.TARGET_RATE[0] = target
    targets = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
               if v["class"] == "contextual" and v["in_range"]]
    print(f"  frozen arm: mult {frozen_mult:.6f}, matched at {100*target:.4f}%   SCAN = {SCAN}")

    cfg_r2 = C.load_r2_config(); client, budget = ops.connect(cfg_r2)
    ledger = ops.load(client, cfg_r2.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(SR.fetch(client, cfg_r2.bucket, SR.MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        SR.fetch(client, cfg_r2.bucket, man["labels_key"]).decode())))
    spans, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen: continue
        seen.add(cid)
        spans[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]),
                              [x.strip() for x in r["class"].strip("[]").split(",")]))

    chans = list(man["channels"])
    if limit: chans = chans[:limit]
    proj = 1 + 1 + 1 + 2 * len(chans)
    print(f"  projected {proj} Class B for {len(chans)} channels")

    store = SR.D.WEIGHT_STORE
    before = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    per, skipped = {}, {}
    t0 = time.time()
    for i, ch in enumerate(chans, 1):
        cid = ch["channel_id"]; keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["train"])))
        test = np.load(io.BytesIO(SR.fetch(client, cfg_r2.bucket, keys["test"])))
        try:
            det = SR.registry.build("gru-telemanom"); det.config = SR.proportional_config(len(test))
            v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
            det.fit(v_tr, np.ones(len(v_tr), dtype=bool), SR.ctx_for(cid, len(v_tr)))
            e_s = np.asarray(det._smoothed_errors(v_te, SR.ctx_for(cid, len(v_te))),
                             dtype=np.float64)[:, 0]
        except Exception as exc:
            skipped[cid] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(f"    [{i}/{len(chans)}] {cid}: SKIP -- {skipped[cid][:60]}"); continue
        x = v_te[:, 0].astype(np.float64)
        anom = np.zeros(len(x), dtype=bool)
        for (lo_, hi_), _c in spans.get(cid, []): anom[lo_:hi_ + 1] = True
        nominal = ~anom; nominal[: int(det.warmup_steps)] = False
        span = det.config.error_window
        zr = ARMS.zstat(e_s, span)
        zd = ARMS.zstat(np.abs(np.diff(x, prepend=x[0])), span)
        per[cid] = dict(e_s=e_s, nominal=nominal, cfg=det.config, steps=len(x),
                        S={"fused": np.maximum(zr, zd)})
        SR.D.clear_caches()
        if i % 10 == 0 or i == len(chans):
            print(f"    [{i}/{len(chans)}] {cid}  {time.time()-t0:6.1f}s")
    after = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    print(f"  loaded {len(per)} channels in {time.time()-t0:.1f}s; skipped {list(skipped)}")

    rows = []
    base = {}
    for cid, d in per.items():
        d["S"]["frozen"] = np.asarray(telemanom.channel_ratios(d["e_s"], d["cfg"]), float)
        base[cid] = d["S"]["frozen"] >= frozen_mult
    rows.append(ARMS.report("frozen (stage 4)", base, per, targets, "gate 1"))
    cut, rate, masks = ARMS.solve_threshold(per, "fused", target)
    rows.append(ARMS.report("fused: max(z_r, z_d)", masks, per, targets, f"gate 2, cut {cut:.6f}"))

    # -- the measurement: first crossing from onset - SCAN ---------------------
    leads = []
    for c, lo, hi in targets:
        if c not in per or not masks[c][lo:hi + 1].any():
            continue                                   # not caught: no lead to report
        start = max(0, lo - SCAN)
        fired = np.flatnonzero(masks[c][start:hi + 1])
        t = start + int(fired[0])
        leads.append(dict(event=f"{c}[{lo}]", onset=lo, first_crossing=t, lead=lo - t,
                          at_window_edge=bool(t == start),
                          half=("TUNE" if c in ARMS.TUNE else "EVAL" if c in ARMS.EVAL else "-")))
    vals = [r["lead"] for r in leads]
    pos = [r for r in leads if r["lead"] > 0]
    edge = [r for r in leads if r["at_window_edge"]]
    both = [r for r in leads if r["event"] in FROZEN_LEADS]

    print()
    print(f"  FLOWN RULE, lead = onset - first crossing, scanned from onset - {SCAN}")
    print(f"    n = {len(leads)} caught events    median {statistics.median(vals):+.1f} timesteps")
    print(f"    positive (early)     {len(pos)}/{len(leads)}")
    print(f"    at the window edge   {len(edge)}/{len(leads)}  (lead pinned at +{SCAN}, reported not trusted)")
    if both:
        fz = [FROZEN_LEADS[r['event']] for r in both]
        print(f"    on the {len(both)} events 37.7a also caught: flown median "
              f"{statistics.median([r['lead'] for r in both]):+.1f} against frozen "
              f"{statistics.median(fz):+.1f}")
    print()
    for r in sorted(leads, key=lambda r: -r["lead"])[:12]:
        mark = "  <- at window edge" if r["at_window_edge"] else ""
        print(f"    {r['event']:<14} {r['half']:<5} onset {r['onset']:<6} first {r['first_crossing']:<6} "
              f"lead {r['lead']:+6}{mark}")

    art = dict(section="docs/MODELS.md 45", written=datetime.now(timezone.utc).isoformat(),
               producer="scripts/flown_rule_lead.py", smoke=bool(a.smoke), scan=SCAN,
               target_rate=target, cut=cut, realised_rate=rate, rows=rows, leads=leads,
               frozen_leads_37_7a=FROZEN_LEADS, channels=len(per), skipped=skipped,
               weight_store=dict(before=before, after=after),
               operations=budget.as_dict(), projected_class_b=proj)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = OUT / f"{stamp}-{a.tag}.json"
    path.write_text(json.dumps(art, indent=2, default=str))
    print(f"\n  artifact {path.relative_to(ROOT)}")
    print(f"  projected {proj} Class B, actual {budget.as_dict()['class_b']}")
    try:
        ops.commit(client, cfg_r2.bucket, ledger, budget)
    except Exception as exc:
        print(f"  (!) ledger commit failed: {exc}; the artifact is written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
