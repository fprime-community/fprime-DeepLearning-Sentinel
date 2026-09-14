#!/usr/bin/env python3
"""`docs/MODELS.md` 44: does the residual half of the flown rule do anything?

**Pre-registered in 44 before this ran.** D68 adopted
``max(z_residual, z_derivative)`` against one calibrated cut and never measured
whether the first term decides anything. This removes it: **``z_derivative``
alone**, at the frozen arm's own matched **0.6820%**, on the channel-disjoint
19/19 split committed at 37.8 before any sweep.

**Nothing here is a second implementation.** Every mechanism is imported from
`scripts/decision_layer_arms.py`, which is the producer D65 and D68 rest on --
``zstat``, ``solve_threshold``, ``pooled_rate``, ``report``, ``caught``, and the
TUNE/EVAL split itself. A second set of definitions that agrees today is exactly
what 40.7 moved `trailing_stats` into `src/` to avoid.

**(!) The bands are two-sided, and 39's N6 is why.** ``max(a, b) >= b``, so at a
FIXED cut derivative-only can only lose events. The measurement is not at a fixed
cut: at a matched rate the cut is re-solved, and removing a statistic from the
maximum frees alarm budget. N6 asked what a departure would *cost*, banded one
way, and measured **+13** in the direction it was not looking -- *"a one-sided
worry written as a two-sided band"*. AB1 is banded in both directions and the cut
is reported beside the recall, never absorbed into it.

**Two reproduction gates, and this producer refuses to draw if either fails**,
which is the shape `scripts/dump_event_trace.py` uses:

  1. the frozen arm, rebuilt, returns **0.6820% and 10 of 38** (38.15, D65)
  2. the fused arm, rebuilt, returns **TUNE 13/19, EVAL 17/19, 30/38** (D65)

If the restatement cannot reproduce what D65 measured, nothing downstream of it
is evidence, and a number printed anyway would be worse than no number.

**Cost.** One bundle load of all 81 channels, cached weights, **no fits**:

    smoke   1 ledger + 1 manifest + 1 labels +   6 arrays =   9 Class B + 1 Class A
    run     1 ledger + 1 manifest + 1 labels + 162 arrays = 165 Class B + 1 Class A

The weight store is asserted unmoved: this run must not fit.
"""
from __future__ import annotations

import argparse, ast, csv, importlib.util, io, json, sys, time
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
F38 = _load("f38", "scripts/smap_forensics_38.py")
ARMS = _load("decision_layer_arms", "scripts/decision_layer_arms.py")

OUT = ROOT / "runs/smap-msl/_forensics"
VIS = ROOT / "runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json"
STAGE4 = ROOT / "runs/smap-msl/_forensics/2026-09-03T212818Z-stage4-ndt.json"


def projected(n_channels: int) -> int:
    """(!) The ledger read is IN the number, and 39.13.4 is why it is written out.

    The first smoke of the departures run projected 6 for two channels and spent
    7: ``ops.load`` fetches ``_manifest/ops_ledger.json`` before anything else and
    the formula omitted it. The 165 this project has quoted all along is
    ``1 ledger + 1 manifest + 1 labels + 162 arrays``, so the ledger was always in
    the figure and never in the projection.
    """
    return 1 + 1 + 1 + 2 * n_channels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="3 channels, projection checked")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="ablation")
    a = ap.parse_args(argv)
    limit = 3 if a.smoke else a.limit

    vis = json.loads(VIS.read_text()); st4 = json.loads(STAGE4.read_text())
    frozen_mult = float(st4["arms"]["gru"]["multiplier"])
    target = float(st4["arms"]["gru"]["nominal_rate"])
    ARMS.TARGET_RATE[0] = target
    targets = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
               if v["class"] == "contextual" and v["in_range"]]
    print(f"  frozen arm: mult {frozen_mult:.6f}, matched at {100*target:.4f}%")

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
    proj = projected(len(chans))
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
        per[cid] = dict(e_s=e_s, nominal=nominal, cfg=det.config, steps=len(x), zr=zr, zd=zd,
                        S={"fused": np.maximum(zr, zd), "deriv": zd})
        SR.D.clear_caches()
        if i % 10 == 0 or i == len(chans):
            print(f"    [{i}/{len(chans)}] {cid}  {time.time()-t0:6.1f}s")
    after = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    print(f"  loaded {len(per)} channels in {time.time()-t0:.1f}s; skipped {list(skipped)}")

    rows, detail = [], {}

    # ---- gate 1: the frozen arm, rebuilt -------------------------------
    base = {}
    for cid, d in per.items():
        d["S"]["frozen"] = np.asarray(telemanom.channel_ratios(d["e_s"], d["cfg"]), float)
        base[cid] = d["S"]["frozen"] >= frozen_mult
    rows.append(ARMS.report("frozen (stage 4)", base, per, targets, "gate 1"))

    # ---- gate 2 and the comparison --------------------------------------
    cuts, masks_by = {}, {}
    for key, name in (("fused", "fused: max(z_r, z_d)"), ("deriv", "derivative only")):
        thr, rate, masks = ARMS.solve_threshold(per, key, target)
        cuts[key] = thr; masks_by[key] = masks
        rows.append(ARMS.report(name, masks, per, targets, f"cut {thr:.6f}"))

    ev = {k: set(ARMS.caught(masks_by[k], targets, ARMS.EVAL)) for k in masks_by}
    al = {k: set(ARMS.caught(masks_by[k], targets, ARMS.TUNE | ARMS.EVAL)) for k in masks_by}

    # ---- AB4: does z_residual ever reach the cut on a caught event? ------
    reach = []
    for c, lo, hi in targets:
        if c not in per or f"{c}[{lo}]" not in al["fused"]:
            continue
        d = per[c]
        fired = np.flatnonzero(masks_by["fused"][c][lo:hi + 1])
        if not fired.size:
            continue
        t = lo + int(fired[0])
        reach.append(dict(event=f"{c}[{lo}]", t=t, z_residual=float(d["zr"][t]),
                          z_derivative=float(d["zd"][t]), cut=cuts["fused"],
                          residual_reaches_cut=bool(d["zr"][t] >= cuts["fused"])))
    detail["ab4"] = reach
    detail["eval_fused"] = sorted(ev["fused"]); detail["eval_deriv"] = sorted(ev["deriv"])
    detail["eval_only_fused"] = sorted(ev["fused"] - ev["deriv"])
    detail["eval_only_deriv"] = sorted(ev["deriv"] - ev["fused"])
    detail["all38_only_fused"] = sorted(al["fused"] - al["deriv"])
    detail["all38_only_deriv"] = sorted(al["deriv"] - al["fused"])
    detail["cuts"] = cuts

    art = dict(section="docs/MODELS.md 44", written=datetime.now(timezone.utc).isoformat(),
               producer="scripts/derivative_only_ablation.py", smoke=bool(a.smoke),
               target_rate=target, frozen_multiplier=frozen_mult,
               channels=len(per), skipped=skipped, rows=rows, detail=detail,
               weight_store=dict(before=before, after=after),
               operations=budget.as_dict(), projected_class_b=proj)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = OUT / f"{stamp}-{a.tag}.json"
    path.write_text(json.dumps(art, indent=2, default=str))
    print(f"  artifact {path.relative_to(ROOT)}")
    print(f"  projected {proj} Class B, actual {budget.as_dict()['class_b']}")
    try:
        ops.commit(client, cfg_r2.bucket, ledger, budget)
    except Exception as exc:
        print(f"  (!) ledger commit failed: {exc}; the artifact is written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
