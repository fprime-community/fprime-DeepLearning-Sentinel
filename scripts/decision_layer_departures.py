"""N5 and N6 of `docs/MODELS.md` 39: what the port's two departures cost.

**Pre-registered in `docs/MODELS.md` 39 before this ran.** Both departures were
chosen for reasons that do not depend on their price, and 39 registered that the
price would be measured rather than assumed:

  N5   THE GUARD-CELL CHOICE. telemanom's window includes the 70 steps it judges,
       so a segment is decided at its end and a flight detector emitting per tick
       inherits up to 69 timesteps of latency. Guard cells remove that -- and
       `docs/MODELS.md` 10.7 measured them costing recall 38/46 -> 34/46 with
       alarm ranges +27% on `m1-g8.9.10`, while *helping* on `m1-ss5`, with the
       sign reversing for reasons not established. **Neither is adopted.** The
       band: the two agree within 1 event HOLD, 2-3 NO VERDICT, 4 or more FAIL.

  N6   THE PRICE OF DROPPING BACKWARD DILATION. `telemanom._buffered` widens every
       exceedance by 99 steps on both sides; the backward half marks timesteps
       already emitted and a warn-only component cannot go back and speak. The
       flight rule drops it. The band: 0-1 event HOLD, 2-3 NO VERDICT, 4 or more
       FAIL, in which case 39.5's choice is re-opened rather than defended.

**Both ride one read.** They are two decision-layer variations over the same
smoothed errors, so the 165 Class B that loads the arrays answers both. Stated
because it is the reason this costs one read and not two.

Every arm is brought to the frozen arm's own pooled nominal rate -- **0.6820%**,
not 0.6838%, which is the commanded arm's (26.18) -- on the channel-disjoint
19/19 split committed at 37.8 before any sweep.

The forward-only variant comes from `src/sentinel_models/flight_reference.ratios`,
which reproduces `telemanom.channel_ratios` byte-for-byte when the departure is
switched off (`tests/test_flight_reference.py`).

Smoke first: `--limit 2` reads 2 channels and costs 1 manifest + 1 labels +
4 arrays = 6 Class B. State projected against actual before spending the rest.
"""
from __future__ import annotations

import argparse, ast, csv, importlib.util, io, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                   # noqa: E402
from sentinel_eval import ops                                           # noqa: E402
from sentinel_models import telemanom                                   # noqa: E402
from sentinel_models.flight_reference import ratios                     # noqa: E402


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


SR = _load("smap_rungs", "scripts/smap_rungs.py")
ARMS = _load("decision_layer_arms", "scripts/decision_layer_arms.py")

TUNE, EVAL = ARMS.TUNE, ARMS.EVAL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visibility", required=True)
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="departures")
    a = ap.parse_args(argv)

    vis = json.loads(Path(a.visibility).read_text())
    st4 = json.loads(Path(a.stage4).read_text())
    frozen_mult = float(st4["arms"]["gru"]["multiplier"])
    target = float(st4["arms"]["gru"]["nominal_rate"])
    ARMS.TARGET_RATE[0] = target
    print(f"  frozen arm: mult {frozen_mult:.6f}, matched at {100*target:.4f}% "
          f"(0.6838% is the COMMANDED arm's, 26.18)")
    targets = [(v["channel"], v["start"], v["end"]) for v in vis["sequences"]
               if v["class"] == "contextual" and v["in_range"]]

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
    if a.limit: chans = chans[: a.limit]
    # (!) THE LEDGER READ COUNTS, AND THE FIRST SMOKE CAUGHT ME OMITTING IT.
    # `ops.load` fetches `_manifest/ops_ledger.json` before anything else, which
    # is a GetObject like any other. Projected 6 against an actual 7 on the
    # 2-channel smoke; the run was right and the arithmetic was wrong. The full
    # 81-channel figure this project has always quoted -- **165** -- is
    # 1 + 1 + 1 + 162, so the ledger was always in it and never in the formula.
    projected = 3 + 2 * len(chans)
    print(f"  projected cost: 1 ledger + 1 manifest + 1 labels + {2*len(chans)} "
          f"arrays = {projected} Class B, plus 1 Class A to commit the ledger")

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
        for (lo_, hi_), _c in spans.get(cid, []): anom[lo_:hi_+1] = True
        nominal = ~anom; nominal[: int(det.warmup_steps)] = False
        per[cid] = dict(e_s=e_s, nominal=nominal, cfg=det.config, steps=len(x), S={})
        SR.D.clear_caches()
        if i % 10 == 0 or i == len(chans):
            print(f"    [{i}/{len(chans)}] {cid}  {time.time()-t0:6.1f}s")
    after = sum(1 for q in store.iterdir() if q.suffix == ".npz")
    assert after == before, f"weight store moved {before} -> {after}; this run must not fit"
    load_wall = time.time() - t0
    print(f"  loaded {len(per)} channels in {load_wall:.1f}s; skipped {list(skipped)}")

    rows, detail = [], {}

    # ---- the reproduction gate -------------------------------------------
    base = {}
    for cid, d in per.items():
        d["S"]["frozen"] = np.asarray(telemanom.channel_ratios(d["e_s"], d["cfg"]), float)
        base[cid] = d["S"]["frozen"] >= frozen_mult
    rows.append(ARMS.report("frozen (stage 4)", base, per, targets, "reproduction gate"))
    detail["frozen_eval"] = ARMS.caught(base, targets, EVAL)

    # (!) The gate that makes the rest meaningful: `flight_reference.ratios` with
    # the departure switched off must BE `channel_ratios`, not merely agree with it.
    identical = all(
        np.array_equal(d["S"]["frozen"],
                       np.asarray(ratios(d["e_s"], d["cfg"], forward_only=False), float))
        for d in per.values())
    print(f"  reference gate: forward_only=False is channel_ratios exactly -- {identical}")
    detail["reference_gate"] = bool(identical)
    if not identical:
        print("  (!) STOP: the reference restatement is not the reference. "
              "Nothing below is reported.")
        return 1

    # ---- N6: forward-only dilation ---------------------------------------
    for cid, d in per.items():
        d["S"]["fwd"] = np.asarray(ratios(d["e_s"], d["cfg"], forward_only=True), float)
    thr, rate, masks = ARMS.solve_threshold(per, "fwd", target, floor=1.0)
    rows.append(ARMS.report("N6: forward-only dilation", masks, per, targets,
                            f"cut {thr:.4f}"))
    detail["n6_eval"] = ARMS.caught(masks, targets, EVAL)

    # ---- N5: guard cells --------------------------------------------------
    for cid, d in per.items():
        g = telemanom.Config(error_window=d["cfg"].error_window, stride=d["cfg"].stride,
                             guard_segment=True)
        d["S"]["guard"] = np.asarray(telemanom.channel_ratios(d["e_s"], g), float)
    thr_g, rate_g, masks_g = ARMS.solve_threshold(per, "guard", target, floor=1.0)
    rows.append(ARMS.report("N5: guard cells", masks_g, per, targets, f"cut {thr_g:.4f}"))
    detail["n5_eval"] = ARMS.caught(masks_g, targets, EVAL)

    # ---- the verdicts, against the bands 39 registered --------------------
    def band(delta, hold, noverdict):
        if abs(delta) <= hold: return "HOLD"
        return "NO VERDICT" if abs(delta) <= noverdict else "FAIL"

    frozen_all = set(ARMS.caught(base, targets, TUNE | EVAL))
    n6_all = set(ARMS.caught(masks, targets, TUNE | EVAL))
    n5_all = set(ARMS.caught(masks_g, targets, TUNE | EVAL))
    d6, d5 = len(n6_all) - len(frozen_all), len(n5_all) - len(frozen_all)
    detail["n6_delta"], detail["n5_delta"] = d6, d5
    detail["n6_verdict"] = band(d6, 1, 3)
    detail["n5_verdict"] = band(d5, 1, 3)
    print(f"\n  N6  forward-only vs reference dilation:  {len(frozen_all)}/38 -> "
          f"{len(n6_all)}/38   delta {d6:+d}   **{detail['n6_verdict']}**")
    print(f"  N5  guard cells vs segment-end:          {len(frozen_all)}/38 -> "
          f"{len(n5_all)}/38   delta {d5:+d}   **{detail['n5_verdict']}**")
    print(f"      events only the reference catches: "
          f"{sorted(frozen_all - n6_all)} (N6), {sorted(frozen_all - n5_all)} (N5)")
    print(f"      events only the variant catches:   "
          f"{sorted(n6_all - frozen_all)} (N6), {sorted(n5_all - frozen_all)} (N5)")
    print("  NEITHER IS ADOPTED HERE. 39.4 and 39.5 stand or are re-opened on these.")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = dict(generated_utc=stamp, pre_registration="docs/MODELS.md 39, N5 and N6",
               matched_rate=target, frozen_multiplier=frozen_mult,
               tune=sorted(TUNE), eval=sorted(EVAL), rows=rows, detail=detail,
               channels=len(per), skipped=skipped, load_wall_s=round(load_wall, 1),
               weight_store=dict(before=before, after=after),
               projected_class_b=projected, operations=budget.as_dict())
    path = ROOT / "runs" / "smap-msl" / "_forensics" / f"{stamp}-{a.tag}.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\n  artifact {path.relative_to(ROOT)}\n{budget.report()}")
    try: ops.commit(client, cfg_r2.bucket, ledger, budget)
    except Exception as exc: print(f"  (!) LEDGER COMMIT FAILED -- artifact is written: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
