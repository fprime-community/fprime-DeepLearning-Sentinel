"""Work item 9.9 study 1 stage 2. Pre-registered in `docs/MODELS.md` 26.7-26.12.

**Not a test of the cross-channel thesis.** Objective.md 9.2 stands: SMAP and MSL
are 81 unsynchronised univariate streams. This tests the **in-limits** claim -- can
a forecaster see anomalies that never leave their channel's historical range, which
a range check provably cannot -- and **D6**, open since work item 4.

Four arms, per channel, on the paper's own split (`train` fits and calibrates,
`test` is scored):

    gru-quantile+cmd   command columns as exogenous inputs
    gru-quantile       the same, commands withheld     <- the pair IS D6's ablation
    rstd               trailing standard deviation, window 120
    range              per-channel training min/max, widened by a swept multiplier

    PYTHONPATH=src .venv/bin/python scripts/smap_stage2.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import ops                                          # noqa: E402
from sentinel_eval.detector import Context, Detector, reduce_scores     # noqa: E402
from sentinel_models import detectors as D                             # noqa: E402
from sentinel_models import registry                                   # noqa: E402
from sentinel_models import telemanom                                  # noqa: E402

MANIFEST_KEY = "_manifest/smap_msl.json"
ARMS = ("gru+cmd", "gru", "rstd", "range")
#: 26.8's sweep. Multipliers on each arm's own calibrated threshold; 1.0 is its
#: natural operating point. For `range`, 1.0 is exactly the training min/max.
GRID = np.unique(np.concatenate([np.geomspace(0.02, 1.0, 60),
                                 np.geomspace(1.0, 50.0, 40), [1.0]]))


def fetch(client, bucket, key):
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def ctx_for(cid, n, commands):
    return Context(mission="smap-msl", channels=(cid,), groups=(0,),
                   period_seconds=1.0, fold=0, window=(0, n),
                   commands=commands, command_ids=())


def proportional_config(n: int):
    """D47: `error_window` scaled to the series, telemanom's own definition.

    The default is `ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT` = 2100, an absolute
    fixed against ESA-ADB where one fold is ~3.5M steps. The median SMAP/MSL
    training series is 2,690, so that warm-up (250 + 2100) consumes the whole
    channel: 16 of 81 test arrays sat entirely inside it and 48 of 81 had their
    threshold computed from warm-up scores. Proportional here, absolute on
    ESA-ADB, which is untouched -- 0.05 x 3.5M would be 175,000 and that is a
    different detector, not a correction. `docs/MODELS.md` 26.14, D47.
    """
    window = max(1, int(telemanom.SMOOTHING_PERC * n))
    return telemanom.Config(error_window=window,
                            stride=min(telemanom.ERROR_WINDOW_BATCH, max(1, window // 2)))


def forecaster(cid, train, test, use_commands):
    """Fit on `train`, score `test`. The recipe is unchanged: 99.9th percentile."""
    det = registry.build("gru-quantile")
    det.config = proportional_config(len(train))       # D47
    if use_commands:
        det.wants_commands = True          # instance flag; no registry entry added
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    c_tr = train[:, 1:].astype(np.float32) if use_commands else None
    c_te = test[:, 1:].astype(np.float32) if use_commands else None
    f_ctx, s_ctx = ctx_for(cid, len(v_tr), c_tr), ctx_for(cid, len(v_te), c_te)
    usable = np.ones(len(v_tr), dtype=bool)
    det.fit(v_tr, usable, f_ctx)
    tr_raw = det.score(v_tr, None, f_ctx)
    tr, _ = reduce_scores(tr_raw, len(v_tr))
    thr = float(Detector.threshold_from(det, tr))
    te_raw = det.score(v_te, None, s_ctx)
    te, _ = reduce_scores(te_raw, len(v_te))
    warm = int(getattr(det, "warmup_steps", 0))
    D.clear_caches()
    return np.asarray(te, dtype=np.float64), thr, warm


def floor(cid, train, test):
    det = registry.build("rstd")           # no forecast, so no warm-up to scale
    v_tr, v_te = train[:, :1].astype(np.float32), test[:, :1].astype(np.float32)
    f_ctx, s_ctx = ctx_for(cid, len(v_tr), None), ctx_for(cid, len(v_te), None)
    det.fit(v_tr, np.ones(len(v_tr), dtype=bool), f_ctx)
    tr, _ = reduce_scores(det.score(v_tr, None, f_ctx), len(v_tr))
    thr = float(det.threshold_from(tr))
    te, _ = reduce_scores(det.score(v_te, None, s_ctx), len(v_te))
    warm = int(getattr(det, "warmup_steps", 0))
    D.clear_caches()
    return np.asarray(te, dtype=np.float64), thr, warm


def range_check(train, test):
    """max |x - centre| / half over column 0. 1.0 IS the training min/max."""
    a = train[:, 0].astype(np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    centre, half = (lo + hi) / 2.0, max((hi - lo) / 2.0, np.finfo(np.float64).tiny)
    with np.errstate(invalid="ignore"):
        dev = np.abs(test[:, 0].astype(np.float64) - centre) / half
    return np.where(np.isfinite(dev), dev, -np.inf), 1.0, 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acknowledge-tripwire", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N channels")
    args = ap.parse_args(argv)

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    man = json.loads(fetch(client, cfg.bucket, MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        fetch(client, cfg.bucket, man["labels_key"]).decode())))

    seqs, seen = {}, set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen:                    # P-2 counted once, 26.8
            continue
        seen.add(cid)
        classes = [x.strip() for x in r["class"].strip("[]").split(",") if x.strip()]
        seqs[cid] = list(zip(ast.literal_eval(r["anomaly_sequences"]), classes))

    channels = man["channels"][: args.limit] if args.limit else man["channels"]
    store = D.WEIGHT_STORE
    before = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    per_channel, events, excluded = {}, [], {}
    print(f"  fitting {len(channels)} channels x 2 forecaster arms ...")

    for i, ch in enumerate(channels, 1):
        cid = ch["channel_id"]
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["train"])))
        test = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["test"])))
        # D17's first-epoch guard can refuse a channel outright, and it did.
        # A stall is recorded and the channel is excluded from the scored
        # population -- never dropped silently, and never half-dropped: the
        # ablation only means anything where BOTH arms fitted.
        arms, stalls = {}, {}
        for name, use_cmd in (("gru+cmd", True), ("gru", False)):
            try:
                arms[name] = forecaster(cid, train, test, use_cmd)
            except Exception as exc:
                stalls[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        if stalls:
            excluded[cid] = stalls
            print(f"    {cid}: EXCLUDED -- " + "; ".join(f"{k} {v}" for k, v in stalls.items()))
            continue
        arms["rstd"] = floor(cid, train, test)
        arms["range"] = range_check(train, test)

        n = len(test)
        warm = max(w for _, _, w in arms.values())
        scorable = np.zeros(n, dtype=bool)
        scorable[min(warm, n):] = True     # no history bridges train->test, 26.13
        anomaly = np.zeros(n, dtype=bool)
        for (a, b), _ in seqs.get(cid, []):
            anomaly[a:b + 1] = True
        per_channel[cid] = {"n": n, "warmup": warm,
                            "scorable": scorable, "nominal": scorable & ~anomaly,
                            "arms": {k: (v[0], v[1]) for k, v in arms.items()}}
        for (a, b), cls in seqs.get(cid, []):
            events.append({"channel": cid, "start": int(a), "end": int(b),
                           "class": cls, "steps": int(b - a + 1)})   # scored channels only
        if i % 10 == 0:
            print(f"    {i}/{len(channels)}")

    after = sum(1 for p in store.iterdir() if p.suffix == ".npz")
    print(f"  weight store {before} -> {after} (+{after-before}; 26.11 expects +162)")
    print(f"  channels scored {len(per_channel)}/{len(channels)}; "
          f"EXCLUDED {len(excluded)} on a training stall (D17's guard)")

    # -- pooled sweep: one multiplier per arm, matched on nominal-step rate ----
    nominal_total = int(sum(v["nominal"].sum() for v in per_channel.values()))

    by_channel = {}
    for k, e in enumerate(events):
        by_channel.setdefault(e["channel"], []).append((k, e["start"], e["end"]))

    def at(arm, m):
        flagged, caught = 0, set()
        for cid, v in per_channel.items():
            s, thr = v["arms"][arm]
            # `range` is strict: touching the historical extreme is not leaving
            # it, the rule 24.9 fixed after the strictness defect it caught.
            fired = (s > thr * m) if arm == "range" else (s >= thr * m)
            fired &= v["scorable"]
            flagged += int((fired & v["nominal"]).sum())
            for k, a, b in by_channel.get(cid, ()):
                if fired[a:b + 1].any():
                    caught.add(k)
        return flagged / max(1, nominal_total), caught

    base_rate, base_caught = at("gru+cmd", 1.0)
    print(f"  gru+cmd nominal-step rate at its own threshold: {base_rate*100:.4f}%")

    matched = {"gru+cmd": (1.0, base_rate, base_caught)}
    for arm in ("gru", "rstd", "range"):
        best = None
        for m in GRID:
            rate, caught = at(arm, m)
            if rate <= base_rate and (best is None or len(caught) > len(best[2])):
                best = (float(m), rate, caught)
        matched[arm] = best if best else (float(GRID[-1]), *at(arm, GRID[-1]))
    return report(matched, events, per_channel, base_rate, nominal_total,
                  client, cfg, ledger, budget, before, after, excluded)


def report(matched, events, per_channel, base_rate, nominal_total,
           client, cfg, ledger, budget, before, after, excluded) -> int:
    vis = json.loads(sorted(Path("runs/smap-msl/_forensics").glob("*visibility.json"))[-1]
                     .read_text())
    key = {(v["channel"], v["start"], v["end"]): v for v in vis["sequences"]}
    for e in events:
        v = key.get((e["channel"], e["start"], e["end"]))
        e["in_range"] = None if v is None else v["in_range"]
        e["steps_outside"] = None if v is None else v["steps_outside"]

    pops = {
        "in_range_contextual": [i for i, e in enumerate(events)
                                if e["class"] == "contextual" and e["in_range"]],
        "out_of_range_contextual": [i for i, e in enumerate(events)
                                    if e["class"] == "contextual" and e["in_range"] is False],
        "in_range_point": [i for i, e in enumerate(events)
                           if e["class"] == "point" and e["in_range"]],
        "all": list(range(len(events))),
    }
    print(f"\n  populations: " + "  ".join(f"{k} {len(v)}" for k, v in pops.items()))
    print(f"  matched on nominal-step rate = gru+cmd's {base_rate*100:.4f}% "
          f"over {nominal_total:,} nominal steps\n")
    hdr = f"  {'arm':9s} {'mult':>7s} {'nominal':>9s} " + "".join(
        f"{k:>26s}" for k in pops)
    print(hdr)
    rows = {}
    for arm in ARMS:
        m, rate, caught = matched[arm]
        cells, counts = "", {}
        for k, idx in pops.items():
            hit = len(caught & set(idx))
            counts[k] = {"k": hit, "n": len(idx)}
            cells += f"{f'{hit}/{len(idx)}':>26s}"
        rows[arm] = {"multiplier": m, "nominal_rate": rate, "counts": counts,
                     "caught": sorted(caught)}
        print(f"  {arm:9s} {m:7.3f} {rate*100:8.4f}% {cells}")

    w = rows["range"]["multiplier"]
    print(f"\n  V7: the range check's matched multiplier is w = {w:.3f} "
          f"({'BELOW 1.0 as predicted' if w < 1.0 else 'AT OR ABOVE 1.0 -- V7 REFUTED'})")
    if w >= 1.0:
        print("      -> it catches the in-range population 0 by construction; the")
        print("         comparison is UNINFORMATIVE, not a win. V8 is withdrawn (26.9).")

    out = {"pre_registration": "docs/MODELS.md 26.7-26.12", "dataset": "smap-msl/v1",
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "split": "the paper's own train/test; not comparable to ESA-ADB folds",
           "not_a_cross_channel_test": "Objective.md 9.2; 81 unsynchronised streams",
           "scorable_note": ("no history bridges train->test, so the first warmup "
                             "steps of each test array are not scorable"),
           "matched_on": "pooled nominal-step false-alarm rate = gru+cmd at its own threshold",
           "nominal_steps": nominal_total, "base_rate": base_rate,
           "weight_store": {"before": before, "after": after},
           "populations": {k: len(v) for k, v in pops.items()},
           "excluded_channels": excluded,
           "arms": rows, "events": events}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs/smap-msl/_forensics") / f"{stamp}-stage2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out["operations"] = budget.as_dict()
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  artifact: {path}")
    ops.commit(client, cfg.bucket, ledger, budget)
    print(f"  {budget.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
