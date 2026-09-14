"""How close to the rail do the 39 in-range contextual sequences actually sit?

`docs/DECISIONS.md` D46 says 39 of 43 labelled contextual anomalies stay inside
their channel's training min/max. `scripts/smap_visibility.py:74` makes that test
**inclusive** -- `(span < lo) | (span > hi)`, so a value *equal* to a rail is not
outside, on 24.9's rule that touching an extreme is not leaving it.

**That leaves a question the visibility artifact cannot answer**: of the 39, how
many sit *strictly* inside, and how many merely touch? The artifact records a count
of steps outside and the envelope, and no per-step margin. This measures it.

For every step of every in-range contextual sequence:

    margin(t) = min(x[t] - train_min, train_max - x[t])

and the sequence's margin is the minimum over its steps -- how close it ever came
to either rail. Zero means it touched.

**Cost.** The envelopes and the spans both come from the visibility artifact
already on disk, so no train array and no labels table is fetched:

    smoke   1 ledger + 1 manifest                    =  2 Class B
    run     1 ledger + 1 manifest + 26 test arrays   = 28 Class B
                                                     + 1 Class A, the ledger
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                      # noqa: E402
from sentinel_eval import ops                              # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VIS = ROOT / "runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json"
OUT = ROOT / "runs/smap-msl/_forensics"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args(argv)

    vis = json.loads(VIS.read_text())
    want = [s for s in vis["sequences"] if s["class"] == "contextual" and s["in_range"]]
    channels = sorted({s["channel"] for s in want})
    print(f"  {len(want)} in-range contextual sequences across {len(channels)} channels")

    SR = _load("smap_rungs", "scripts/smap_rungs.py")
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    print(f"  month so far: {ledger.month_class_a} Class A, {ledger.month_class_b} Class B")

    projected = 2 if a.smoke else 2 + len(channels)
    print(f"  projected {projected} Class B (1 ledger + 1 manifest"
          f"{'' if a.smoke else f' + {len(channels)} test arrays'})")
    man = json.loads(SR.fetch(client, cfg.bucket, SR.MANIFEST_KEY))
    by_id = {c["channel_id"]: c for c in man["channels"]}

    if a.smoke:
        n_obj = sum(len([o for o in by_id[c]["objects"] if o["split"] == "test"])
                    for c in channels)
        print(f"  actual {budget.class_b} Class B, projected {projected}: "
              f"{'MATCH' if budget.class_b == projected else '(!) DIVERGED'}")
        print(f"  the run will fetch {n_obj} test object(s) for {len(channels)} channels")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 0 if budget.class_b == projected and n_obj == len(channels) else 1

    tests = {}
    for cid in channels:
        keys = {o["split"]: o["key"] for o in by_id[cid]["objects"]}
        tests[cid] = np.load(io.BytesIO(SR.fetch(client, cfg.bucket, keys["test"])))[:, 0]
    print(f"  actual {budget.class_b} Class B, projected {projected}: "
          f"{'MATCH' if budget.class_b == projected else '(!) DIVERGED'}")
    if budget.class_b != projected:
        print("  STOPPING: the projection was wrong.")
        if not a.no_ledger:
            ops.commit(client, cfg.bucket, ledger, budget)
        return 1

    rows = []
    for s in want:
        x = tests[s["channel"]][s["start"]:s["end"] + 1].astype(np.float64)
        lo, hi = s["train_min"], s["train_max"]
        to_lo, to_hi = x - lo, hi - x
        margin = float(np.minimum(to_lo, to_hi).min())
        width = hi - lo
        rows.append({
            "channel": s["channel"], "spacecraft": s["spacecraft"],
            "start": s["start"], "end": s["end"], "steps": s["steps"],
            "train_min": lo, "train_max": hi, "envelope_width": width,
            "margin": margin,
            "relative_margin": (margin / width) if width > 0 else None,
            "nearer_rail": "min" if float(to_lo.min()) <= float(to_hi.min()) else "max",
            "steps_at_zero": int(np.count_nonzero(np.minimum(to_lo, to_hi) <= 0.0)),
        })

    rows.sort(key=lambda r: r["margin"])
    eps = np.finfo(np.float64).eps
    bands = [("exactly 0", 0.0), ("<= 1e-15", 1e-15), ("<= 1e-12", 1e-12),
             ("<= 1e-9", 1e-9), ("<= 1e-6", 1e-6), ("<= 1e-3", 1e-3)]
    print(f"\n  {'sequence':<18}{'margin':>14}{'rel':>12}  rail   envelope")
    for r in rows:
        rel = f"{r['relative_margin']:.3e}" if r["relative_margin"] is not None else "  -"
        print(f"  {r['channel']+'['+str(r['start'])+']':<18}{r['margin']:>14.3e}{rel:>12}"
              f"  {r['nearer_rail']:<5}  {r['envelope_width']:.4f}")
    print(f"\n  float64 eps = {eps:.3e}")
    for label, t in bands:
        n = sum(1 for r in rows if r["margin"] <= t)
        print(f"  margin {label:<10}: {n:>2} of {len(rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    art = OUT / f"{stamp}-margin.json"
    art.write_text(json.dumps({
        "question": "of the 39 in-range contextual sequences, how many touch a rail?",
        "definition": "margin = min over the span of min(x - train_min, train_max - x); "
                      "the in-range test at scripts/smap_visibility.py:74 is inclusive",
        "source_visibility": str(VIS.relative_to(ROOT)),
        "sequences": rows,
        "bands": {label: sum(1 for r in rows if r["margin"] <= t) for label, t in bands},
        "float64_eps": float(eps),
        "producer": "scripts/smap_margin.py",
        "operations": budget.as_dict(),
    }, indent=2) + "\n")
    print(f"\n  artifact {art.relative_to(ROOT)}")
    if not a.no_ledger:
        ops.commit(client, cfg.bucket, ledger, budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
