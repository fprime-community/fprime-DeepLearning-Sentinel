"""Work item 9.9 study 1 stage 1: is the labelled contextual class actually in range?

Pre-registered in `docs/MODELS.md` 26, V1 to V5, committed before this ran.

**Not a test of the cross-channel thesis.** Objective.md 9.2 stands. This asks one
question: of the 43 sequences SMAP/MSL labels `contextual`, how many stay strictly
inside their own channel's training min/max -- the honest denominator for anything
stage 2 could claim, and the direct answer to Wu & Keogh's triviality critique.

Manifest-addressed reads only, through `_manifest/smap_msl.json`. Never a LIST.

    PYTHONPATH=src .venv/bin/python scripts/smap_visibility.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_eval import ops                                          # noqa: E402

MANIFEST_KEY = "_manifest/smap_msl.json"


def fetch(client, bucket, key) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acknowledge-tripwire", action="store_true")
    args = ap.parse_args(argv)

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg, acknowledge_tripwire=args.acknowledge_tripwire)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b

    manifest = json.loads(fetch(client, cfg.bucket, MANIFEST_KEY))
    labels = list(csv.DictReader(io.StringIO(
        fetch(client, cfg.bucket, manifest["labels_key"]).decode())))
    print(f"  manifest: {manifest['channel_count']} channels; labels: {len(labels)} rows")
    for d in manifest.get("labelling_defects", []):
        print(f"  (!) labelling defect carried from ingest: {d}")

    by_id = {ch["channel_id"]: ch for ch in manifest["channels"]}
    rows, seen = [], set()
    for r in labels:
        cid = r["chan_id"]
        if cid in seen:                       # P-2: counted once, 26.1
            continue
        seen.add(cid)
        ch = by_id[cid]
        keys = {o["split"]: o["key"] for o in ch["objects"]}
        train = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["train"])))
        test = np.load(io.BytesIO(fetch(client, cfg.bucket, keys["test"])))
        # 26.1: column 0 is telemetry; the rest are one-hot commands.
        lo, hi = float(train[:, 0].min()), float(train[:, 0].max())
        classes = [x.strip() for x in r["class"].strip("[]").split(",") if x.strip()]
        for (a, b), cls in zip(ast.literal_eval(r["anomaly_sequences"]), classes):
            span = test[a:b + 1, 0]
            # strict: touching an extreme is not leaving it (24.9's rule)
            out = int(np.sum((span < lo) | (span > hi)))
            rows.append({"channel": cid, "spacecraft": r["spacecraft"], "class": cls,
                         "start": int(a), "end": int(b), "steps": int(span.size),
                         "train_min": lo, "train_max": hi,
                         "steps_outside": out, "in_range": bool(out == 0)})

    tot = Counter((x["class"], x["in_range"]) for x in rows)
    per_sc = Counter((x["spacecraft"], x["class"], x["in_range"]) for x in rows)
    n_ctx = sum(1 for x in rows if x["class"] == "contextual")
    n_pt = sum(1 for x in rows if x["class"] == "point")
    ctx_in, pt_in = tot[("contextual", True)], tot[("point", True)]
    print(f"\n  sequences: {len(rows)}   contextual {n_ctx}   point {n_pt}")
    print(f"  IN RANGE (no step of column 0 outside the training min/max):")
    print(f"    contextual  {ctx_in}/{n_ctx}  ({100*ctx_in/max(1,n_ctx):.1f}%)")
    print(f"    point       {pt_in}/{n_pt}  ({100*pt_in/max(1,n_pt):.1f}%)")
    gap = 100*ctx_in/max(1, n_ctx) - 100*pt_in/max(1, n_pt)
    print(f"    gap         {gap:+.1f} percentage points")
    for sc in ("SMAP", "MSL"):
        c_n = sum(1 for x in rows if x["spacecraft"] == sc and x["class"] == "contextual")
        p_n = sum(1 for x in rows if x["spacecraft"] == sc and x["class"] == "point")
        c_i, p_i = per_sc[(sc, "contextual", True)], per_sc[(sc, "point", True)]
        g = 100*c_i/max(1, c_n) - 100*p_i/max(1, p_n)
        print(f"    {sc:5s} contextual {c_i}/{c_n}   point {p_i}/{p_n}   gap {g:+.1f} pp")

    out = {"pre_registration": "docs/MODELS.md 26", "dataset": "smap-msl/v1",
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "definition": "column 0 only; strict; training min/max from the train split",
           "labelling_defects": manifest.get("labelling_defects", []),
           "totals": {"sequences": len(rows), "contextual": n_ctx, "point": n_pt,
                      "contextual_in_range": ctx_in, "point_in_range": pt_in},
           "sequences": rows}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = Path("runs") / "smap-msl" / "_forensics" / f"{stamp}-visibility.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  artifact: {path}")
    out["operations"] = budget.as_dict()
    path.write_text(json.dumps(out, indent=2))
    ops.commit(client, cfg.bucket, ledger, budget)
    print(f"  {budget.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
