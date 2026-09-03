"""Ingest NASA SMAP/MSL (telemanom) into R2. Work item 9.9 study 1, stage 1.

**This is not a test of the cross-channel thesis.** Objective.md 9.2 stands: SMAP
and MSL are 81 unsynchronised univariate streams, so there are no genuine
cross-channel relationships in them to find. What they do carry, which ESA-ADB
does not, is a **labelled contextual population seven times larger** -- 43 of 105
sequences -- and per-channel **command inputs**, which is D6's open question.

Provenance and integrity, in that order:

  source     kaggle patrickfleith/nasa-anomaly-detection-dataset-smap-msl,
             the dataset the telemanom README itself now names
  labels     labeled_anomalies.csv, fetched from khundman/telemanom directly,
             never from the mirror, so the arrays are checked against the
             canonical file rather than against their own packaging
  gate       every channel's test array length must equal `num_values`, every
             anomaly span must lie inside it, both arrays must exist. Any
             mismatch aborts and uploads nothing

**A separate manifest object.** `src/sentinel_data/manifest.py` hard-codes the
single dataset key `esa-adb`, and `Catalog.load` reads it. Extending that module
would put every published ESA-ADB result at risk for no benefit, so this writes
`_manifest/smap_msl.json` as its own manifest and leaves `_manifest/manifest.json`
byte-identical. Additive only, which is the condition every entry in
`docs/HARNESS.md` 5a's register cites.

    PYTHONPATH=src .venv/bin/python scripts/ingest_smap_msl.py --stage <dir> [--go]
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import socket
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel_data import config as C                                  # noqa: E402
from sentinel_data import r2                                           # noqa: E402

LABELS_URL = ("https://raw.githubusercontent.com/khundman/telemanom/master/"
              "labeled_anomalies.csv")
DATASET = "smap-msl"
VERSION = "v1"
KAGGLE_REF = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
UPSTREAM = "https://github.com/khundman/telemanom"
LICENSE = "Apache-2.0 (telemanom); dataset redistributed on Kaggle"
MANIFEST_KEY = "_manifest/smap_msl.json"
PREFIX = f"{DATASET}/{VERSION}"


def canonical_labels() -> list[dict]:
    socket.setdefaulttimeout(30)
    raw = urllib.request.urlopen(LABELS_URL).read()
    return list(csv.DictReader(io.StringIO(raw.decode()))), hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> tuple[str, str, int]:
    blob = path.read_bytes()
    return (hashlib.sha256(blob).hexdigest(),
            base64.b64encode(hashlib.md5(blob).digest()).decode(),
            len(blob))


def gate(rows: list[dict], stage: Path) -> list[str]:
    """Every channel checked against the canonical labels. Any failure aborts."""
    fails = []
    for r in rows:
        cid, n = r["chan_id"], int(r["num_values"])
        te, tr = stage / "test" / f"{cid}.npy", stage / "train" / f"{cid}.npy"
        if not te.exists() or not tr.exists():
            fails.append(f"{cid}: array missing"); continue
        a = np.load(te, mmap_mode="r")
        if a.shape[0] != n:
            fails.append(f"{cid}: test length {a.shape[0]} != num_values {n}")
        for lo, hi in ast.literal_eval(r["anomaly_sequences"]):
            if not (0 <= lo <= hi < a.shape[0]):
                fails.append(f"{cid}: span [{lo},{hi}] outside length {a.shape[0]}")
    return fails


def plan_objects(rows: list[dict], stage: Path) -> tuple[list[dict], list[str]]:
    """One entry per unique channel. P-2 is duplicated in the labels, not on disk."""
    channels, seen, notes = [], set(), []
    for r in rows:
        cid = r["chan_id"]
        if cid in seen:
            notes.append(f"{cid}: duplicated label row -- spans "
                         f"{r['anomaly_sequences']} class {r['class']}")
            continue
        seen.add(cid)
        entry = {"channel_id": cid, "spacecraft": r["spacecraft"],
                 "num_values": int(r["num_values"]), "objects": []}
        for split in ("train", "test"):
            p = stage / split / f"{cid}.npy"
            sha, md5, size = digest(p)
            arr = np.load(p, mmap_mode="r")
            entry["objects"].append({
                "key": f"{PREFIX}/{split}/{cid}.npy", "split": split,
                "bytes": size, "sha256": sha, "md5_b64": md5,
                "rows": int(arr.shape[0]), "columns": int(arr.shape[1]),
                "local": str(p)})
        channels.append(entry)
    return channels, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, help="extracted data/data directory")
    ap.add_argument("--go", action="store_true", help="actually upload")
    args = ap.parse_args(argv)
    stage = Path(args.stage)

    rows, labels_sha = canonical_labels()
    print(f"  canonical labels: {len(rows)} rows, "
          f"{len({r['chan_id'] for r in rows})} unique channels, sha256 {labels_sha[:16]}...")

    fails = gate(rows, stage)
    if fails:
        print(f"  GATE FAILED: {len(fails)} mismatches. Uploading nothing.")
        for f in fails[:10]:
            print("   ", f)
        return 3
    print(f"  GATE PASSED: all {len(rows)} label rows agree with the arrays")

    channels, notes = plan_objects(rows, stage)
    objects = [o for ch in channels for o in ch["objects"]]
    total = sum(o["bytes"] for o in objects)
    class_a = len(objects) + 2          # arrays + labels + manifest
    print(f"  plan: {len(channels)} channels, {len(objects)} arrays, "
          f"{total/1e6:.1f} MB")
    print(f"  COST: {class_a} Class A PutObject, 0 Class B")
    for n in notes:
        print(f"  (!) labelling defect recorded: {n}")
    if class_a > 200:
        print("  ABORT: over the 200 Class A limit approved for this ingest.")
        return 4

    manifest = {
        "schema_version": 1, "dataset": DATASET, "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_kaggle": KAGGLE_REF, "upstream": UPSTREAM, "license": LICENSE,
        "labels_url": LABELS_URL, "labels_sha256": labels_sha,
        "labels_key": f"{PREFIX}/labeled_anomalies.csv",
        "provenance_note": (
            "Arrays from the Kaggle dataset the telemanom README names; labels "
            "fetched from khundman/telemanom directly and used to verify the "
            "arrays. Not a test of the cross-channel claim -- Objective.md 9.2."),
        "labelling_defects": notes,
        "channel_count": len(channels), "channels": channels,
    }
    if not args.go:
        print("  DRY RUN. Re-run with --go to upload.")
        return 0

    cfg = C.load_r2_config()
    client, counter = r2.make_client(cfg)
    for i, o in enumerate(objects, 1):
        r2.put(client, cfg.bucket, o["key"], Path(o["local"]), o["md5_b64"],
               {"sha256": o["sha256"]})
        if i % 40 == 0:
            print(f"    uploaded {i}/{len(objects)}")
    raw = urllib.request.urlopen(LABELS_URL).read()
    r2.put_bytes(client, cfg.bucket, manifest["labels_key"], raw,
                 base64.b64encode(hashlib.md5(raw).digest()).decode(),
                 {"sha256": labels_sha})
    for ch in manifest["channels"]:
        for o in ch["objects"]:
            o.pop("local", None)
    blob = json.dumps(manifest, indent=2, sort_keys=True).encode()
    r2.put_bytes(client, cfg.bucket, MANIFEST_KEY, blob,
                 base64.b64encode(hashlib.md5(blob).digest()).decode(),
                 {"sha256": hashlib.sha256(blob).hexdigest()})
    print(f"  manifest published: {MANIFEST_KEY}")
    print(f"  spent: {counter.class_a} Class A, {counter.class_b} Class B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
