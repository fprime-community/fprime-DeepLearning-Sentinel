"""D86: mirror SMAP/MSL and ESA-ADB m1-g8.9.10 from R2 onto files, once, on the Mac.

    PYTHONPATH=src .venv/bin/python scripts/d86_mirror.py            # project only
    PYTHONPATH=src .venv/bin/python scripts/d86_mirror.py --fetch    # project, then fetch
    PYTHONPATH=src .venv/bin/python scripts/d86_mirror.py --reconcile runs/_mirror/R2_OPS_*.json

Owner correction (2026-09-26): the remote machine fetches directly, with a separate
READ-ONLY token that cannot write the ledger. So this script never writes the ledger
itself: each run writes its own R2_OPS_<utc>.json (every operation it counted, by class
and by name -- one file per run, so no run's count can overwrite another's), and
`--reconcile` on the Mac, with the owner's read-write keys, adds those counts to the
month-keyed ledger. The counts are the budget's own, so nothing is estimated.

Every later D86 run reads these files through `sentinel_eval.read.DirSource` and
never reaches the bucket. The projection is
printed before any object is fetched, from the two manifests alone, and the run
stops if it is over the tripwire (docs/DATA.md 150; stop 66). Every object is
checked against its manifest sha256 on arrival. The ESA bundle is then loaded
twice through the real loading path: once with R2 as the fallback, so a key the
projection missed is still mirrored and counted, and once with no fallback,
which proves the mirror is complete. MIRROR_SHA256.txt is written before the
operation counts, as every run here does (the artifact before the ledger).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_data import config as C                                # noqa: E402
from sentinel_data import r2                                         # noqa: E402
from sentinel_eval import bundle as bundle_mod                       # noqa: E402
from sentinel_eval import ops, read, tasks                           # noqa: E402
from sentinel_eval.catalog import Catalog                            # noqa: E402
from sentinel_eval.labels import LabelSet                            # noqa: E402

MIRROR = ROOT / "runs" / "_mirror"
SMAP_MANIFEST = "_manifest/smap_msl.json"                # scripts/smap_rungs.py:67
ESA_TASK = "m1-g8.9.10"
TRIPWIRE = C.OPS_TRIPWIRE                                # 1,000


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


STARTED = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_ops(budget, stage: str) -> None:
    """This run's own counts, for `--reconcile` on the Mac. One file per run."""
    MIRROR.mkdir(parents=True, exist_ok=True)
    (MIRROR / f"R2_OPS_{STARTED}.json").write_text(json.dumps({
        "producer": "scripts/d86_mirror.py", "stage": stage, "month": r2.current_month(),
        "class_a": budget.class_a, "class_b": budget.class_b,
        "by_operation": dict(budget.by_operation)}, indent=2) + "\n")


def reconcile(paths: list[Path]) -> int:
    """Add read-only runs' counts to the ledger, plus this read and write. Mac only."""
    runs = [json.loads(p.read_text()) for p in paths]
    for p, counted in zip(paths, runs):
        if counted["month"] != r2.current_month():
            raise SystemExit(f"  (!) {p}: counts are for {counted['month']}, the ledger month "
                             f"is {r2.current_month()} -- add them by hand to the right month")
    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    a_ = sum(c["class_a"] for c in runs)
    b_ = sum(c["class_b"] for c in runs) + budget.class_b             # + this ledger read
    document = r2.roll_and_add(ledger.document, a_ + 1, b_)            # + this ledger write
    blob = json.dumps(document, indent=2).encode() + b"\n"
    import base64
    r2.put_bytes(client, cfg.bucket, C.LEDGER_KEY, blob,
                 base64.b64encode(hashlib.md5(blob).digest()).decode())
    month = document["months"][r2.current_month()]
    print(f"  reconciled {len(paths)} runs: +{a_ + 1} A, +{b_} B; month now "
          f"A {month['class_a']:,}  B {month['class_b']:,}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="fetch after projecting")
    ap.add_argument("--reconcile", type=Path, nargs="+", help="add R2_OPS_*.json to the ledger (Mac)")
    a = ap.parse_args(argv)
    if a.reconcile:
        return reconcile(a.reconcile)

    cfg = C.load_r2_config()
    client, budget = ops.connect(cfg)
    ledger = ops.load(client, cfg.bucket)
    budget.prior_class_a, budget.prior_class_b = ledger.month_class_a, ledger.month_class_b
    source = read.DirSource(MIRROR, fallback=read.R2Source(client, cfg.bucket))

    smap = json.loads(source.get(SMAP_MANIFEST))
    smap_objects = [o for ch in smap["channels"] for o in ch["objects"]]
    catalog = Catalog.load(source)
    task = tasks.get(ESA_TASK)
    channel_ids = task.selection.resolve(catalog)
    esa_objects = [o for cid in channel_ids for o in catalog.channel(task.mission, cid).objects]
    labels_obj = catalog.annotation("labels")

    spent_b = budget.class_b
    todo = ([(o["key"], o["sha256"]) for o in smap_objects]
            + [(smap["labels_key"], smap["labels_sha256"])]
            + [(o.key, o.sha256) for o in esa_objects]
            + [(labels_obj.key, labels_obj.sha256)])
    missing = [(k, s) for k, s in todo if not (MIRROR / k).is_file()]
    projected_b = spent_b + len(missing)
    projected_a = 0                                      # read-only: no write here
    print(f"  SMAP/MSL  {len(smap['channels'])} channels, {len(smap_objects)} arrays + labels")
    print(f"  ESA-ADB   {ESA_TASK}: {len(channel_ids)} channels {channel_ids}")
    print(f"            {len(esa_objects)} channel objects + labels")
    print(f"  already mirrored {len(todo) - len(missing)} of {len(todo)}")
    print(f"  PROJECTED  Class B {projected_b} (spent so far {spent_b}), Class A {projected_a}; "
          f"tripwire {TRIPWIRE}")
    print(f"  month so far  A {ledger.month_class_a:,}  B {ledger.month_class_b:,}")
    if projected_b + projected_a > TRIPWIRE:
        print("  (!) over the tripwire -- stopping before any object is fetched (stop 66)")
        write_ops(budget, "projection, stopped")
        return 2
    if not a.fetch:
        write_ops(budget, "projection")
        print(f"  projection only; {budget.class_b} B spent on manifests, in R2_OPS_{STARTED}.json")
        return 0

    for key, want in missing:
        blob = source.get(key)
        if sha256(blob) != want:
            (MIRROR / key).unlink()
            raise SystemExit(f"  (!) {key}: sha256 mismatch against its manifest")
    for key, want in todo:
        if sha256((MIRROR / key).read_bytes()) != want:
            raise SystemExit(f"  (!) {key}: mirrored bytes do not match the manifest")
    print(f"  {len(todo)} objects mirrored and verified against their manifests")

    # The real loading path, twice: fallback first (counts anything the projection
    # missed), then offline, which is how the remote machine will read it.
    labels = LabelSet.from_table(read.read_annotation(source, catalog, "labels"))
    bundle_mod.load(source, catalog, labels, mission=task.mission, channel_ids=channel_ids)
    offline = read.DirSource(MIRROR)
    labels = LabelSet.from_table(read.read_annotation(offline, Catalog.load(offline), "labels"))
    loaded = bundle_mod.load(offline, Catalog.load(offline), labels,
                             mission=task.mission, channel_ids=channel_ids)
    print(loaded.describe())
    print(f"  offline load complete: zero R2 operations after the fallback pass")

    lines = sorted(f"{sha256(p.read_bytes())}  {p.relative_to(MIRROR)}"
                   for p in MIRROR.rglob("*") if p.is_file() and p.name != "MIRROR_SHA256.txt"
                   and not p.name.startswith("R2_OPS_"))
    (MIRROR / "MIRROR_SHA256.txt").write_text("\n".join(lines) + "\n")
    print(f"  wrote MIRROR_SHA256.txt ({len(lines)} files)")
    write_ops(budget, "fetched")
    print(budget.report())
    print(f"  counts in R2_OPS_{STARTED}.json -- reconcile on the Mac with --reconcile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
