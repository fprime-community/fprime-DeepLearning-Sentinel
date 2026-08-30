"""ESA-ADB -> Cloudflare R2 ingest.

    spike     validate the environment against the real archive (~80 MB, no R2 ops)
    download  fetch the three source archives and verify their MD5s
    prepare   transcode to parquet and size-check locally  (no R2 ops)
    upload    push to R2, verify, then publish the manifest  (needs --confirm)
    clean     remove the scratch directory

prepare performs zero R2 operations and stops with a report. Nothing touches R2
until a human has read that report and re-invoked with `upload --confirm`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from . import config as C
from . import docs_gen, esa_adb, manifest as M, prepare as P, r2, transcode, upload as U, zenodo


def _fmt(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# --------------------------------------------------------------------------
def cmd_spike(args) -> int:
    """Retire the environment risks before committing to an 11.6 GB download."""
    src = C.SOURCE_FILES[2]  # Mission3, the smallest archive
    print(f"SPIKE  source: {src.key} ({src.size/1e9:.2f} GB) via HTTP range requests\n")
    fh = zenodo.open_remote(src)
    arch = esa_adb.open_fileobj(fh, src.mission, scratch=C.SCRATCH)

    print(f"  archive root        : {arch.root}")
    channels = arch.channel_members()
    print(f"  channel members     : {len(channels)}  (expected {C.EXPECTED_CHANNELS[src.mission]})")
    print(f"  telecommand members : {len(arch.telecommand_members())}  "
          f"(paper says {C.EXPECTED_TELECOMMANDS[src.mission]})")
    ok = True

    smallest = min(channels, key=lambda n: arch.zf.getinfo(n).compress_size)
    print(f"\n  [1] unpickle test on {smallest.split('/')[-1]}")
    try:
        import pandas as pd
        df = arch.read_series(smallest)
        print(f"      pandas {pd.__version__} loaded a pandas 1.5.3 pickle")
        print(f"      rows {len(df):,}   ts {df['timestamp'].dtype}   value {df['value'].dtype}")
        print(f"      span {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    except Exception as exc:
        ok = False
        print(f"      FAILED: {type(exc).__name__}: {exc}")

    print("\n  [2] channels.csv (Deflate64) test")
    try:
        ch = arch.read_csv("channels.csv")
        print(f"      columns {list(ch.columns)}")
        for col in ("Target", "Categorical"):
            print(f"      {col:12s} {sorted(map(str, ch[col].unique()))}")
    except Exception as exc:
        ok = False
        print(f"      FAILED: {type(exc).__name__}: {exc}")

    print("\n  [3] annotation CSVs")
    for name in ("labels.csv", "anomaly_types.csv", "telecommands.csv", "events.csv"):
        if not arch.has_member(name):
            print(f"      {name:20s} absent")
            continue
        d = arch.read_csv(name)
        print(f"      {name:20s} {len(d):6,} rows  {list(d.columns)}")

    raw = fh.raw  # type: ignore[attr-defined]
    print(f"\n  network: {raw.requests_made} range requests, {_fmt(raw.bytes_fetched)} fetched")
    print(f"\nSPIKE {'PASSED' if ok else 'FAILED'} -- R2 operations used: 0")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def cmd_download(args) -> int:
    for src in C.SOURCE_FILES:
        zenodo.download(src, C.RAW_DIR)
    print("all archives present and MD5-verified")
    return 0


# --------------------------------------------------------------------------
def cmd_transcode(args) -> int:
    """Transcode one mission to parquet. CPU-bound, so it overlaps cleanly with
    downloading the next archive. Writes a per-mission plan fragment; the final
    `prepare` assembles the fragments and costs nothing extra, because every
    channel is already cached."""
    src = next(s for s in C.SOURCE_FILES if s.mission == args.mission)
    path = C.RAW_DIR / src.key
    if not path.exists() or path.stat().st_size != src.size:
        print(f"{src.key} is not fully downloaded yet", file=sys.stderr)
        return 2
    arch = esa_adb.open_local(path, src.mission, scratch=C.SCRATCH)
    cached = P.cache_annotations(arch, src.mission)
    print(f"  cached annotation CSVs: {cached}")
    entry = P.process_mission(src, arch)
    out = C.SCRATCH / f"plan_{src.mission}.json"
    out.write_text(json.dumps(entry, indent=2))

    objs = [o for c in entry["channels"] for o in (c["shards"] or [c])]
    if entry.get("telecommand_series"):
        objs.append(entry["telecommand_series"])
    largest = max(objs, key=lambda o: o["bytes"])
    sharded = sum(1 for c in entry["channels"] if c.get("shards"))
    print(f"\n  {src.mission}: {entry['channel_count']} channels "
          f"({entry['target_channel_count']} target), {len(objs)} objects")
    print(f"    total   {sum(o['bytes'] for o in objs)/1e9:.2f} GB")
    print(f"    largest {largest['bytes']/1048576:.1f} MiB  "
          f"[limit {C.MAX_OBJECT_BYTES/1048576:.0f} MiB]  "
          f"{'OK' if largest['bytes'] < C.MAX_OBJECT_BYTES else 'FLAG'}")
    print(f"    sharded channels: {sharded}")
    print(f"    wrote {out}")
    return 0


# --------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    print("PREPARE  (zero R2 operations)\n")
    missions = {}
    for src in C.SOURCE_FILES:
        frag = C.SCRATCH / f"plan_{src.mission}.json"
        if not frag.exists():
            print(f"missing {frag}; run `transcode --mission {src.mission}` first",
                  file=sys.stderr)
            return 2
        missions[src.mission] = json.loads(frag.read_text())

    print("  merging annotations from the cached CSVs")
    annotations = P.build_annotations(list(missions))

    plan = M.build(missions, annotations)
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    C.PREFLIGHT_FILE.write_text(json.dumps(plan, indent=2))

    objects = U._objects_with_paths(plan)
    sharded = sum(
        1 for m in missions.values() for c in m["channels"] if c.get("shards")
    )
    largest = max(objects, key=lambda o: o["bytes"])
    total = sum(o["bytes"] for o in objects)
    class_a = len(objects) + 2
    class_b = len(objects) + 2 + 2
    peak = P.peak_rss_bytes()

    checks = []
    checks.append(("Largest object", _fmt(largest["bytes"]),
                   f"[limit {C.MAX_OBJECT_BYTES/1048576:.0f} MiB]",
                   largest["bytes"] < C.MAX_OBJECT_BYTES))
    checks.append(("Total upload size", f"{total/1e9:.2f} GB", "[R2 free tier 10 GB]", total < 10e9))
    checks.append(("Peak RSS", _fmt(peak), "[of 16 GB]", peak < 8 * 1024**3))
    checks.append(("Projected Class A", f"{class_a:,}", f"[tripwire {C.OPS_TRIPWIRE:,}]",
                   class_a < C.OPS_TRIPWIRE))
    checks.append(("Projected Class B", f"{class_b:,}", f"[tripwire {C.OPS_TRIPWIRE:,}]",
                   class_b < C.OPS_TRIPWIRE))
    checks.append(("Channels sharded", str(sharded), "[expect 0 unless flagged]", sharded == 0))

    print("\n  ESA-ADB PRE-FLIGHT")
    print(f"    Missions:            {len(missions)}")
    for name in sorted(missions):
        m = missions[name]
        tc = "  +telecommands" if m.get("telecommand_series") else ""
        print(f"      {name}: {m['channel_count']} channels "
              f"({m['target_channel_count']} target){tc}")
    print(f"    Objects:             {len(objects)}  "
          f"(annotations: {len(annotations)})")
    for label, value, limit, ok in checks:
        print(f"    {label:20s} {value:>12s}  {limit:28s} {'OK' if ok else 'FLAG'}")

    all_ok = all(c[3] for c in checks)
    print(f"\n  {'all checks green' if all_ok else 'ATTENTION REQUIRED -- see FLAG rows above'}")
    print(f"  wrote {C.PREFLIGHT_FILE}")
    print("\n  to upload:  python -m sentinel_data.ingest_esa_adb upload --confirm")
    return 0


# --------------------------------------------------------------------------
def _load_state() -> dict:
    return json.loads(C.STATE_FILE.read_text()) if C.STATE_FILE.exists() else {}


def _save_state(state: dict) -> None:
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    C.STATE_FILE.write_text(json.dumps(state, indent=2))


def cmd_upload_stage(args) -> int:
    """Upload and verify ONE mission's data objects. Never writes the manifest.

    Staging is safe precisely because the manifest is still written last: until
    it exists, the bucket holds data objects that nothing refers to, and a
    consumer reading the manifest first finds nothing rather than a half-truth.
    """
    frag = C.SCRATCH / f"plan_{args.stage}.json"
    if not frag.exists():
        print(f"no {frag}; run `transcode --mission {args.stage}` first", file=sys.stderr)
        return 2

    entry = json.loads(frag.read_text())
    objects = U.objects_of_mission(entry)
    cfg = C.load_r2_config()
    client, counter = r2.make_client(cfg)
    state = _load_state()

    print(f"UPLOAD STAGE {args.stage} -> {cfg.bucket}   ({len(objects)} objects)\n")
    done = set(state.get("uploaded", []))
    try:
        for i, obj in enumerate(objects, 1):
            if obj["key"] in done:
                continue
            r2.put(client, cfg.bucket, obj["key"], obj["local_path"], obj["md5_b64"],
                   metadata={"sha256": obj["sha256"], "rows": obj.get("rows", 0),
                             "channel_id": obj.get("channel_id", "")})
            done.add(obj["key"])
            # persist as we go: a killed run must not forget what it already wrote
            if i % 10 == 0:
                state["uploaded"] = sorted(done)
                _save_state(state)
            print(f"\r  uploaded {i:3d}/{len(objects)}  {obj['key'][-58:]:58s}",
                  end="", flush=True)
        print()
    finally:
        state["uploaded"] = sorted(done)
        _save_state(state)

    verified = set(state.get("verified", []))
    problems = U.verify_all(client, cfg.bucket, objects, skip=verified)
    if problems:
        print(f"\n  VERIFICATION FAILED ({len(problems)}). No manifest exists, so nothing "
              "downstream can read a broken dataset.")
        for pr in problems[:20]:
            print(f"    {pr}")
        return 1
    verified |= {o["key"] for o in objects}
    state["verified"] = sorted(verified)
    _save_state(state)

    # Everything for this mission is now in R2 and checksum-verified, and its
    # annotation CSVs are cached, so the local copies are pure redundancy.
    freed = 0
    src = next(s for s in C.SOURCE_FILES if s.mission == args.stage)
    raw = C.RAW_DIR / src.key
    if raw.exists():
        freed += raw.stat().st_size
        raw.unlink()
    pq = C.PARQUET_DIR / args.stage
    if pq.exists():
        freed += sum(f.stat().st_size for f in pq.rglob("*") if f.is_file())
        shutil.rmtree(pq)
    print(f"\n  purged local copies of {args.stage}: {freed/1e9:.2f} GB freed")

    print(f"  stage {args.stage} complete and verified")
    print(counter.summary())
    print("  manifest NOT written -- it is published only after every mission verifies")
    return 0


def cmd_upload(args) -> int:
    if args.stage:
        return cmd_upload_stage(args)
    if not args.confirm:
        print("refusing to upload without --confirm", file=sys.stderr)
        return 2
    if not C.PREFLIGHT_FILE.exists():
        print(f"no {C.PREFLIGHT_FILE}; run `prepare` first", file=sys.stderr)
        return 2

    plan = json.loads(C.PREFLIGHT_FILE.read_text())
    cfg = C.load_r2_config()
    client, counter = r2.make_client(cfg)
    state = _load_state()

    print(f"UPLOAD -> {cfg.bucket}\n")
    try:
        objects = U.upload_all(client, cfg.bucket, plan, state)
    finally:
        _save_state(state)

    problems = U.verify_all(client, cfg.bucket, objects, skip=set(state.get("verified", [])))
    if problems:
        print(f"\n  VERIFICATION FAILED ({len(problems)} problems). "
              "No manifest written -- the bucket has no index, which is the safe state.")
        for p in problems[:20]:
            print(f"    {p}")
        return 1

    published = U.publish_manifest(client, cfg.bucket, plan)
    U.publish_ledger(client, cfg.bucket, counter)
    for key in (C.MANIFEST_KEY, C.LEDGER_KEY):
        r2.head(client, cfg.bucket, key)

    written = docs_gen.regenerate(published, cfg.bucket)
    for path in written:
        print(f"  wrote {path.relative_to(C.PROJECT_ROOT)}")

    print("\n  INGEST COMPLETE")
    print(counter.summary())
    print(f"    Objects written: {len(objects) + 2}")
    print(f"    Total stored:  {sum(o['bytes'] for o in objects)/1e9:.2f} GB")

    shutil.rmtree(C.SCRATCH, ignore_errors=True)
    assert not C.SCRATCH.exists(), "scratch directory was not removed"
    print(f"    Scratch deleted: {not C.SCRATCH.exists()}")
    return 0


def cmd_clean(args) -> int:
    shutil.rmtree(C.SCRATCH, ignore_errors=True)
    print(f"scratch removed: {not C.SCRATCH.exists()}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ingest_esa_adb", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("spike", "download", "prepare", "clean"):
        sub.add_parser(name)
    tr = sub.add_parser("transcode")
    tr.add_argument("--mission", required=True, choices=sorted(C.EXPECTED_CHANNELS))
    up = sub.add_parser("upload")
    up.add_argument("--confirm", action="store_true")
    up.add_argument("--stage", choices=sorted(C.EXPECTED_CHANNELS),
                    help="upload and verify one mission only; never writes the manifest")
    args = p.parse_args(argv)
    return {
        "spike": cmd_spike, "download": cmd_download, "prepare": cmd_prepare,
        "transcode": cmd_transcode, "upload": cmd_upload, "clean": cmd_clean,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
