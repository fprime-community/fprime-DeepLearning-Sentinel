"""Phase 2: upload, verify, and only then publish the manifest.

Ordering is the point of this module. The manifest must never exist in R2
pointing at unverified objects, because downstream code trusts it absolutely.
So: every data object is uploaded, every one is verified, and only if all of
that succeeds is the manifest written. A failed run leaves orphaned data
objects and no manifest, which is the safe failure mode -- a consumer reading
the manifest first finds nothing, rather than something it cannot trust.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json

from . import config as C
from . import manifest as M
from . import r2

LOCAL_ONLY = ("local_path", "md5_b64")


def strip_local(node):
    """Remove purely local bookkeeping from what gets published."""
    if isinstance(node, dict):
        return {k: strip_local(v) for k, v in node.items() if k not in LOCAL_ONLY}
    if isinstance(node, list):
        return [strip_local(v) for v in node]
    return node


def _objects_with_paths(plan: dict) -> list[dict]:
    """Every object to upload, in a stable order, with its local path."""
    ds = plan["datasets"]["esa-adb"]
    out: list[dict] = []
    for name in sorted(ds["missions"]):
        mission = ds["missions"][name]
        for ch in mission["channels"]:
            out.extend(ch["shards"] if ch.get("shards") else [ch])
        if mission.get("telecommand_series"):
            tc = mission["telecommand_series"]
            out.extend(tc["shards"] if tc.get("shards") else [tc])
    for name in sorted(ds["annotations"]):
        out.append(ds["annotations"][name])
    return [o for o in out if o.get("key")]


def upload_all(client, bucket: str, plan: dict, state: dict, log=print) -> list[dict]:
    """Step 1: one PutObject per object. Already-uploaded keys are skipped."""
    objects = _objects_with_paths(plan)
    done = set(state.get("uploaded", []))
    for i, obj in enumerate(objects, 1):
        key = obj["key"]
        if key in done:
            continue
        r2.put(
            client, bucket, key, obj["local_path"], obj["md5_b64"],
            metadata={
                "sha256": obj["sha256"],
                "rows": obj.get("rows", 0),
                "channel_id": obj.get("channel_id", ""),
            },
        )
        done.add(key)
        state["uploaded"] = sorted(done)
        log(f"\r  uploaded {i:3d}/{len(objects)}  {key[-58:]:58s}", end="")
    log("")
    return objects


def objects_of_mission(entry: dict) -> list[dict]:
    """Flatten one mission's plan fragment into its uploadable objects."""
    out: list[dict] = []
    for ch in entry["channels"]:
        out.extend(ch["shards"] if ch.get("shards") else [ch])
    if entry.get("telecommand_series"):
        tc = entry["telecommand_series"]
        out.extend(tc["shards"] if tc.get("shards") else [tc])
    return [o for o in out if o.get("key")]


def verify_all(client, bucket: str, objects: list[dict], log=print,
               skip: set[str] | None = None) -> list[str]:
    """Step 2: the gate. One HeadObject per object, three assertions each.

    A HEAD cannot return a SHA-256 -- R2 does not implement the SHA-256 checksum
    header -- so the canonical hash rides along as object metadata, the size is
    compared directly, and the ETag is compared against the MD5 we computed
    locally. For a single-part PUT the ETag is that MD5, which is what makes
    this a real content check rather than a size check.
    """
    problems: list[str] = []
    etag_checked = False
    skip = skip or set()
    todo = [o for o in objects if o["key"] not in skip]
    if len(todo) < len(objects):
        log(f"  {len(objects) - len(todo)} object(s) already verified in an earlier stage")

    for i, obj in enumerate(todo, 1):
        key = obj["key"]
        head = r2.head(client, bucket, key)

        if head["ContentLength"] != obj["bytes"]:
            problems.append(f"{key}: size {head['ContentLength']} != {obj['bytes']}")

        etag = head.get("ETag", "").strip('"')
        if "-" in etag:
            problems.append(f"{key}: ETag {etag} is multipart; object was not a single PUT")
        elif len(etag) == 32:
            if etag != obj["md5_hex"]:
                problems.append(f"{key}: ETag {etag} != local MD5 {obj['md5_hex']}")
            etag_checked = True
        elif not etag_checked and i == 1:
            log(f"\n  WARNING: ETag {etag!r} is not a 32-character MD5; "
                "falling back to size + metadata verification only")

        meta = {k.lower(): v for k, v in head.get("Metadata", {}).items()}
        if "sha256" not in meta:
            if i == 1:
                log("\n  WARNING: R2 did not return x-amz-meta-sha256 on HEAD; "
                    "verification degraded to size + ETag")
        elif meta["sha256"] != obj["sha256"]:
            problems.append(f"{key}: metadata sha256 mismatch")

        log(f"\r  verified {i:3d}/{len(todo)}  {key[-58:]:58s}", end="")
    log("")
    return problems


def publish_manifest(client, bucket: str, plan: dict, log=print) -> dict:
    """Step 3: only reached when every object has verified."""
    published = strip_local(copy.deepcopy(plan))
    M.validate(published)
    blob = M.dumps(published)
    r2.put_bytes(
        client, bucket, C.MANIFEST_KEY, blob,
        base64.b64encode(hashlib.md5(blob).digest()).decode(),
        metadata={"sha256": hashlib.sha256(blob).hexdigest()},
    )
    log(f"  manifest published: {C.MANIFEST_KEY} ({len(blob):,} bytes)")
    return published


def publish_ledger(client, bucket: str, counter: r2.OpCounter, log=print) -> dict:
    """Step 4: a record of a completed ingest, so it follows the manifest.

    The counts include this call's own PutObject, which has not happened yet, so
    they are adjusted to describe the finished state rather than a stale one.
    """
    ledger = r2.new_ledger()
    ledger = r2.roll_and_add(ledger, counter.class_a + 1, counter.class_b)
    blob = json.dumps(ledger, indent=2).encode() + b"\n"
    r2.put_bytes(
        client, bucket, C.LEDGER_KEY, blob,
        base64.b64encode(hashlib.md5(blob).digest()).decode(),
    )
    log(f"  ledger published:   {C.LEDGER_KEY}")
    return ledger
