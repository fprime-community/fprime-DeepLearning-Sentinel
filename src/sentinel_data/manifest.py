"""The manifest: the contract between this ingest and every downstream consumer.

Code reads keys from here and never discovers them, because discovery means LIST
and LIST is both a Class A operation and banned by constraint 4.

The manifest is uploaded LAST, only after every object it references has been
verified. Its presence in the bucket is therefore the guarantee that everything
it names exists and matched its checksum.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import (
    ANONYMISATION,
    CONCEPT_DOI,
    DATASET_LICENSE,
    MANIFEST_SCHEMA_VERSION,
    REFERENCE_HINTS,
    SOURCE_DOI,
    SOURCE_FILES,
    ZENODO_RECORD,
)

REQUIRED_CHANNEL_FIELDS = (
    "channel_id", "key", "shards", "rows", "bytes", "sha256",
    "subsystem", "channel_group", "is_target", "is_categorical",
    "value_dtype", "time_start_utc", "time_end_utc",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(missions: dict, annotations: dict) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_utc": now_utc(),
        "datasets": {
            "esa-adb": {
                "version": "v1",
                "source_doi": SOURCE_DOI,
                "concept_doi": CONCEPT_DOI,
                "source_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "license": DATASET_LICENSE,
                "source_files": [
                    {"key": s.key, "bytes": s.size, "md5": s.md5} for s in SOURCE_FILES
                ],
                "anonymisation": ANONYMISATION,
                "missions": missions,
                "annotations": annotations,
                "reference_hints": REFERENCE_HINTS,
            }
        },
    }


def all_objects(manifest: dict) -> list[dict]:
    """Every object the manifest references, flattened. The only key source."""
    out: list[dict] = []
    ds = manifest["datasets"]["esa-adb"]
    for mission in ds["missions"].values():
        for ch in mission["channels"]:
            if ch.get("shards"):
                out.extend(ch["shards"])
            else:
                out.append(ch)
        tc = mission.get("telecommand_series")
        if tc:
            if tc.get("shards"):
                out.extend(tc["shards"])
            else:
                out.append(tc)
    out.extend(ds["annotations"].values())
    return [o for o in out if o.get("key")]


def validate(manifest: dict) -> None:
    """Structural validation. Raises on the first problem found."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {manifest.get('schema_version')}")
    ds = manifest["datasets"]["esa-adb"]

    seen: set[str] = set()
    for name, mission in ds["missions"].items():
        if len(mission["channels"]) != mission["channel_count"]:
            raise ValueError(
                f"{name}: channel_count {mission['channel_count']} != "
                f"{len(mission['channels'])} entries"
            )
        for ch in mission["channels"]:
            missing = [f for f in REQUIRED_CHANNEL_FIELDS if f not in ch]
            if missing:
                raise ValueError(f"{name}/{ch.get('channel_id')}: missing fields {missing}")
            if bool(ch["key"]) == bool(ch["shards"]):
                raise ValueError(
                    f"{name}/{ch['channel_id']}: exactly one of key / shards must be set"
                )

    for obj in all_objects(manifest):
        for field in ("key", "bytes", "sha256"):
            if not obj.get(field):
                raise ValueError(f"object {obj.get('key')!r} missing {field}")
        if obj["key"] in seen:
            raise ValueError(f"duplicate key {obj['key']}")
        seen.add(obj["key"])
        if len(obj["sha256"]) != 64:
            raise ValueError(f"{obj['key']}: sha256 is not 64 hex characters")


def dumps(manifest: dict) -> bytes:
    return json.dumps(manifest, indent=2, sort_keys=False).encode() + b"\n"
