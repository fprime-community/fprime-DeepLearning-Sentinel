"""Phase 1: download, transcode and locally verify. Zero R2 operations.

Runs to completion and stops. Nothing touches R2 until a human has read the
pre-flight report and re-invoked with `upload --confirm`.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from . import config as C
from . import esa_adb, transcode, zenodo
from .transcode import ObjectSpec

YES = {"YES", "Y", "TRUE", "1"}


def peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; Linux in kibibytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() in YES


def _spec_to_entry(spec: ObjectSpec, key: str) -> dict:
    return {
        "key": key,
        "rows": spec.rows,
        "bytes": spec.nbytes,
        "sha256": spec.sha256,
        "md5_hex": spec.md5_hex,
        "md5_b64": spec.md5_b64,
        "local_path": str(spec.local_path),
        "time_start_utc": spec.time_start,
        "time_end_utc": spec.time_end,
    }


def _assign(specs: list[ObjectSpec], prefix: str) -> list[dict]:
    return [_spec_to_entry(s, f"{prefix}/{s.local_path.name}") for s in specs]


def _channel_metadata(arch: esa_adb.MissionArchive) -> dict[str, dict]:
    df = arch.read_csv("channels.csv")
    meta: dict[str, dict] = {}
    for row in df.to_dict("records"):
        meta[str(row["Channel"])] = {
            "subsystem": row.get("Subsystem"),
            "channel_group": int(row["Group"]) if pd.notna(row.get("Group")) else None,
            "is_target": _truthy(row.get("Target")),
            "is_categorical": _truthy(row.get("Categorical")),
            "physical_unit": None if pd.isna(row.get("Physical Unit")) else row.get("Physical Unit"),
        }
    return meta


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _append_cache(path: Path, record: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def process_mission(src, arch: esa_adb.MissionArchive, log=print) -> dict:
    """Transcode one mission. Resumable: completed channels are cached."""
    mission = src.mission
    out_dir = C.PARQUET_DIR / mission
    cache_path = C.SCRATCH / f"cache_{mission}.jsonl"
    cache = _load_cache(cache_path)

    meta = _channel_metadata(arch)
    members = arch.channel_members()
    expected = C.EXPECTED_CHANNELS[mission]
    if len(members) != expected:
        raise RuntimeError(f"{mission}: found {len(members)} channels, expected {expected}")

    channels: list[dict] = []
    t0 = time.monotonic()
    for i, member in enumerate(members, 1):
        cid = arch.id_of(member)
        if cid in cache:
            channels.append(cache[cid]["entry"])
            continue

        df = arch.read_series(member)
        specs = transcode.write_series(df, out_dir, f"ch_{cid}")
        dtype = str(df["value"].dtype)
        nrows = len(df)
        del df

        objs = _assign(specs, f"{C.ARCHIVE_PREFIX}/{mission}")
        info = meta.get(cid, {})
        entry = {
            "channel_id": cid,
            "key": objs[0]["key"] if len(objs) == 1 else None,
            "shards": None if len(objs) == 1 else objs,
            "rows": nrows,
            "bytes": sum(o["bytes"] for o in objs),
            "sha256": objs[0]["sha256"] if len(objs) == 1 else "",
            "md5_hex": objs[0]["md5_hex"] if len(objs) == 1 else "",
            "md5_b64": objs[0]["md5_b64"] if len(objs) == 1 else "",
            "local_path": objs[0]["local_path"] if len(objs) == 1 else None,
            "subsystem": info.get("subsystem"),
            "channel_group": info.get("channel_group"),
            "is_target": info.get("is_target", False),
            "is_categorical": info.get("is_categorical", False),
            "physical_unit": info.get("physical_unit"),
            "value_dtype": dtype,
            "time_start_utc": objs[0]["time_start_utc"],
            "time_end_utc": objs[-1]["time_end_utc"],
        }
        channels.append(entry)
        _append_cache(cache_path, {"id": cid, "entry": entry})

        rate = (time.monotonic() - t0) / max(i - len(cache), 1)
        log(
            f"\r  {mission}  {i:3d}/{len(members)}  {cid:14s} "
            f"{nrows:>12,} rows  {sum(o['bytes'] for o in objs)/1048576:6.1f} MiB"
            f"{'  SHARDED x' + str(len(objs)) if len(objs) > 1 else '':>14s}"
            f"  ~{rate*(len(members)-i)/60:4.1f} min left   ",
            end="",
        )
    log("")

    mission_entry = {
        "channel_count": len(channels),
        "target_channel_count": sum(1 for c in channels if c["is_target"]),
        "channels": channels,
    }

    # Telecommand series: merged to one object per mission, long format.
    tc_members = arch.telecommand_members()
    if tc_members:
        tc_cache = C.SCRATCH / f"cache_{mission}_tc.json"
        if tc_cache.exists():
            mission_entry["telecommand_series"] = json.loads(tc_cache.read_text())
        else:
            frames = []
            for j, member in enumerate(tc_members, 1):
                tid = arch.id_of(member)
                d = arch.read_series(member)
                d = d[~d["timestamp"].duplicated()]  # telecommand indices can repeat
                d.insert(0, "telecommand_id", tid)
                frames.append(d)
                log(f"\r  {mission}  telecommands {j:4d}/{len(tc_members)}   ", end="")
            log("")
            merged = pd.concat(frames, ignore_index=True)
            del frames
            merged = merged.sort_values(["timestamp", "telecommand_id"], kind="stable")
            # write_series is for two-column series; telecommands carry an id too
            spec = transcode.write_table(merged, out_dir, "telecommands")
            entry = _spec_to_entry(spec, f"{C.ARCHIVE_PREFIX}/{mission}/telecommands.parquet")
            entry["telecommand_files"] = len(tc_members)
            entry["distinct_telecommands"] = int(merged["telecommand_id"].nunique())
            entry["empty_telecommands"] = len(tc_members) - int(merged["telecommand_id"].nunique())
            entry["time_start_utc"] = pd.Timestamp(merged["timestamp"].iloc[0]).isoformat()
            entry["time_end_utc"] = pd.Timestamp(merged["timestamp"].iloc[-1]).isoformat()
            del merged
            tc_cache.write_text(json.dumps(entry))
            mission_entry["telecommand_series"] = entry
    return mission_entry


ANNOTATION_CSVS = ("labels.csv", "anomaly_types.csv", "channels.csv",
                   "telecommands.csv", "events.csv")


def cache_annotations(arch: esa_adb.MissionArchive, mission: str) -> list[str]:
    """Copy a mission's annotation CSVs into scratch while the archive is open.

    These are a few hundred KB in total. Caching them means the multi-GB source
    archive can be deleted the moment its channels are uploaded and verified,
    instead of being kept alive purely so the manifest can be assembled later.
    """
    out = C.SCRATCH / "annotations_cache"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ANNOTATION_CSVS:
        if not arch.has_member(name):
            continue
        (out / f"{mission}_{name}").write_bytes(arch.read_bytes(name))
        written.append(name)
    return written


def _cached_csv(mission: str, name: str):
    path = C.SCRATCH / "annotations_cache" / f"{mission}_{name}"
    return pd.read_csv(path) if path.exists() else None


def build_annotations(missions: list[str]) -> dict:
    """Merge every annotation CSV into as few objects as possible.

    labels.csv carries no class or category -- every consumer joins it to
    anomaly_types.csv on ID. Doing that join here means the harness reads one
    object instead of two on every experiment run.
    """
    labels, channels, telecommands, events = [], [], [], []

    for mission in missions:
        lab = _cached_csv(mission, "labels.csv")
        types = _cached_csv(mission, "anomaly_types.csv")
        if lab is None or types is None:
            raise RuntimeError(
                f"{mission}: annotation CSVs missing from the cache. They are copied "
                f"during `transcode --mission {mission}`; re-run it."
            )
        merged = lab.merge(types, on="ID", how="left", validate="many_to_one")
        merged.insert(0, "mission", mission)
        labels.append(merged)

        ch = _cached_csv(mission, "channels.csv")
        ch.insert(0, "mission", mission)
        channels.append(ch)

        tc = _cached_csv(mission, "telecommands.csv")
        if tc is not None:
            tc.insert(0, "mission", mission)
            telecommands.append(tc)

        ev = _cached_csv(mission, "events.csv")
        if ev is not None:
            ev.insert(0, "mission", mission)
            events.append(ev)

    out = {}
    frames = {
        "labels": labels,
        "channels": channels,
        "telecommands": telecommands,
        "events": events,
    }
    for name, parts in frames.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        for col in ("StartTime", "EndTime"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True).dt.tz_localize(None)
        spec = transcode.write_table(df, C.PARQUET_DIR / "annotations", name)
        out[name] = _spec_to_entry(spec, f"{C.ANNOTATIONS_PREFIX}/{name}.parquet")
    return out
