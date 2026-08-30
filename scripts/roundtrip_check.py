#!/usr/bin/env python3
"""Prove the round trip: manifest -> key -> object -> DataFrame.

This is the definition-of-done check. It reads the manifest, resolves one
channel's key from it, fetches that object and prints what came back. It never
lists the bucket and never constructs a key by hand -- which is exactly how
every downstream consumer is expected to behave.

Cost: 2 Class B operations (one GET for the manifest, one for the channel).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sentinel_data import config as C, r2


def main() -> int:
    cfg = C.load_r2_config()
    client, counter = r2.make_client(cfg)

    manifest = json.loads(r2.get_bytes(client, cfg.bucket, C.MANIFEST_KEY))
    esa = manifest["datasets"]["esa-adb"]
    print(f"manifest schema {manifest['schema_version']}, generated {manifest['generated_utc']}")
    print(f"  source {esa['source_doi']}  licence {esa['license']}")

    mission_name = sorted(esa["missions"])[0]
    mission = esa["missions"][mission_name]
    channel = next(
        (c for c in mission["channels"] if c["is_target"] and not c["is_categorical"]),
        mission["channels"][0],
    )
    keys = [s["key"] for s in channel["shards"]] if channel["shards"] else [channel["key"]]
    print(f"\n  resolved {mission_name}/{channel['channel_id']} -> {len(keys)} object(s)")
    for k in keys:
        print(f"    {k}")

    frames = [
        pd.read_parquet(io.BytesIO(r2.get_bytes(client, cfg.bucket, k))) for k in keys
    ]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    print(f"\n  shape       : {df.shape}")
    print(f"  dtypes      : {df.dtypes.to_dict()}")
    print(f"  time range  : {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[-1]}")
    print(f"  subsystem   : {channel['subsystem']}   group {channel['channel_group']}")
    print(f"  manifest rows {channel['rows']:,}  vs fetched {len(df):,}  "
          f"{'MATCH' if channel['rows'] == len(df) else 'MISMATCH'}")

    print("\n  ROUND TRIP OK")
    print(counter.summary())
    return 0 if channel["rows"] == len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
