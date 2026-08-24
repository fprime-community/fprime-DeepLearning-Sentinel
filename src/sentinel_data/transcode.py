"""DataFrame -> parquet, with the 90 MiB object ceiling enforced at write time.

Constraint 2 says every uploaded object must be under 90 MiB so that each one is
a single PutObject, and therefore exactly one Class A operation. That is asserted
here, before an object can ever become a candidate for upload. Anything too large
is split on time boundaries into shards that are each independently checked.
"""
from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    MAX_OBJECT_BYTES,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
)

# Leave headroom so a shard that lands slightly over its estimate still passes.
SHARD_TARGET_BYTES = 80 * 1024 * 1024
READ_CHUNK = 8 * 1024 * 1024


@dataclass
class ObjectSpec:
    """One object destined for R2, fully described before it is uploaded."""

    local_path: Path
    key: str = ""
    rows: int = 0
    nbytes: int = 0
    sha256: str = ""
    md5_hex: str = ""
    md5_b64: str = ""
    time_start: str | None = None
    time_end: str | None = None
    extra: dict = field(default_factory=dict)

    def finalise(self) -> "ObjectSpec":
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with open(self.local_path, "rb") as fh:
            for block in iter(lambda: fh.read(READ_CHUNK), b""):
                sha.update(block)
                md5.update(block)
        self.nbytes = self.local_path.stat().st_size
        self.sha256 = sha.hexdigest()
        self.md5_hex = md5.hexdigest()
        self.md5_b64 = base64.b64encode(md5.digest()).decode()
        if self.nbytes >= MAX_OBJECT_BYTES:
            raise AssertionError(
                f"{self.local_path.name} is {self.nbytes/1048576:.1f} MiB, "
                f"at or above the {MAX_OBJECT_BYTES/1048576:.0f} MiB single-PUT ceiling"
            )
        return self


def _table(df: pd.DataFrame) -> pa.Table:
    """Build the archive schema, preserving dtypes exactly.

    Timestamps stay nanosecond, values keep their native type: some channels are
    categorical and arrive as strings, and the anonymisation is documented as
    numerically lossless, so nothing is downcast here.
    """
    ts = pa.array(df["timestamp"].to_numpy(), type=pa.timestamp("ns"))
    values = pa.array(df["value"].to_numpy())
    return pa.table({"timestamp": ts, "value": values})


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        version="2.6",  # nanosecond timestamps
        use_dictionary=True,
        store_schema=True,
    )


def _span(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    return (
        pd.Timestamp(df["timestamp"].iloc[0]).isoformat(),
        pd.Timestamp(df["timestamp"].iloc[-1]).isoformat(),
    )


def write_series(df: pd.DataFrame, out_dir: Path, stem: str) -> list[ObjectSpec]:
    """Write one series as parquet, sharding on time if it exceeds the ceiling.

    Returns one spec for a normal series, or several for a sharded one. Sharding
    is a first-class path, not an error branch -- the largest channels are
    expected to need it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    single = out_dir / f"{stem}.parquet"
    _write(_table(df), single)

    if single.stat().st_size < MAX_OBJECT_BYTES:
        start, end = _span(df)
        return [ObjectSpec(single, rows=len(df), time_start=start, time_end=end).finalise()]

    # Too large: split into equal row ranges sized from the observed bytes/row,
    # growing the part count until every shard fits.
    observed = single.stat().st_size
    single.unlink()
    parts = max(2, math.ceil(observed / SHARD_TARGET_BYTES))

    while True:
        specs: list[ObjectSpec] = []
        edges = [round(i * len(df) / parts) for i in range(parts + 1)]
        too_big = False
        for i in range(parts):
            chunk = df.iloc[edges[i] : edges[i + 1]]
            path = out_dir / f"{stem}.part{i:03d}.parquet"
            _write(_table(chunk), path)
            if path.stat().st_size >= MAX_OBJECT_BYTES:
                too_big = True
                break
            start, end = _span(chunk)
            specs.append(
                ObjectSpec(path, rows=len(chunk), time_start=start, time_end=end).finalise()
            )
        if not too_big:
            return specs
        for s in specs:
            s.local_path.unlink(missing_ok=True)
        for leftover in out_dir.glob(f"{stem}.part*.parquet"):
            leftover.unlink(missing_ok=True)
        parts += 1


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> ObjectSpec:
    """Write a small annotation or description table as a single object."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    _write(table, path)
    return ObjectSpec(path, rows=len(df)).finalise()
