"""A generated stand-in for the bucket, so development costs nothing.

Rule 1 forbids caching data locally and Rule 4 caps operations per month, which
together create a problem: a warm run is impossible, so every real run costs
operations, and iterating on the harness would burn through the 1,000 tripwire in
a hundred runs. The way out is not to weaken either rule -- it is to make the
whole harness runnable against data that never came from R2.

This module builds a **fake bucket**, not a fake bundle. It emits real parquet
bytes under real manifest keys with real SHA-256 digests, so `catalog.py`,
`read.py`, `labels.py` and everything above them execute their genuine code
paths, checksum verification included. Only the transport is different, and it
is different behind :class:`~sentinel_eval.read.ObjectSource`.

What it deliberately reproduces, because each of these has broken something:

* **a sharded channel** with ``key: null`` and an empty parent ``sha256``,
* **cross-channel structure** -- one latent driver, per-channel lags -- so a
  detector blind to it genuinely underperforms rather than merely scoring lower,
* **all six anomaly taxonomy cells** the real labels use, point cells included,
* **rare nominal events** that look anomalous and are not,
* **communication gaps and invalid segments**, which are unscorable, not nominal,
* **per-channel irregular sampling**, so resampling is exercised,
* **group-wide min-max scaling**, preserving amplitude ratios within a group,
* **a categorical (string) channel**, which arithmetic must refuse.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MISSION = "missionX"
EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")
PERIOD_S = 30

#: Anomaly cells injected, matching ESA-ADB's Dimensionality/Locality/Length.
TAXONOMY = (
    ("Multivariate", "Global", "Subsequence"),
    ("Multivariate", "Global", "Point"),
    ("Multivariate", "Local", "Subsequence"),
    ("Univariate", "Global", "Subsequence"),
    ("Univariate", "Local", "Subsequence"),
    ("Univariate", "Local", "Point"),
)


@dataclass
class SyntheticBucket:
    """An in-memory bucket. Implements the :class:`ObjectSource` protocol."""

    objects: dict[str, bytes]
    manifest: dict
    truth: dict = field(default_factory=dict)

    def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise KeyError(f"no such object in the synthetic bucket: {key}")
        return self.objects[key]

    @property
    def total_bytes(self) -> int:
        return sum(len(b) for b in self.objects.values())


# --------------------------------------------------------------------------
def build(seed: int = 0, n: int = 20_000, sharded_channel: str = "channel_7") -> SyntheticBucket:
    """Generate the bucket. Deterministic in ``seed``: same seed, same bytes."""
    rng = np.random.default_rng(seed)
    spec = _channel_spec()
    grids = {c["id"]: _irregular_grid(rng, n) for c in spec}

    driver = _latent_driver(rng, n)
    raw = {c["id"]: _channel_signal(rng, driver, c, grids[c["id"]], n) for c in spec}

    events = _inject(rng, raw, grids, spec, n)
    values = _scale_within_groups(raw, spec)

    objects: dict[str, bytes] = {}
    channels: list[dict] = []
    for c in spec:
        cid = c["id"]
        entry = _write_channel(objects, cid, grids[cid], values[cid], c,
                               shard=(cid == sharded_channel))
        channels.append(entry)

    annotations = _write_annotations(objects, events, spec)
    manifest = _manifest(channels, annotations)
    objects["_manifest/manifest.json"] = json.dumps(manifest, indent=2).encode() + b"\n"

    return SyntheticBucket(
        objects=objects,
        manifest=manifest,
        truth={"events": events, "spec": spec, "n": n, "seed": seed,
               "sharded_channel": sharded_channel},
    )


# -- signal ----------------------------------------------------------------
def _channel_spec() -> list[dict]:
    """Eight channels over three groups, one of them categorical.

    Groups matter: ESA min-max scales within a group, so amplitude ratios survive
    inside one and not between two. Three groups make both cases testable.
    """
    return [
        {"id": "channel_1", "group": 1, "subsystem": "subsystem_A", "lag": 0,  "gain": 1.00, "target": True},
        {"id": "channel_2", "group": 1, "subsystem": "subsystem_A", "lag": 6,  "gain": 0.55, "target": True},
        {"id": "channel_3", "group": 1, "subsystem": "subsystem_A", "lag": 14, "gain": 0.20, "target": True},
        {"id": "channel_4", "group": 2, "subsystem": "subsystem_A", "lag": 3,  "gain": 4.00, "target": True},
        {"id": "channel_5", "group": 2, "subsystem": "subsystem_A", "lag": 9,  "gain": 2.20, "target": True},
        {"id": "channel_6", "group": 3, "subsystem": "subsystem_B", "lag": 0,  "gain": 1.00, "target": True},
        {"id": "channel_7", "group": 3, "subsystem": "subsystem_B", "lag": 20, "gain": 0.70, "target": True},
        {"id": "channel_8", "group": 3, "subsystem": "subsystem_B", "lag": 0,  "gain": 1.00,
         "target": False, "categorical": True},
    ]


def _irregular_grid(rng, n: int) -> np.ndarray:
    """Per-channel timestamps: nominal period with jitter, as the real data has."""
    step = np.full(n, PERIOD_S, dtype=np.float64)
    step += rng.normal(0.0, PERIOD_S * 0.05, n)
    step[0] = 0.0
    offsets = np.cumsum(np.clip(step, 1.0, None))
    return EPOCH + (offsets * 1e9).astype("timedelta64[ns]")


def _latent_driver(rng, n: int) -> np.ndarray:
    """What every channel is really watching: the thing anomalies decouple from."""
    t = np.arange(n, dtype=np.float64)
    orbit = np.sin(2 * np.pi * t / 480.0)          # short cycle
    season = 0.4 * np.sin(2 * np.pi * t / 7_000.0)  # slow cycle
    drift = 0.15 * np.sin(2 * np.pi * t / 45_000.0)
    return orbit + season + drift + rng.normal(0.0, 0.02, n)


def _channel_signal(rng, driver: np.ndarray, spec: dict, grid: np.ndarray, n: int) -> np.ndarray:
    lagged = np.roll(driver, spec["lag"])
    lagged[: spec["lag"]] = driver[0]
    return spec["gain"] * lagged + rng.normal(0.0, 0.01, n)


def _scale_within_groups(raw: dict[str, np.ndarray], spec: list[dict]) -> dict[str, np.ndarray]:
    """Min-max to [0,1] using ONE range per group, exactly as ESA did.

    Shared limits are the whole point: they are why amplitude ratios between
    related channels survive, and why a per-channel rescale would destroy them.
    """
    out: dict[str, np.ndarray] = {}
    by_group: dict[int, list[str]] = {}
    for c in spec:
        by_group.setdefault(c["group"], []).append(c["id"])
    for members in by_group.values():
        stacked = np.concatenate([raw[cid] for cid in members])
        lo, hi = float(stacked.min()), float(stacked.max())
        span = hi - lo or 1.0
        for cid in members:
            out[cid] = ((raw[cid] - lo) / span).astype(np.float32)
    return out


# -- injection -------------------------------------------------------------
def _inject(rng, raw: dict, grids: dict, spec: list[dict], n: int) -> list[dict]:
    """Write anomalies, rare events and gaps into the signal, and label them.

    Windows are allocated evenly across the middle of the series and clamped to
    their slot, so the plan fits whatever ``n`` the caller asked for. An earlier
    version hard-coded the stride and pushed the last two events past the end of
    the data, which silently produced a fixture with no unscorable segments in
    it -- the exact class of quiet, plausible wrongness this harness exists to
    refuse.
    """
    numeric = [c["id"] for c in spec if not c.get("categorical")]
    group1 = [c["id"] for c in spec if c["group"] == 1]
    pair = numeric[3:5]

    # (category, dimensionality, locality, length, channels, relative duration)
    #
    # Deliberately INTERLEAVED, not grouped by category. Laid out in plan order
    # across the timeline, a plan that lists all anomalies first puts every one
    # of them in the seed window: the last fold scores zero anomalies and the
    # headline cell never appears test-side at all. Which is a fair description
    # of a fixture that silently cannot test what it was built to test.
    A, R, G, I = "Anomaly", "Rare Event", "Communication Gap", "Invalid Segment"
    MV, UV, GL, LO, SUB, PT = ("Multivariate", "Univariate", "Global", "Local",
                               "Subsequence", "Point")
    plan = [
        (A, MV, GL, SUB, group1, 400),          # headline cell -- one per third
        (R, MV, GL, SUB, group1, 120),
        (A, MV, GL, PT, group1, 1),
        (A, UV, GL, SUB, [numeric[0]], 350),
        (R, MV, GL, SUB, group1, 120),
        (A, MV, LO, SUB, pair, 250),
        (A, UV, LO, PT, [numeric[1]], 1),
        (G, MV, GL, SUB, numeric, 150),
        (A, MV, GL, SUB, group1, 400),          # headline cell
        (R, MV, GL, SUB, group1, 120),
        (A, UV, LO, SUB, [numeric[5]], 200),
        (A, MV, GL, PT, group1, 1),
        (R, MV, GL, SUB, group1, 120),
        (A, MV, LO, SUB, pair, 250),
        (I, MV, GL, SUB, numeric, 90),
        (A, UV, GL, SUB, [numeric[0]], 350),
        (A, MV, GL, SUB, group1, 400),          # headline cell
        (R, MV, GL, SUB, group1, 120),
        (A, UV, LO, PT, [numeric[1]], 1),
        (A, MV, GL, PT, group1, 1),
        (R, MV, GL, SUB, group1, 120),
        (A, MV, LO, SUB, pair, 250),
        (A, UV, LO, SUB, [numeric[5]], 200),
    ]

    first, last = int(n * 0.15), int(n * 0.93)
    slot = max(4, (last - first) // len(plan))
    events: list[dict] = []

    for i, (category, dim, loc, length, channels, want) in enumerate(plan):
        lo = first + i * slot
        hi = min(lo + max(1, min(want, slot // 2)), n - 1)
        if hi <= lo:
            raise ValueError(f"synthetic: n={n} is too small to lay out {len(plan)} events")

        if category == "Anomaly":
            _write_anomaly(rng, raw, dim, loc, length, channels, lo, hi)
        elif category == "Rare Event":
            for cid in channels:                      # commanded, abrupt, and fine
                raw[cid][lo:hi] += 1.5
        # gaps and invalid segments are labelled, not written: there is no
        # measurement there to corrupt

        events.append({
            "id": f"id_{i + 1}", "category": category, "channels": list(channels),
            "dimensionality": dim, "locality": loc, "length": length,
            "start": lo, "end": hi,
        })
    return events


def _write_anomaly(rng, raw, dim, loc, length, channels, lo, hi) -> None:
    """Each taxonomy cell gets the failure shape its name describes."""
    if length == "Point":
        for cid in channels:
            raw[cid][lo] += 3.0 * rng.choice([-1.0, 1.0])
    elif dim == "Multivariate" and loc == "Global":
        # the coupling breaks: these channels freeze while the group moves on
        for cid in channels[1:]:
            raw[cid][lo:hi] = raw[cid][lo]
    elif dim == "Multivariate":
        for cid in channels:
            raw[cid][lo:hi] += np.linspace(0, 1.2, hi - lo)
    elif loc == "Global":
        raw[channels[0]][lo:hi] += 2.5
    else:
        raw[channels[0]][lo:hi] += np.linspace(0, 0.8, hi - lo)


# -- serialisation ---------------------------------------------------------
def _parquet(t_anon: np.ndarray, value: np.ndarray) -> bytes:
    table = pa.table({
        "timestamp": pa.array(t_anon, type=pa.timestamp("ns")),
        "value": pa.array(value),
    })
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd", version="2.6")
    return sink.getvalue()


def _digest(blob: bytes) -> tuple[str, str]:
    return hashlib.sha256(blob).hexdigest(), hashlib.md5(blob).hexdigest()


def _iso(value) -> str:
    return str(np.datetime64(value, "ns"))


def _write_channel(objects: dict, cid: str, grid: np.ndarray, values: np.ndarray,
                   spec: dict, *, shard: bool) -> dict:
    categorical = bool(spec.get("categorical"))
    payload = _categorise(values) if categorical else values
    prefix = f"esa-adb/v1/archive/{MISSION}"
    common = {
        "channel_id": cid, "subsystem": spec["subsystem"], "channel_group": spec["group"],
        "is_target": bool(spec["target"]), "is_categorical": categorical,
        "physical_unit": f"physical_unit_{spec['group']}",
        "value_dtype": "str" if categorical else "float32",
        "time_start_utc": _iso(grid[0]), "time_end_utc": _iso(grid[-1]),
    }

    if not shard:
        blob = _parquet(grid, payload)
        sha, md5 = _digest(blob)
        key = f"{prefix}/ch_{cid}.parquet"
        objects[key] = blob
        return {**common, "key": key, "shards": None, "rows": len(grid),
                "bytes": len(blob), "sha256": sha, "md5_hex": md5}

    # The shape that breaks naive loaders: parent key AND parent sha256 empty.
    cut = len(grid) // 2
    shards = []
    for i, (lo, hi) in enumerate(((0, cut), (cut, len(grid)))):
        blob = _parquet(grid[lo:hi], payload[lo:hi])
        sha, md5 = _digest(blob)
        key = f"{prefix}/ch_{cid}.part{i:03d}.parquet"
        objects[key] = blob
        shards.append({"key": key, "rows": hi - lo, "bytes": len(blob), "sha256": sha,
                       "md5_hex": md5, "time_start_utc": _iso(grid[lo]),
                       "time_end_utc": _iso(grid[hi - 1])})
    return {**common, "key": None, "shards": shards, "rows": len(grid),
            "bytes": sum(s["bytes"] for s in shards), "sha256": "", "md5_hex": ""}


def _categorise(values: np.ndarray) -> np.ndarray:
    states = np.array(["state_off", "state_idle", "state_on"], dtype=object)
    return states[np.clip((values * 3).astype(int), 0, 2)]


def _write_annotations(objects: dict, events: list[dict], spec: list[dict]) -> dict:
    grid_index = {c["id"]: i for i, c in enumerate(spec)}
    rows = {k: [] for k in ("mission", "ID", "Channel", "StartTime", "EndTime",
                            "Class", "Subclass", "Category", "Dimensionality",
                            "Locality", "Length")}
    for e in events:
        for cid in e["channels"]:
            rows["mission"].append(MISSION)
            rows["ID"].append(e["id"])
            rows["Channel"].append(cid)
            rows["StartTime"].append(EPOCH + np.timedelta64(e["start"] * PERIOD_S, "s"))
            rows["EndTime"].append(EPOCH + np.timedelta64(e["end"] * PERIOD_S, "s"))
            rows["Class"].append(f"class_{grid_index[cid] + 1}")
            rows["Subclass"].append("subclass_1")
            rows["Category"].append(e["category"])
            rows["Dimensionality"].append(e["dimensionality"])
            rows["Locality"].append(e["locality"])
            rows["Length"].append(e["length"])

    labels = pa.table({
        **{k: pa.array(v) for k, v in rows.items() if k not in ("StartTime", "EndTime")},
        "StartTime": pa.array(np.array(rows["StartTime"], dtype="datetime64[us]"),
                              type=pa.timestamp("us")),
        "EndTime": pa.array(np.array(rows["EndTime"], dtype="datetime64[us]"),
                            type=pa.timestamp("us")),
    }).select(list(rows))

    channels = pa.table({
        "mission": pa.array([MISSION] * len(spec)),
        "Channel": pa.array([c["id"] for c in spec]),
        "Subsystem": pa.array([c["subsystem"] for c in spec]),
        "Physical Unit": pa.array([f"physical_unit_{c['group']}" for c in spec]),
        "Group": pa.array([c["group"] for c in spec], type=pa.int64()),
        "Target": pa.array(["YES" if c["target"] else "NO" for c in spec]),
        "Categorical": pa.array(["YES" if c.get("categorical") else "NO" for c in spec]),
    })
    telecommands = pa.table({
        "mission": pa.array([MISSION, MISSION]),
        "Telecommand": pa.array(["telecommand_1", "telecommand_2"]),
        "Priority": pa.array([1, 2], type=pa.int64()),
    })
    events_table = pa.table({
        "mission": pa.array([MISSION]), "Event": pa.array(["event_1"]),
        "StartTime": pa.array(np.array([EPOCH], dtype="datetime64[us]"), type=pa.timestamp("us")),
        "EndTime": pa.array(np.array([EPOCH], dtype="datetime64[us]"), type=pa.timestamp("us")),
    })

    out = {}
    for name, table in (("labels", labels), ("channels", channels),
                        ("telecommands", telecommands), ("events", events_table)):
        sink = io.BytesIO()
        pq.write_table(table, sink, compression="zstd", version="2.6")
        blob = sink.getvalue()
        sha, md5 = _digest(blob)
        key = f"esa-adb/v1/annotations/{name}.parquet"
        objects[key] = blob
        out[name] = {"key": key, "rows": table.num_rows, "bytes": len(blob),
                     "sha256": sha, "md5_hex": md5,
                     "time_start_utc": None, "time_end_utc": None}
    return out


def _manifest(channels: list[dict], annotations: dict) -> dict:
    return {
        "schema_version": "1.1",
        "generated_utc": "2000-01-01T00:00:00Z",
        "datasets": {
            "esa-adb": {
                "version": "v1",
                "source_doi": "synthetic",
                "concept_doi": "synthetic",
                "source_url": "synthetic",
                "license": "synthetic fixture -- not ESA data",
                "source_files": [],
                "anonymisation": {"timestamps": "synthetic", "values": "synthetic",
                                  "names": "synthetic"},
                "missions": {
                    MISSION: {
                        "channel_count": len(channels),
                        "target_channel_count": sum(1 for c in channels if c["is_target"]),
                        "channels": channels,
                    }
                },
                "annotations": annotations,
                "reference_hints": {
                    "dominant_sampling_hz": {MISSION: 1.0 / PERIOD_S},
                    "reference_resampling": {MISSION: f"{PERIOD_S}s"},
                    "monotonic_channels_requiring_diff": {},
                    "annotation_label_enum": {"NOMINAL": 0, "ANOMALY": 1, "RARE_EVENT": 2,
                                              "GAP": 3, "INVALID": 4},
                    "resampling_method": "zero-order hold (forward fill); never linear or Fourier",
                },
            }
        },
    }
