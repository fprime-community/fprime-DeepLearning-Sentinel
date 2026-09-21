"""The manifest, resolved into typed objects. The only place keys come from.

docs/DATA.md: *the manifest is the single source of truth; read it first, resolve
every key from it, and never list the bucket.* This module is that rule made into
an object, so no other module has to remember it.

**Nothing here returns a DataFrame.** While gathering the evidence for the
channel-set decision, `frame.sub` and `frame.cat` silently resolved to pandas'
``.sub()`` method and ``.cat`` accessor rather than to columns of those names,
and every mission filter came back empty -- a confident, wrong, all-zero table
that looked exactly like a real one. Attribute access on a DataFrame is a
silent-corruption surface. :class:`Channel` is a frozen dataclass, so the same
mistake is an ``AttributeError``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .errors import TaskError

DATASET = "esa-adb"


@dataclass(frozen=True)
class StoredObject:
    """One R2 object, with the checksum that proves it arrived intact."""

    key: str
    sha256: str
    nbytes: int
    rows: int
    time_start: np.datetime64 | None = None
    time_end: np.datetime64 | None = None


@dataclass(frozen=True)
class Channel:
    """One telemetry channel and every fact about it the harness needs.

    ``objects`` is always populated, sharded or not, so callers never branch on
    ``key is None``. Sharded entries also carry an *empty* parent ``sha256``
    (``prepare.py`` fills it only for single-object channels), which is why
    integrity is verified per object here rather than per channel.
    """

    mission: str
    channel_id: str
    objects: tuple[StoredObject, ...]
    rows: int
    nbytes: int
    subsystem: str
    group: int | None
    is_target: bool
    is_categorical: bool
    value_dtype: str
    time_start: np.datetime64
    time_end: np.datetime64

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(o.key for o in self.objects)

    @property
    def sharded(self) -> bool:
        return len(self.objects) > 1

    @property
    def numeric(self) -> bool:
        """Categorical channels are stored as strings; arithmetic is invalid."""
        return not self.is_categorical and not self.value_dtype.startswith("str")

    def __str__(self) -> str:
        return f"{self.mission}/{self.channel_id}"


def _dt(value) -> np.datetime64 | None:
    return None if value is None else np.datetime64(value, "ns")


class Catalog:
    """The manifest, resolved. Construct with :meth:`load`."""

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        try:
            self.dataset = manifest["datasets"][DATASET]
        except KeyError:
            raise TaskError(f"manifest carries no {DATASET!r} dataset") from None

    #: (!) THE KEY IS SPELLED HERE RATHER THAN IMPORTED FROM `sentinel_data`.
    #: Importing it dragged the whole ingest package into the import graph of
    #: every reader, and from there `sentinel_data.r2`'s boto3. The value is one
    #: string and the ingest side owns it, so
    #: `tests/test_toolkit_closure_is_offline.py` asserts the two spellings
    #: still agree -- a duplicated constant with a guard, rather than a
    #: dependency. D80.
    MANIFEST_KEY = "_manifest/manifest.json"

    @classmethod
    def load(cls, source) -> "Catalog":
        """Read `_manifest/manifest.json`. One Class B against R2."""
        return cls(json.loads(source.get(cls.MANIFEST_KEY)))

    # -- provenance --------------------------------------------------------
    def provenance(self) -> dict:
        """Pinned into every scorecard, so a result names the data it scored."""
        return {
            "dataset": DATASET,
            "version": self.dataset["version"],
            "manifest_schema_version": self.manifest["schema_version"],
            "manifest_generated_utc": self.manifest["generated_utc"],
            "source_doi": self.dataset["source_doi"],
            "license": self.dataset["license"],
        }

    @property
    def missions(self) -> list[str]:
        return sorted(self.dataset["missions"])

    # -- channels ----------------------------------------------------------
    @cached_property
    def _channels(self) -> dict[str, dict[str, Channel]]:
        out: dict[str, dict[str, Channel]] = {}
        for mission, entry in self.dataset["missions"].items():
            by_id: dict[str, Channel] = {}
            for raw in entry["channels"]:
                by_id[raw["channel_id"]] = Channel(
                    mission=mission,
                    channel_id=raw["channel_id"],
                    objects=_objects_of(raw),
                    rows=raw["rows"],
                    nbytes=raw["bytes"],
                    subsystem=raw["subsystem"],
                    group=raw["channel_group"],
                    is_target=bool(raw["is_target"]),
                    is_categorical=bool(raw["is_categorical"]),
                    value_dtype=raw["value_dtype"],
                    time_start=_dt(raw["time_start_utc"]),
                    time_end=_dt(raw["time_end_utc"]),
                )
            out[mission] = by_id
        return out

    def channel(self, mission: str, channel_id: str) -> Channel:
        try:
            return self._channels[mission][channel_id]
        except KeyError:
            raise TaskError(f"no such channel in the manifest: {mission}/{channel_id}") from None

    def channels(
        self,
        mission: str,
        *,
        subsystem: str | None = None,
        groups: list[int] | None = None,
        channel_ids: list[str] | None = None,
        target_only: bool = False,
        numeric_only: bool = False,
    ) -> list[Channel]:
        """Select channels by manifest attribute. Never by naming convention."""
        if mission not in self._channels:
            raise TaskError(f"no such mission in the manifest: {mission}")
        if channel_ids is not None:
            return [self.channel(mission, cid) for cid in channel_ids]
        out = list(self._channels[mission].values())
        if subsystem is not None:
            out = [c for c in out if c.subsystem == subsystem]
        if groups is not None:
            out = [c for c in out if c.group in set(groups)]
        if target_only:
            out = [c for c in out if c.is_target]
        if numeric_only:
            out = [c for c in out if c.numeric]
        return sorted(out, key=lambda c: (c.group or 0, _natural(c.channel_id)))

    def subsystems(self, mission: str) -> list[str]:
        return sorted({c.subsystem for c in self._channels[mission].values() if c.subsystem})

    # -- annotations -------------------------------------------------------
    def annotation(self, name: str) -> StoredObject:
        try:
            raw = self.dataset["annotations"][name]
        except KeyError:
            raise TaskError(
                f"no annotation table {name!r}; have {sorted(self.dataset['annotations'])}"
            ) from None
        return StoredObject(raw["key"], raw["sha256"], raw["bytes"], raw["rows"])

    def telecommand_series(self, mission: str) -> StoredObject:
        """The mission's telecommand executions, as one object.

        telemanom's LSTM takes telemetry **and encoded command information**
        (docs/DECISIONS.md D6); this is where the second half comes from. Mission1
        holds 1,594,722 executions of 681 distinct commands in a single 7.7 MB
        object spanning the whole timeline, so supplying them costs one Class B.

        Separate from :meth:`annotation` because this is a time series in the
        archive, not a description table in the annotations. The 821-row
        description table -- which carries the priority grading -- is
        ``annotation("telecommands")``, and the two are easy to confuse because
        they share a name.
        """
        try:
            raw = self.dataset["missions"][mission]["telecommand_series"]
        except KeyError:
            raise TaskError(
                f"{mission} has no telecommand series in the manifest; "
                f"mission3 carries none"
            ) from None
        return StoredObject(raw["key"], raw["sha256"], raw["bytes"], raw["rows"])

    # -- ESA's magic numbers, which exist in no CSV ------------------------
    @property
    def hints(self) -> dict:
        return self.dataset["reference_hints"]

    def resample_period(self, mission: str) -> tuple[np.timedelta64, str]:
        """ESA's prescribed period, or the derived one, and which of the two.

        They are not equally authoritative and the caller is told which it got:
        a prescribed rule comes from ESA's own scripts, a derived one is our
        arithmetic on the dominant sampling rate.
        """
        rule = self.hints.get("reference_resampling", {}).get(mission)
        if rule:
            seconds = int(str(rule).rstrip("s"))
            return np.timedelta64(seconds, "s"), "prescribed"
        hz = self.hints.get("dominant_sampling_hz", {}).get(mission)
        if not hz:
            raise TaskError(f"no resampling rule and no sampling rate for {mission}")
        return np.timedelta64(round(1.0 / hz), "s"), "derived"

    def monotonic_channels(self, mission: str) -> list[str]:
        """Channels ESA differences before use, resolved to channel ids.

        The manifest records these as bare integers, exactly as ESA's scripts do.
        They are read as channel numbers -- ``4`` means ``channel_4`` -- and every
        id is checked against the manifest, so a wrong reading fails here rather
        than silently differencing some other series.
        """
        numbers = self.hints.get("monotonic_channels_requiring_diff", {}).get(mission, [])
        ids = [f"channel_{int(n)}" for n in numbers]
        unknown = [i for i in ids if i not in self._channels.get(mission, {})]
        if unknown:
            raise TaskError(
                f"{mission}: reference_hints names monotonic channels {unknown} that are not "
                "in the manifest -- the integer-to-channel-id reading is wrong"
            )
        return ids

    @property
    def resampling_method(self) -> str:
        return self.hints["resampling_method"]


def _objects_of(raw: dict) -> tuple[StoredObject, ...]:
    """Normalise sharded and unsharded entries to the same shape.

    A sharded channel carries ``key: null`` *and* ``sha256: ''``; its real keys
    and checksums live one level down. Handling that here means no reader ever
    sees the difference.
    """
    if raw.get("shards"):
        shards = sorted(raw["shards"], key=lambda s: s["time_start_utc"] or "")
        return tuple(
            StoredObject(s["key"], s["sha256"], s["bytes"], s["rows"],
                         _dt(s.get("time_start_utc")), _dt(s.get("time_end_utc")))
            for s in shards
        )
    if not raw.get("key"):
        raise TaskError(f"{raw.get('channel_id')}: manifest entry has neither key nor shards")
    return (
        StoredObject(raw["key"], raw["sha256"], raw["bytes"], raw["rows"],
                     _dt(raw.get("time_start_utc")), _dt(raw.get("time_end_utc"))),
    )


def _natural(name: str) -> tuple[str, int]:
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (name, 0)
