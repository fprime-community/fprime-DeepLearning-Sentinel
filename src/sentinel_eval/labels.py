"""ESA-ADB annotations, turned into what scoring actually needs.

The source table is one row per *(event, channel)* segment. Scoring needs
**events**, because "did we catch it?" is a question about an event even when it
shows up on eleven channels at once. Mission1's anomalies touch a median of three
channels and one touches fifty-eight.

Four categories appear, and they are **not** three positives and a negative:

===================  ======================================================
``Anomaly``          the positive class. 157 dataset-wide, 51 on m1-ss5
``Rare Event``       manoeuvres, resets, calibrations. **Nominal.** Firing here
                     is a false alarm, and it is the false alarm that decides
                     adoption (Objective.md 6.2). 716 across our ingest
``Communication Gap``
``Invalid Segment``  neither positive nor negative. There is no measurement to
                     be right or wrong about, so both are masked out of every
                     metric rather than counted as nominal -- 401 gaps and 156
                     invalid segments of free specificity otherwise
===================  ======================================================

**Contextual anomalies.** Objective.md 2.4 defines one as an event where every
channel is individually legal but the combination or trajectory is wrong.
ESA-ADB's nearest operational encoding is ``Dimensionality == "Multivariate"``:
the event manifests across more than one channel at once. That equivalence is an
interpretation, stated here once so every number derived from it is traceable to
it, and :data:`HEADLINE_CELL` names the single taxonomy cell the thesis rests on.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

ANOMALY = "Anomaly"
RARE_EVENT = "Rare Event"
COMMUNICATION_GAP = "Communication Gap"
INVALID_SEGMENT = "Invalid Segment"

#: Categories carrying no ground truth, so no metric may be computed over them.
UNSCORABLE = (COMMUNICATION_GAP, INVALID_SEGMENT)

#: The taxonomy cell the project's central claim lives in.
HEADLINE_CELL = "Multivariate/Global/Subsequence"


@dataclass(frozen=True)
class Segment:
    channel: str
    start: np.datetime64
    end: np.datetime64


@dataclass(frozen=True)
class Event:
    """One annotated event, unioned over every channel it was seen on."""

    mission: str
    event_id: str
    category: str
    start: np.datetime64
    end: np.datetime64
    dimensionality: str
    locality: str
    length: str
    esa_class: str
    subclass: str
    segments: tuple[Segment, ...]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted({s.channel for s in self.segments}))

    @property
    def cell(self) -> str:
        """`Dimensionality/Locality/Length` -- the taxonomy key."""
        return f"{self.dimensionality}/{self.locality}/{self.length}"

    @property
    def contextual(self) -> bool:
        """Multivariate: legal alone, wrong together. See the module docstring."""
        return self.dimensionality == "Multivariate"

    @property
    def is_point(self) -> bool:
        return self.length == "Point"

    def segments_on(self, channels) -> list[Segment]:
        keep = set(channels)
        return [s for s in self.segments if s.channel in keep]

    def __str__(self) -> str:
        return f"{self.mission}/{self.event_id} [{self.category}] {self.cell}"


@dataclass(frozen=True)
class Truth:
    """Per-timestep ground truth on one grid, for one channel selection."""

    anomaly: np.ndarray        # bool[T]   the positive class
    rare_event: np.ndarray     # bool[T]   nominal; alarms here are false alarms
    unscorable: np.ndarray     # bool[T]   gaps and invalid segments
    per_channel: np.ndarray    # bool[T,C] anomaly, attributed to channels
    events: tuple[Event, ...]  # every event overlapping the grid, any category
    spans: dict                # event_id -> (lo, hi) timesteps on this grid

    @property
    def scorable(self) -> np.ndarray:
        return ~self.unscorable

    def of(self, category: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.category == category)

    @property
    def anomalies(self) -> tuple[Event, ...]:
        return self.of(ANOMALY)

    @property
    def rare_events(self) -> tuple[Event, ...]:
        return self.of(RARE_EVENT)


class LabelSet:
    """Every annotation, queryable by mission, channel selection and window."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    @classmethod
    def from_table(cls, table) -> "LabelSet":
        """Build from the pyarrow Table of `annotations/labels.parquet`."""
        return cls(table.to_pylist())

    @cached_property
    def _events(self) -> list[Event]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in self._rows:
            grouped.setdefault((row["mission"], row["ID"]), []).append(row)

        events = []
        for (mission, event_id), rows in grouped.items():
            head = rows[0]
            segments = tuple(
                Segment(str(r["Channel"]), np.datetime64(r["StartTime"], "ns"),
                        np.datetime64(r["EndTime"], "ns"))
                for r in rows
            )
            events.append(Event(
                mission=str(mission),
                event_id=str(event_id),
                category=str(head["Category"]),
                start=min(s.start for s in segments),
                end=max(s.end for s in segments),
                dimensionality=_text(head.get("Dimensionality")),
                locality=_text(head.get("Locality")),
                length=_text(head.get("Length")),
                esa_class=_text(head.get("Class")),
                subclass=_text(head.get("Subclass")),
                segments=segments,
            ))
        return sorted(events, key=lambda e: (e.mission, e.start, e.event_id))

    def events(self, mission, *, channels=None, category=None, start=None, end=None) -> list[Event]:
        """Events for a mission, optionally restricted to channels and a window.

        An event is kept when *any* of its segments lies on a selected channel
        and overlaps the window. Its own ``start``/``end`` stay the union over
        all its segments, so its identity does not shift with the selection;
        :meth:`Event.segments_on` gives the part actually visible.
        """
        keep = set(channels) if channels is not None else None
        out = []
        for event in self._events:
            if event.mission != mission or (category is not None and event.category != category):
                continue
            segments = event.segments if keep is None else event.segments_on(keep)
            if not segments:
                continue
            if start is not None and max(s.end for s in segments) < np.datetime64(start, "ns"):
                continue
            if end is not None and min(s.start for s in segments) >= np.datetime64(end, "ns"):
                continue
            out.append(event)
        return out

    # -- rasterisation -----------------------------------------------------
    def truth(self, grid, mission: str, channels: list[str]) -> Truth:
        """Rasterise the annotations onto a :class:`~sentinel_eval.grid.Grid`."""
        n, c = len(grid), len(channels)
        anomaly = np.zeros(n, dtype=bool)
        rare = np.zeros(n, dtype=bool)
        unscorable = np.zeros(n, dtype=bool)
        per_channel = np.zeros((n, c), dtype=bool)
        column = {name: i for i, name in enumerate(channels)}
        spans: dict[str, tuple[int, int]] = {}

        events = self.events(mission, channels=channels, start=grid.start, end=grid.end)
        for event in events:
            target = (anomaly if event.category == ANOMALY
                      else rare if event.category == RARE_EVENT
                      else unscorable)
            lo_all, hi_all = n, 0
            for segment in event.segments_on(channels):
                lo, hi = grid.slice_of(segment.start, segment.end)
                if hi <= lo:
                    continue
                target[lo:hi] = True
                if event.category == ANOMALY:
                    per_channel[lo:hi, column[segment.channel]] = True
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
            if hi_all > lo_all:
                spans[event.event_id] = (lo_all, hi_all)

        # A timestep annotated both anomalous and inside a gap contradicts
        # itself in the source. Trust the anomaly: an operator wrote it.
        unscorable &= ~anomaly
        return Truth(anomaly, rare, unscorable, per_channel, tuple(events), spans)

    # -- reporting ---------------------------------------------------------
    def census(self, events: list[Event]) -> dict:
        """The k/n denominators a scorecard must carry. Counts, never rates."""
        anomalies = [e for e in events if e.category == ANOMALY]
        by_cell: dict[str, int] = {}
        for event in anomalies:
            by_cell[event.cell] = by_cell.get(event.cell, 0) + 1
        return {
            "events": len(events),
            "anomalies": len(anomalies),
            "contextual": sum(1 for e in anomalies if e.contextual),
            "headline_cell": by_cell.get(HEADLINE_CELL, 0),
            "point": sum(1 for e in anomalies if e.is_point),
            "rare_events": sum(1 for e in events if e.category == RARE_EVENT),
            "unscorable": sum(1 for e in events if e.category in UNSCORABLE),
            "by_cell": dict(sorted(by_cell.items(), key=lambda kv: (-kv[1], kv[0]))),
        }


def _text(value) -> str:
    return "Unknown" if value is None else str(value)
