"""How long before an event started did the detector first say something.

Every other metric in this package answers *did you catch it*. None answers *how
early*, and early warning is the project's headline claim -- Objective.md 4:
"BattVoltage predicted to cross RED_LO in ~4h 12m". Phase 1 has measured
detection and never once measured lead.

It is also an architecture discriminator. Forty-six anomalies cannot separate an
LSTM from a GRU from a TCN on recall: the denominators are too small and one
event moves the rate by 0.022. If the three catch the same events at materially
different lead times, that is a difference the gate can actually see.

**Timesteps, never hours.** ESA-ADB timestamps are anonymised mission time,
scaled by an undisclosed factor greater than one (docs/HARNESS.md section 4).
Any figure here expressed in wall-clock units would be wrong, so none is
available to express.

**Negative lead is a real answer.** A detector that fires three hundred steps
*into* an event caught it late, and clipping that to zero would turn a
late detection into a perfect one. Negatives are reported.

### The definition, stated once

Lead time for a caught event is ``event_start - alarm_start``, where
``alarm_start`` is the beginning of the **earliest predicted alarm range that
overlaps the event**. Positive means the alarm was already raised when the event
began.

The honest weakness: a very wide alarm range that happens to overlap inflates
lead, because its start may have nothing to do with the event. Nothing here
guesses at attribution to correct for it -- instead :class:`LeadTimeScore`
carries the width of the alarm that was credited, so a reader can see when a
lead figure rests on a long-running alarm rather than on an early one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..labels import ANOMALY, Event
from .ranges import Range, mask_to_ranges, overlaps


@dataclass(frozen=True)
class LeadTimeScore:
    """Lead time over the events a detector caught, in timesteps."""

    #: event_id -> (lead_timesteps, width_of_the_credited_alarm)
    per_event: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: taxonomy cell -> list of lead times
    by_cell: dict[str, list[int]] = field(default_factory=dict)

    @property
    def leads(self) -> list[int]:
        return [lead for lead, _ in self.per_event.values()]

    @staticmethod
    def _quantiles(values: list[int]) -> dict:
        if not values:
            return {"n": 0, "median": None, "p25": None, "p75": None,
                    "min": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": len(values),
            "median": float(np.median(array)),
            "p25": float(np.quantile(array, 0.25)),
            "p75": float(np.quantile(array, 0.75)),
            "min": int(array.min()),
            "max": int(array.max()),
            "negative": int((array < 0).sum()),
        }

    def summary(self) -> dict:
        return self._quantiles(self.leads)

    def as_dict(self) -> dict:
        return {
            "units": "timesteps",
            "definition": "event_start - start of the earliest alarm range "
                          "overlapping the event; negative means detected late",
            "overall": self.summary(),
            "by_cell": {cell: self._quantiles(v) for cell, v in sorted(self.by_cell.items())},
            "credited_alarm_width": self._quantiles(
                [width for _, width in self.per_event.values()]
            ),
            "per_event": {k: {"lead_timesteps": lead, "alarm_width_timesteps": width}
                          for k, (lead, width) in sorted(self.per_event.items())},
        }

    def render(self) -> list[str]:
        overall = self.summary()
        if not overall["n"]:
            return ["    LEAD TIME  no caught event to measure"]
        out = [f"    LEAD TIME timesteps, never hours -- anonymised mission time",
               f"              median {overall['median']:>10,.0f}   "
               f"p25 {overall['p25']:>10,.0f}   p75 {overall['p75']:>10,.0f}   "
               f"over {overall['n']} caught",
               f"              range  {overall['min']:>10,} .. {overall['max']:,}"
               f"   detected late: {overall['negative']}/{overall['n']}"]
        for cell, values in sorted(self.by_cell.items()):
            q = self._quantiles(values)
            out.append(f"        {cell:<38} median {q['median']:>9,.0f}  n={q['n']}")
        return out


def score(events: list[Event], spans: dict[str, Range], predicted: np.ndarray,
          scorable: np.ndarray) -> LeadTimeScore:
    """Lead time for every anomaly this prediction caught. Uncaught events absent.

    An event nobody detected has no lead time -- not a zero, and not a negative.
    Recall already reports that it was missed, and inventing a number here would
    let a detector that catches nothing report an excellent median.
    """
    fired = np.asarray(predicted, dtype=bool) & np.asarray(scorable, dtype=bool)
    alarms = mask_to_ranges(fired)
    if not alarms:
        return LeadTimeScore()

    per_event: dict[str, tuple[int, int]] = {}
    by_cell: dict[str, list[int]] = {}
    for event in events:
        if event.category != ANOMALY or event.event_id not in spans:
            continue
        span = spans[event.event_id]
        overlapping = [a for a in alarms if overlaps(a, span)]
        if not overlapping:
            continue                      # missed: recall's business, not ours
        earliest = min(overlapping, key=lambda a: a[0])
        lead = int(span[0] - earliest[0])
        per_event[event.event_id] = (lead, int(earliest[1] - earliest[0]))
        by_cell.setdefault(event.cell, []).append(lead)
    return LeadTimeScore(per_event=per_event, by_cell=by_cell)


def pool(scores: list[LeadTimeScore]) -> LeadTimeScore:
    """Pool folds. Each event is scored in exactly one fold, so this is a union."""
    per_event: dict[str, tuple[int, int]] = {}
    by_cell: dict[str, list[int]] = {}
    for item in scores:
        per_event.update(item.per_event)
        for cell, values in item.by_cell.items():
            by_cell.setdefault(cell, []).extend(values)
    return LeadTimeScore(per_event=per_event, by_cell=by_cell)
