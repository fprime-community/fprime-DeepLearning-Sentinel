"""False alarms -- the number that decides whether anyone flies this.

Objective.md 11, rule 2: *a detector that cries wolf gets ignored, and an ignored
detector is worse than none.* ESA-ADB is the only public set that lets this be
measured honestly, because it labels **716 rare nominal events** -- commanded
manoeuvres, resets, calibrations. They look abnormal and are not. A detector that
alarms on them is muted within a week in orbit.

So the headline adoption figure is deliberately *not* an overall false-positive
rate, which real spacecraft data flatters: anomalies occupy under 2% of the
timeline, so a detector that never fires scores a superb specificity. It is the
fraction of **rare nominal events** the detector alarmed on, k over n.

Two supporting figures come with it:

* the clean-nominal step rate -- how much of genuinely quiet time was flagged,
* alarms per 1,000 nominal **timesteps**, never per hour. ESA-ADB timestamps are
  anonymised mission time scaled by an undisclosed factor, so an hour here is not
  an hour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..labels import RARE_EVENT, Event
from .counts import Count
from .ranges import Range, mask_to_ranges

PER_STEPS = 1_000


@dataclass(frozen=True)
class FalseAlarmScore:
    rare_events: Count           # THE adoption number: rare events alarmed on
    nominal_steps: Count         # clean nominal timesteps flagged
    alarms_per_1000_steps: float | None
    nominal_alarms: int

    def as_dict(self) -> dict:
        return {
            "rare_event_false_alarms": self.rare_events.as_dict(),
            "nominal_step_false_alarms": self.nominal_steps.as_dict(),
            "alarms_per_1000_nominal_timesteps": self.alarms_per_1000_steps,
            "nominal_alarm_count": self.nominal_alarms,
        }


def score(
    events: list[Event],
    spans: dict[str, Range],
    predicted: np.ndarray,
    anomaly: np.ndarray,
    rare: np.ndarray,
    scorable: np.ndarray,
) -> FalseAlarmScore:
    """Count the alarms that should never have been raised."""
    fired = np.asarray(predicted, dtype=bool) & np.asarray(scorable, dtype=bool)

    rare_events = [e for e in events if e.category == RARE_EVENT and e.event_id in spans]
    tripped = 0
    for event in rare_events:
        lo, hi = spans[event.event_id]
        if fired[lo:hi].any():
            tripped += 1

    # Clean nominal: measured, and annotated as neither anomalous nor rare.
    clean = np.asarray(scorable, dtype=bool) & ~np.asarray(anomaly, dtype=bool) \
        & ~np.asarray(rare, dtype=bool)
    clean_steps = int(clean.sum())
    flagged = int((fired & clean).sum())

    # An alarm is a contiguous run: one claim an operator receives, not N.
    alarms = sum(1 for lo, hi in mask_to_ranges(fired) if clean[lo:hi].any())
    rate = None if clean_steps == 0 else alarms * PER_STEPS / clean_steps

    return FalseAlarmScore(
        rare_events=Count(tripped, len(rare_events)),
        nominal_steps=Count(flagged, clean_steps),
        alarms_per_1000_steps=rate,
        nominal_alarms=alarms,
    )
