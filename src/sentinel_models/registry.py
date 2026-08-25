"""Name -> detector. Resolved at the CLI, never inside the harness.

`sentinel_eval` must not import this module: it is the composition root's job to
turn ``--detector gru`` into an object. Keeping the direction one-way is what
lets work items 4, 5 and 6 add architectures without touching the referee.
"""
from __future__ import annotations

from sentinel_eval.detector import Detector

from . import baselines

#: Names are lowercase and hyphenated. Items 4-6 add `lstm-telemanom`, `gru`, `tcn`.
BUILDERS = {
    "mavg": baselines.MovingAverage,
    "rstd": baselines.RollingStd,
    "quiet": baselines.AlwaysQuiet,
    "random": baselines.RandomScore,
}


def available() -> list[str]:
    return sorted(BUILDERS)


def build(name: str, **params) -> Detector:
    try:
        return BUILDERS[name](**params)
    except KeyError:
        raise SystemExit(
            f"unknown detector {name!r}; available: {', '.join(available())}"
        ) from None
