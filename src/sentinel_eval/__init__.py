"""The Sentinel evaluation harness -- the referee, built before the players.

It loads ESA-ADB labels, streams telemetry from R2, and scores *any* candidate
detector on the numbers that decide this project (Objective.md, section 13):

  * recall over point anomalies
  * recall over contextual anomalies -- the class that matters
  * false-alarm rate, including on rare nominal events

Nothing in here knows about LSTMs, GRUs or TCNs. A detector is anything with
``fit`` and ``score``; the harness treats them all identically, so that the
architecture selection gate is a table of numbers rather than an argument.

Two rules hold everywhere in this package:

1. **Every count is reported as k/n.** A recall of 0.36 over eleven events can
   take twelve values, and the difference between two detectors can be a single
   event. Denominators travel with their numerators, always.
2. **No dataset persists on this machine.** Telemetry streams into memory and
   is dropped when the process exits. Outputs are different: scorecards and
   trained weights live under `runs/`, gitignored. See docs/HARNESS.md,
   Rule 1 stated precisely.
"""
from __future__ import annotations

__version__ = "0.1.0"

HARNESS_VERSION = __version__
