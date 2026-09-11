"""The ground toolkit: a mission's healthy telemetry in, a `model.bin` out.

Pre-registered at `docs/MODELS.md` 40 before any of this existed. **It is a
composition root over code that already exists and is already held to a
reference by committed vectors** -- it trains nothing of its own, writes no
format of its own, and computes no statistic of its own. What it adds is the
one command, the refusals, and the label-free calibration report.

Three things it does NOT do, each registered rather than discovered:

* **It takes no target alarm rate.** `docs/HARNESS.md` 1 struck that framing:
  a threshold is a noise floor measured from nominal residuals, never a dial
  chosen from results. The cut is derived; the rate curve is reported around
  it and selected at no value.
* **It states no early-warning claim** and produces no figure that could be
  read as one.
* **It does not choose the trailing span.** That is a flight compile-time
  constant, and section 40.3a records why the toolkit may not pick it.
"""
from .errors import ToolkitError
from .limits import FLIGHT_LIMITS

__all__ = ["FLIGHT_LIMITS", "ToolkitError"]
