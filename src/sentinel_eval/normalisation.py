"""Where values may be transformed. Objective.md 14.8, RESOLVED: identity.

ESA min-max normalised ESA-ADB to [0,1] **within each channel group** -- one
shared range across a group's channels -- which is precisely why the amplitude
ratios between related channels survive. The data arrives model-ready.

**Cross-group spanning is acceptable. Per-channel rescaling is not.** They are
different operations and only one destroys information:

* Across groups, each group carries its own arbitrary scale. Those offsets are
  fixed, invertible and uninformative -- different anonymised quantities in
  different anonymised units -- and a learned model absorbs them in its first
  layer. ESA's own baselines span all 176 benchmark channels and face the same
  condition, so it is the realistic operating case, not a compromise.
* Per-channel rescaling forces every channel to its own range. The fact that
  channel A normally moves ten times more than its group-mate B is then gone
  from the data, and no model can recover it, because the original scale is no
  longer present anywhere. That is the information the cross-channel claim rests
  on, and a routine z-score would erase it silently -- our method would look weak
  when we had broken the data ourselves.

So this module is a chokepoint, not a toolbox. Every value entering the harness
passes through :func:`apply`, and the only policy it accepts is ``identity``.
`tests/test_no_per_channel_scaler.py` asserts it, so the ban is executable rather
than advisory.
"""
from __future__ import annotations

import numpy as np

from .errors import NormalisationError

IDENTITY = "identity"

#: Named so a reader sees what was rejected and why, not merely that it was.
REJECTED = {
    "per_channel_zscore": "erases within-group amplitude ratios (Objective.md 5)",
    "per_channel_minmax": "erases within-group amplitude ratios (Objective.md 5)",
    "per_channel_robust": "erases within-group amplitude ratios (Objective.md 5)",
    "standard": "ambiguous; name the axis explicitly -- there is no safe default here",
}


def apply(values: np.ndarray, *, policy: str = IDENTITY) -> np.ndarray:
    """Return ``values`` unchanged. Any other policy is refused.

    The signature takes a policy it will not honour on purpose: a caller that
    wants scaling has to name it, and gets told exactly why it is refused rather
    than finding the option quietly absent and reaching for sklearn instead.
    """
    if policy != IDENTITY:
        reason = REJECTED.get(policy, "not an accepted policy")
        raise NormalisationError(
            f"normalisation policy {policy!r} refused: {reason}.\n"
            f"ESA-ADB is already min-max scaled within each channel group; the harness "
            f"applies {IDENTITY!r} and nothing else. See Objective.md 14.8."
        )
    return values
