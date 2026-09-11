"""The label-free calibration: the cut is derived, the curve is reported.

**(!) No target alarm rate is an input to anything here**, and that is a
standing principle rather than a preference. `docs/HARNESS.md` 1:

    calibration   what does this spacecraft's normal noise look like     DO THIS
    fitting       what threshold makes my numbers look good              NEVER

So the cut is the `(1 - q)` quantile of the mission's own **pooled nominal**
fused statistic -- a noise-floor determination on its own data -- and the rate
that cut produces on **held-out** healthy data is *reported and never targeted*.
The rate-against-threshold curve is reported around it as a measurement of
sensitivity, at every value and selected at none, which is the treatment
`scripts/oscfar_curve.py:12-19` gives the admission rate.

The pooled quantile rather than a bisection is `scripts/decision_layer_arms.py`
:117-125's reason: for step-shaped scores a bisection lands on a discontinuity
and reports whatever rate sits there. A quantile asks the question directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import CalibrationError

#: (!) The one constant this module has to defend, and `docs/MODELS.md` 40.5
#: registers it as owed rather than settled.
#:
#: A quantile is a legal dimensionless form under D55 -- "a fraction, a quantile,
#: or a multiple of the series' own dispersion" -- so the *form* is right. The
#: *value* is inherited: 0.999 is what `src/sentinel_eval/detector.py:36` uses
#: for D25's version-1 cut, which is a percentile of a smoothed error **in data
#: units**. The version-2 cut is a **z-score** (`docs/MODEL_FILE.md` 6.2), and
#: nothing establishes that a quantile transfers between the two. T6 is the test:
#: it measures whether the value chosen on the fit half survives on the held-out
#: half, which is the only evidence available without labels.
DEFAULT_QUANTILE = 0.999

#: Where the reported curve is swept. Three decades around the cut, which is the
#: shape `scripts/oscfar_curve.py:45` reports for the same reason: so a reader
#: sees the response rather than a single point being defended.
CURVE_QUANTILES = (0.99, 0.995, 0.999, 0.9995, 0.9999)


@dataclass(frozen=True)
class Calibration:
    """What was measured, on what, and everything needed to repeat it."""

    cut: float
    quantile: float
    span: int
    fit_steps: int
    holdout_steps: int
    fit_alarms: int
    holdout_alarms: int
    curve: tuple[tuple[float, float, int, int], ...] = field(default=())

    @property
    def fit_rate(self) -> float:
        return self.fit_alarms / self.fit_steps if self.fit_steps else float("nan")

    @property
    def holdout_rate(self) -> float:
        return (self.holdout_alarms / self.holdout_steps
                if self.holdout_steps else float("nan"))

    @property
    def ratio(self) -> float:
        """Held-out rate over fit rate. T6's band is stated on this number."""
        if self.fit_rate == 0.0:
            return float("inf") if self.holdout_rate > 0.0 else 1.0
        return self.holdout_rate / self.fit_rate

    def provenance(self) -> str:
        """<= 63 ASCII bytes, for the model file's own `provenance` field.

        D29: a threshold is a parameter with a provenance, not a constant. The
        field records which window and which procedure produced the number.
        """
        return f"q{self.quantile:g} pooled nominal, span {self.span}, n {self.fit_steps}"


def derive_cut(scores: np.ndarray, quantile: float = DEFAULT_QUANTILE) -> float:
    """The `(1 - q)` quantile of the pooled nominal scores. F64, as the file is.

    F64 deliberately: `docs/MODEL_FILE.md` 6 stores the threshold as F64 because
    rounding the cut could flip a crossing at the boundary, and the comparison
    is `>=` rather than `>`.
    """
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise CalibrationError(
            "no finite score in the calibration window; the statistic did not "
            "settle, which means the forecaster or the telemetry is wrong rather "
            "than the threshold")
    if not 0.0 < quantile < 1.0:
        raise CalibrationError(f"quantile {quantile} is not in (0, 1)")
    return float(np.quantile(finite, quantile))


def alarms_at(scores: np.ndarray, cut: float) -> int:
    """Steps at or above the cut. `>=`, matching the flight comparison."""
    finite = np.asarray(scores, dtype=np.float64)
    return int(np.count_nonzero(finite[np.isfinite(finite)] >= cut))


def calibrate(fit_scores: np.ndarray, holdout_scores: np.ndarray, span: int,
              quantile: float = DEFAULT_QUANTILE) -> Calibration:
    """Derive the cut on the fit half; measure what it does on the held-out half.

    Never the same data twice: the half the cut was derived on cannot also be
    the half that says what the cut does.
    """
    cut = derive_cut(fit_scores, quantile)
    curve = []
    for q in sorted(set(CURVE_QUANTILES) | {quantile}):
        c = derive_cut(fit_scores, q)
        curve.append((q, c, alarms_at(fit_scores, c), alarms_at(holdout_scores, c)))
    return Calibration(
        cut=cut, quantile=quantile, span=span,
        fit_steps=int(np.count_nonzero(np.isfinite(fit_scores))),
        holdout_steps=int(np.count_nonzero(np.isfinite(holdout_scores))),
        fit_alarms=alarms_at(fit_scores, cut),
        holdout_alarms=alarms_at(holdout_scores, cut),
        curve=tuple(curve),
    )
