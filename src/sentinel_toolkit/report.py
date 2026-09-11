"""The label-free calibration report: the deliverable, not a side effect.

An operator with **no labelled anomalies** has to be able to read this and know
what the detector will do. So every figure in it comes from their own healthy
telemetry, with no label anywhere in it, and the two halves are kept apart on
the page as well as in the arithmetic.

**The rate is reported and never targeted** (`docs/HARNESS.md` 1). Nobody
adjusts anything on the strength of it; an absurd value means the calibration is
broken and should be investigated rather than turned down. The curve beside it
is a measurement of sensitivity, swept at every value and selected at none.
"""
from __future__ import annotations

from .calibrate import Calibration

RULE = "=" * 76


def render(calibration: Calibration, *, channels: int, train_steps: int,
           window: int, hidden, n_predictions: int, model_bytes: int,
           param_version: int, tier: int) -> str:
    c = calibration
    lines = [
        RULE,
        "  PRE-LAUNCH CALIBRATION REPORT",
        "  No label was consulted anywhere in this. Every figure is from your",
        "  own healthy telemetry.",
        RULE,
        "",
        "  THE MODEL",
        f"    forecaster            GRU, {len(tuple(hidden))} layers of "
        f"{tuple(hidden)[0]}, {window}-step window, {n_predictions}-step horizon",
        f"    channels              {channels}",
        f"    trained on            {train_steps:,} timesteps",
        f"    model.bin             {model_bytes:,} bytes, "
        f"param_version {param_version}, tier {tier}",
        "",
        "  THE OPERATING POINT, AND IT IS DERIVED RATHER THAN CHOSEN",
        f"    statistic             max(z_residual, z_derivative), each",
        f"                          standardised over its own trailing "
        f"{c.span:,} samples",
        f"    cut                   {c.cut:.6f}  "
        f"(the {c.quantile:g} quantile of that statistic on your nominal data)",
        f"    provenance            {c.provenance()}",
        "",
        "  WHAT IT DOES ON HEALTHY DATA YOU DID NOT CALIBRATE ON",
        f"    calibration half      {c.fit_alarms:,} / {c.fit_steps:,}"
        f"   = {100 * c.fit_rate:.4f}%",
        f"    held-out half         {c.holdout_alarms:,} / {c.holdout_steps:,}"
        f"   = {100 * c.holdout_rate:.4f}%   <- the pre-launch sanity rate",
        f"    ratio                 {c.ratio:.2f}x",
        "",
        "    (!) REPORTED, NEVER TARGETED. Do not adjust anything to move this",
        "        number. It is an instrument check: a sane value says the",
        "        calibration is sound, an absurd one says it is broken and should",
        "        be investigated rather than turned down.",
        "",
        "  SENSITIVITY, SWEPT AT EVERY VALUE AND SELECTED AT NONE",
        "    quantile        cut     calibration half        held-out half",
    ]
    for q, cut, fit_alarms, holdout_alarms in c.curve:
        mark = "  <- the cut" if q == c.quantile else ""
        lines.append(
            f"    {q:<9g} {cut:9.4f}   {fit_alarms:6,} / {c.fit_steps:<9,}"
            f"  {holdout_alarms:6,} / {c.holdout_steps:<9,}{mark}")
    lines += [
        "",
        "    A swept parameter reported at every value and selected at none is a",
        "    measurement of sensitivity. No operating point is recommended from",
        "    this curve; the cut above was derived from the noise floor.",
        "",
        "  WHAT THIS REPORT DOES NOT TELL YOU",
        "    - It does not say the detector will catch your anomalies. It has",
        "      never seen one of yours. It says what it does when nothing is",
        "      wrong.",
        "    - It makes no early-warning claim. Nothing here is a lead time.",
        "    - It is not a recall figure and cannot be turned into one without",
        "      labelled events, which a deploying mission does not have.",
        RULE,
    ]
    return "\n".join(lines)
