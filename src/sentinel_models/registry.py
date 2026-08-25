"""Name -> detector. Resolved at the CLI, never inside the harness.

`sentinel_eval` must not import this module: it is the composition root's job to
turn ``--detector gru`` into an object. Keeping the direction one-way is what
lets work items 4, 5 and 6 add architectures without touching the referee.

**Every configuration is a name.** `cli.py` calls ``build(name)`` with no
parameters, deliberately -- a run is reproducible from its command line and there
is no way to smuggle a hyperparameter past the artifact. So a variant worth
scoring gets an entry here and appears in the scorecard by name, and the
parameters travel with it through :meth:`Detector.fingerprint`.

**The trained detectors are built lazily.** Importing them pulls in PyTorch,
which costs a couple of seconds, and ``list-tasks`` has no business paying for it.
"""
from __future__ import annotations

from sentinel_eval.detector import Detector

from . import baselines


def _trained(class_name: str):
    """Defer the torch import until something actually asks for the detector."""
    def build(**params) -> Detector:
        from . import detectors
        return getattr(detectors, class_name)(**params)
    return build


#: Names are lowercase and hyphenated. Items 5 and 6 add `gru` and `tcn`, which
#: reuse `detectors.ForecastDetector` with a different forecaster and the same
#: telemanom detection stack -- so the gate compares architectures and not
#: pipelines.
BUILDERS = {
    "mavg": baselines.MovingAverage,
    "rstd": baselines.RollingStd,
    "quiet": baselines.AlwaysQuiet,
    "random": baselines.RandomScore,
    "lstm-telemanom": _trained("ForecastDetector"),
    "lstm-quantile": _trained("TelemanomQuantile"),
    "lstm-smoke": _trained("TelemanomSmoke"),
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
