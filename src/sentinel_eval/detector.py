"""The contract every candidate implements. The harness's only model-facing surface.

`sentinel_eval` never imports a detector, and no detector is special-cased. That
is what makes the architecture gate a table of numbers instead of an argument:
LSTM, GRU, TCN and a two-line moving average all arrive through this one door and
are scored by identical code.

A detector does two things:

``fit``    learn what normal looks like, from a window the harness has already
           stripped of annotated anomalies. Objective.md 6.1 -- normal-only
           training is not a preference, it is the only option: a spacecraft has
           no failure examples.
``score``  emit an anomaly score per timestep -- higher is more anomalous.
           ``(T,)`` for a single verdict, or ``(T, C)`` to attribute divergence
           to individual channels, which is what the explanation layer needs.

**Thresholds are label-free.** :meth:`Detector.threshold_from` picks an operating
point from *training* scores only, never from the labels it is about to be scored
against. The harness separately reports the best achievable F0.5 over a sweep,
clearly marked as an oracle upper bound, so the honest number and the ceiling are
never confused for one another.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .errors import DetectorError

#: A label-free operating point: alarm above this quantile of training scores.
DEFAULT_THRESHOLD_QUANTILE = 0.999

#: Optional, and absent unless a detector sets it: a boolean mask over the scored
#: window marking the timesteps at which the detector could actually **emit**.
#:
#: For most detectors that is simply where the score crosses, and the harness
#: defaults to the alarm mask when this is absent -- so nothing changes for them.
#: It exists because telemanom's `error_buffer` widens every alarm range
#: ``error_buffer - 1`` steps **backwards from a crossing that has already
#: happened**, and lead time is measured from a range's start. A flight component
#: cannot emit retroactively, so that measurement credited warning nobody
#: received: measured, the reported +26 median became **+0.0**
#: (`docs/DECISIONS.md` D21).
EMISSION_ATTRIBUTE = "last_emission"


@dataclass(frozen=True)
class Context:
    """What a detector may know. Deliberately no labels, ever."""

    mission: str
    channels: tuple[str, ...]
    groups: tuple[int, ...]
    period_seconds: int
    fold: int
    window: tuple[int, int]

    #: Telecommand impulses for exactly this window, ``uint8[T, K]``, or None when
    #: the run did not ask for them. **Not a label.** A command is an input the
    #: spacecraft itself has, known before it executes and available in flight;
    #: telemanom feeds its model the same thing (docs/DECISIONS.md D6). The
    #: promise this class makes -- no labels, ever -- is intact.
    #:
    #: Sliced identically to the values a detector is handed, so a detector may
    #: index the two together without knowing which window it is in.
    commands: np.ndarray | None = None
    command_ids: tuple[str, ...] = ()

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def has_commands(self) -> bool:
        return self.commands is not None and self.commands.shape[1] > 0


class Detector(ABC):
    """Base class. Subclasses live in ``sentinel_models``, never here."""

    name: str = "detector"

    def __init__(self, **params) -> None:
        self._params = dict(params)

    # -- identity ----------------------------------------------------------
    @property
    def params(self) -> dict:
        return dict(self._params)

    def fingerprint(self) -> str:
        """Stable hash of name and parameters, for run provenance."""
        blob = json.dumps({"name": self.name, "params": self._params},
                          sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def __str__(self) -> str:
        if not self._params:
            return self.name
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in sorted(self._params.items())) + ")"

    # -- the contract ------------------------------------------------------
    @property
    def warmup_steps(self) -> int:
        """History needed before a score is trustworthy.

        A window-based detector scoring a test fold from its first timestep has
        no history behind it and produces garbage for its first ``window`` steps.
        The harness therefore hands it a prefix reaching back this far and
        discards the corresponding scores, rather than letting a fold boundary
        masquerade as an anomaly.
        """
        return 0

    def fit(self, values: np.ndarray, usable: np.ndarray, context: Context) -> None:
        """Learn normality. ``usable`` is True where a timestep may be trained on."""

    @abstractmethod
    def score(self, values: np.ndarray, valid: np.ndarray, context: Context) -> np.ndarray:
        """Anomaly score per timestep. Higher is more anomalous."""

    def threshold_from(self, train_scores: np.ndarray) -> float | None:
        """An operating point chosen without ever seeing a label."""
        finite = train_scores[np.isfinite(train_scores)]
        if finite.size == 0:
            return None
        return float(np.quantile(finite, DEFAULT_THRESHOLD_QUANTILE))


def reduce_scores(scores: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalise a detector's output to ``(T,)``, keeping per-channel detail.

    A ``(T, C)`` score is reduced by the maximum across channels: one channel
    diverging from its forecast is a divergence, and which one it was is exactly
    what the explanation layer reports, so the detail is returned rather than
    discarded.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        per_channel = None
        combined = scores
    elif scores.ndim == 2:
        per_channel = scores
        # A timestep unobserved on every channel is all-NaN; nanmax would warn
        # and return NaN. Score it -inf instead: no data is not an alarm, and the
        # metrics mask it out anyway.
        observed = np.isfinite(scores).any(axis=1)
        combined = np.full(scores.shape[0], -np.inf, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            combined[observed] = np.nanmax(scores[observed], axis=1)
    else:
        raise DetectorError(f"score must be 1-D or 2-D, got shape {scores.shape}")

    if combined.shape[0] != n_steps:
        raise DetectorError(
            f"score has {combined.shape[0]:,} timesteps, the window has {n_steps:,}"
        )
    return combined, per_channel
