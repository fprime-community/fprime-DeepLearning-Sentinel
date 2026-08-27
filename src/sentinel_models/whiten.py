"""A decision layer that tests the relationship instead of the channels.

`docs/DECISIONS.md` D23 and `docs/MODELS.md` sections 11 and 12. The forecaster
is multivariate -- one model predicts every channel from every channel, which is
why 28 of 32 cross-channel events are caught -- and everything downstream of it
sees C independent error series. Smoothing, thresholding, sequence-finding and
pruning each take one channel at a time; `k`-of-`n` is the only stage that looks
at more than one, and it tests **co-occurrence, not relationship**.

That is not a neutral gap. **A commanded manoeuvre makes several channels
individually surprising at once, which is exactly what `k`-of-`n` rewards, and it
is nominal.** The stage meant to recover cross-channel structure recovers the
kind that generates false alarms, so the adoption number and the project's thesis
fail in the same place for the same reason.

**What this module does.** The residual *vector* is jointly informative even when
every component is unremarkable: a broken relationship is a **direction** in
``R^C`` that nominal data does not visit -- one channel departing while its
group-mate holds. That direction survives in the residuals already computed and
is destroyed downstream, first by ``np.abs`` taking the sign off, then by C
marginal thresholds that cannot express a covariance. So the score here is the
**whitened length** of the signed residual::

    d(t) = sqrt( (r - mu)' S^-1 (r - mu) )

against ``S``, the covariance of the residual over nominal data. A residual
consistent with normal co-variation scores low **however large it is**; one
orthogonal to it scores high **however small it is**.

**Why that is the false-alarm fix rather than a separate one.** A manoeuvre moves
channels together in the pattern the forecaster already learned, so its residual
lies along a high-variance direction of ``S`` and is divided down. A broken link
violates the pattern, so its residual points somewhere nominal data never went
and is divided up. Alarming only on relationship breaks is not a tuning target
reached by suppressing alarms -- it is what a decision layer does once it stops
reading C channels one at a time.

**What it is not.** It tests *linear* co-variation and it **names nothing**: the
strongest whitened component is reported for attribution, which names a channel
and not a pair. Objective.md 4.2 promises a named pair and 7 promises the lagged
map that would produce one. That map does not exist (Objective.md 4.4) and this
module does not add it.

Everything around the score is held identical to telemanom -- the same smoothing
span, the same trailing window, the same ``error_buffer`` dilation and the same
pruning -- so a comparison against `lstm-telemanom` differs in the decision rule
alone (`docs/DECISIONS.md` D21 holds ``error_buffer`` fixed for exactly this
reason). Both constants come from nominal residuals at the operating point this
project already uses; no label is consulted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import telemanom
from .reference import ReferenceError

#: Share of nominal timesteps the rule may alarm on. An operational input, not a
#: property of the detector (`docs/MODELS.md` 10.2); this default is the same
#: label-free operating point every baseline in this project is scored at.
ADMISSION_RATE = 0.001

#: Pulled toward the diagonal before inversion. The covariance of twelve
#: telemetry channels over millions of samples is well conditioned, but two
#: channels that move almost identically make it nearly singular and the inverse
#: then amplifies whichever direction is least observed. A small ridge costs
#: almost nothing and removes the failure mode; recorded rather than tuned.
SHRINKAGE = 0.01

#: Rows kept for the threshold quantile. The covariance is accumulated exactly in
#: one pass; the score's own distribution cannot be until the covariance exists,
#: so a strided sample is held back rather than paying a second forecast pass.
SAMPLE_STRIDE = 37


@dataclass(frozen=True)
class Config:
    """Held identical to telemanom wherever the two overlap."""

    error_window: int = telemanom.ERROR_WINDOW_BATCH * telemanom.ERROR_WINDOW_COUNT
    stride: int = telemanom.ERROR_WINDOW_BATCH
    smoothing_window: int = int(telemanom.ERROR_WINDOW_BATCH
                                * telemanom.ERROR_WINDOW_COUNT
                                * telemanom.SMOOTHING_PERC)
    error_buffer: int = telemanom.ERROR_BUFFER
    pruning_p: float = telemanom.PRUNING_P
    admission_rate: float = ADMISSION_RATE
    shrinkage: float = SHRINKAGE
    sample_stride: int = SAMPLE_STRIDE

    def as_dict(self) -> dict:
        return {"error_window": self.error_window,
                "error_window_stride": self.stride,
                "smoothing_window": self.smoothing_window,
                "error_buffer": self.error_buffer,
                "pruning_p": self.pruning_p,
                "admission_rate": self.admission_rate,
                "shrinkage": self.shrinkage,
                "sample_stride": self.sample_stride}


@dataclass(frozen=True)
class Whitening:
    """What nominal residuals fixed. The whole of what `model.bin` would carry.

    ``precision`` is C x C -- **144 floats at C=12**, against 91,640 model
    parameters -- and applying it in flight is a fixed matrix multiply with no
    state, which is why this shape was scoped as the cheap one.
    """

    mean: np.ndarray            #: (C,) nominal residual mean
    precision: np.ndarray       #: (C, C) inverse covariance, shrunk
    threshold: float            #: whitened length admitting `admission_rate` on nominal
    nominal_steps: int
    condition_number: float

    def as_dict(self) -> dict:
        return {"mean": [float(m) for m in self.mean],
                "threshold": float(self.threshold),
                "nominal_steps": int(self.nominal_steps),
                "condition_number": float(self.condition_number),
                "precision_diagonal": [float(v) for v in np.diag(self.precision)]}


class Accumulator:
    """Exact first and second moments in one pass, plus rows for the quantile.

    The fitting window reaches 11M steps and holding its residual matrix is 530
    MB. The covariance needs only ``n``, the sum and the Gram matrix, all of
    which are O(C^2); the threshold needs the score's own distribution, which
    cannot be formed until the covariance exists, so a strided sample is kept and
    scored afterwards rather than paying a second forecast pass.
    """

    def __init__(self, channels: int, stride: int) -> None:
        self.n = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.gram = np.zeros((channels, channels), dtype=np.float64)
        self.stride = max(1, stride)
        self.rows: list[np.ndarray] = []
        self._offset = 0

    def add(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float64)
        self.n += block.shape[0]
        self.total += block.sum(axis=0)
        self.gram += block.T @ block
        start = (-self._offset) % self.stride
        if start < block.shape[0]:
            self.rows.append(block[start::self.stride].astype(np.float32))
        self._offset += block.shape[0]

    def finish(self, config: Config) -> Whitening:
        if self.n < 2:
            raise ReferenceError("a covariance needs at least two nominal samples")
        mean = self.total / self.n
        covariance = self.gram / self.n - np.outer(mean, mean)
        # Symmetrise: the two halves differ only by float64 rounding, and an
        # asymmetric matrix makes the inverse subtly direction-dependent.
        covariance = 0.5 * (covariance + covariance.T)
        ridge = float(config.shrinkage)
        covariance = (1.0 - ridge) * covariance + ridge * np.diag(np.diag(covariance))
        # A channel that never moves has zero variance and no inverse. Floor it
        # rather than raising: a constant channel carries no relationship and
        # should contribute nothing, which is what a large diagonal achieves.
        floor = max(1e-30, 1e-12 * float(np.mean(np.diag(covariance))))
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), floor))
        precision = np.linalg.inv(covariance)
        precision = 0.5 * (precision + precision.T)

        sample = (np.concatenate(self.rows) if self.rows
                  else np.zeros((1, mean.shape[0]), dtype=np.float32))
        distances = whitened_length(sample, mean, precision)
        threshold = float(np.quantile(distances, 1.0 - config.admission_rate))
        return Whitening(mean=mean, precision=precision, threshold=threshold,
                         nominal_steps=self.n,
                         condition_number=float(np.linalg.cond(covariance)))


def whitened_length(residual: np.ndarray, mean: np.ndarray,
                    precision: np.ndarray) -> np.ndarray:
    """``sqrt((r - mu)' P (r - mu))`` per timestep. One number, not C of them."""
    centred = np.asarray(residual, dtype=np.float64) - mean
    quadratic = np.einsum("ij,ij->i", centred @ precision, centred)
    return np.sqrt(np.maximum(quadratic, 0.0))


def contributions(residual: np.ndarray, mean: np.ndarray,
                  precision: np.ndarray) -> np.ndarray:
    """Which channel carries most of the whitened length, per timestep.

    **This names a channel, not a pair.** Objective.md 4.2 promises
    ``BattTemp / ChargeCurrent decoupled``, which needs the lagged pair map of
    section 7 -- and that does not exist (Objective.md 4.4). Reported so the
    attribution is at least about the whitened direction rather than about which
    raw error happened to be biggest.
    """
    centred = np.asarray(residual, dtype=np.float64) - mean
    return np.abs(centred @ precision).argmax(axis=1).astype(np.int8)


def ratios(residual: np.ndarray, config: Config, whitening: Whitening
           ) -> tuple[np.ndarray, np.ndarray]:
    """Score a whole window. Returns ``(ratios, attribution)``.

    telemanom's encoding and telemanom's alarm shaping, deliberately: the same
    trailing segments, the same ``error_buffer`` dilation, the same pruning, and
    ``d / threshold`` folded so the harness's fixed 1.0 cut reproduces the rule.
    Everything except the decision itself is held constant, so a comparison
    against `lstm-telemanom` differs in the decision rule alone.
    """
    residual = np.asarray(residual, dtype=np.float32)
    if residual.ndim != 2:
        raise ReferenceError(f"ratios expects (steps, channels), got {residual.shape}")
    steps = residual.shape[0]
    distance = whitened_length(residual, whitening.mean, whitening.precision)
    who = contributions(residual, whitening.mean, whitening.precision)

    eps = float(whitening.threshold)
    raw = (distance / eps if eps > 0 else np.zeros_like(distance)).astype(np.float32)
    alarm = np.zeros(steps, dtype=bool)
    stride = max(1, config.stride)

    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - config.error_window)
        window = distance[reference_lo:seg_hi]
        offset = seg_lo - reference_lo
        sequences = telemanom.sequences_at(window, eps, config)
        if not sequences:
            continue
        for keep, (lo, hi) in zip(telemanom.prune(window, sequences, eps, config.pruning_p),
                                  sequences):
            lo, hi = max(lo, offset), min(hi, window.shape[0])
            if keep and hi > lo:
                alarm[reference_lo + lo:reference_lo + hi] = True

    out = raw / (1.0 + raw)
    out[alarm] = np.maximum(raw[alarm], 1.0)
    opening = min(telemanom.EWMA_SETTLE * config.smoothing_window, steps)
    out[:opening] = np.minimum(out[:opening], np.float32(1.0) - np.finfo(np.float32).eps)
    return out, who
