"""A decision layer that tests the relationship instead of the channels.

`docs/DECISIONS.md` D23 and `docs/MODELS.md` sections 11 and 12. The forecaster
is multivariate -- one model predicts every channel from every channel, which is
why 22 of 32 headline-cell events are caught (was "28 of 32", which was
`lstm-telemanom`'s; D37, D38) -- and everything downstream of it
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

#: Pulled toward the **identity of the correlation matrix** before inversion.
#:
#: **This was 0.01 toward `diag(S)` and that was wrong.** Shrinking a covariance
#: toward its own diagonal softens correlations and does nothing about channel
#: variances differing by orders of magnitude -- which ESA's within-group min-max
#: scaling guarantees they do. Measured, the condition number reached **4.3e8**,
#: a channel with tiny residual variance took an enormous precision weight, and
#: its numerical noise dominated the whitened length. Standardising first and
#: shrinking the *correlation* matrix toward the identity bounds the conditioning
#: by correlation structure alone, which is the quantity this stage is about.
SHRINKAGE = 0.05

#: A whitened C-dimensional residual has length near ``sqrt(C)`` by construction.
#: Asserted rather than trusted: the first run of this stage produced thresholds
#: of 25.7 to 36.3 against ``sqrt(12) = 3.5`` and scored anyway, silently. A
#: broken whitening must not be able to produce a number again.
LENGTH_TOLERANCE = 3.0

#: Refused above this. The failure mode is a near-singular covariance whose
#: inverse amplifies whichever direction is least observed, and it produces
#: plausible-looking large numbers rather than an error.
MAX_CONDITION = 1.0e4

#: Rank of the trailing local reference, used only under `local_reference`.
#: The same neighbourhood OS-CFAR draws its scale estimate from, and far below
#: the measured contamination rate, so a handful of anomalous steps inside the
#: window cannot drag the reference up with them (`docs/RESEARCH.md`).
LOCAL_RANK = 0.75

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

    #: **Disabled, and it is a re-derivation rather than a tuning.**
    #: telemanom's pruning sorts the sequence peaks and discards everything below
    #: the last normalised step-drop greater than ``p``. That ladder was derived
    #: against **per-channel absolute errors**; this rule feeds it one joint
    #: whitened length, whose distribution has no reason to share the shape
    #: ``p = 0.13`` was chosen for.
    #:
    #: Measured at 0.13 against 0.0 on identical residuals: `m1-g8.9.10` is
    #: **unchanged** -- 27/46 and 21/32 either way, 40 alarm ranges against 43 --
    #: and `m1-ss5` **improves**, recall 18/42 to 23/42 and headline cell 15/31 to
    #: **18/31**, for six extra ranges and **no** change to the rare-event rate,
    #: 2/48 at both settings. Free or better on both sets, so this is one global
    #: choice and not a per-set dial.
    #:
    #: Intermediate values were **not** swept. The claim is that a filter built
    #: for another quantity does not apply here, not that 0.0 is optimal.
    pruning_p: float = 0.0
    admission_rate: float = ADMISSION_RATE
    shrinkage: float = SHRINKAGE
    sample_stride: int = SAMPLE_STRIDE
    length_tolerance: float = LENGTH_TOLERANCE
    max_condition: float = MAX_CONDITION

    #: Compare the whitened length against a **trailing local quantile of itself**
    #: rather than one global quantile of the nominal pool. Default False, which
    #: is the rule every published whitened figure was measured under.
    #:
    #: The motivation is measured (docs/MODELS.md 13): the events this rule loses
    #: are weak, not invisible, and they peak at a median 0.240 of the global
    #: threshold. A local reference lowers the bar only where the neighbourhood is
    #: quiet. **It is also the shape that collapsed onto the noise floor when the
    #: forecaster improved** (D17, D18), which is why 13.3 defines the collapse
    #: signature before the run.
    local_reference: bool = False
    local_rank: float = LOCAL_RANK

    def as_dict(self) -> dict:
        return {"error_window": self.error_window,
                "error_window_stride": self.stride,
                "smoothing_window": self.smoothing_window,
                "error_buffer": self.error_buffer,
                "pruning_p": self.pruning_p,
                "admission_rate": self.admission_rate,
                "shrinkage": self.shrinkage,
                "sample_stride": self.sample_stride,
                "length_tolerance": self.length_tolerance,
                "max_condition": self.max_condition,
                "local_reference": self.local_reference,
                "local_rank": self.local_rank}


@dataclass(frozen=True)
class Whitening:
    """What nominal residuals fixed. The whole of what `model.bin` would carry.

    ``precision`` is C x C -- **144 floats at C=12**, against 91,640 model
    parameters -- and applying it in flight is a fixed matrix multiply with no
    state, which is why this shape was scoped as the cheap one.
    """

    mean: np.ndarray            #: (C,) nominal residual mean
    scale: np.ndarray           #: (C,) nominal residual sd, so precision is a correlation
    precision: np.ndarray       #: (C, C) inverse *correlation*, shrunk toward identity
    threshold: float            #: whitened length admitting `admission_rate` on nominal
    nominal_steps: int
    condition_number: float
    median_length: float        #: on nominal; must sit near sqrt(C) or this is broken
    local_multiplier: float | None = None   #: set only under `local_reference`
    nominal_admission: float | None = None  #: what the fitted rule admits on nominal

    def as_dict(self) -> dict:
        return {"mean": [float(m) for m in self.mean],
                "scale": [float(v) for v in self.scale],
                "threshold": float(self.threshold),
                "nominal_steps": int(self.nominal_steps),
                "condition_number": float(self.condition_number),
                "median_length": float(self.median_length),
                "expected_length": float(np.sqrt(self.mean.shape[0])),
                "local_multiplier": (None if self.local_multiplier is None
                                     else float(self.local_multiplier)),
                "nominal_admission": (None if self.nominal_admission is None
                                      else float(self.nominal_admission))}


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

    def add(self, block: np.ndarray, usable: np.ndarray | None = None) -> None:
        """``usable`` keeps only genuinely nominal rows.

        **The mask is not optional in a run and it was missing.**
        `harness._score_fold` applies `train_mask` to `detector.fit` and **not**
        to the `detector.score` call this calibration reads, so without it the
        99.9th percentile is set partly by the anomalies inside the fitting
        window -- pushed up by the events it exists to catch
        (`docs/NARRATIVE.md` section 6).
        """
        block = np.asarray(block, dtype=np.float64)
        if usable is not None:
            block = block[np.asarray(usable, dtype=bool)]
        if block.shape[0] == 0:
            return
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
        # Standardise before inverting, so what is inverted is a CORRELATION
        # matrix. Shrinking a covariance toward its own diagonal leaves channel
        # variances untouched, and ESA's within-group min-max scaling makes those
        # differ by orders of magnitude -- measured, a condition number of 4.3e8
        # and a whitened length dominated by the noise of the quietest channel.
        variance = np.diag(covariance).copy()
        floor = max(1e-30, 1e-10 * float(np.mean(variance)))
        scale = np.sqrt(np.maximum(variance, floor))
        correlation = covariance / np.outer(scale, scale)
        ridge = float(config.shrinkage)
        correlation = (1.0 - ridge) * correlation + ridge * np.eye(correlation.shape[0])
        precision = np.linalg.inv(correlation)
        precision = 0.5 * (precision + precision.T)
        condition = float(np.linalg.cond(correlation))
        if not np.isfinite(condition) or condition > config.max_condition:
            raise ReferenceError(
                f"the nominal correlation matrix is ill-conditioned "
                f"({condition:.3g} > {config.max_condition:.3g}). Its inverse would "
                f"amplify whichever direction is least observed and produce large "
                f"numbers that look like detections. Refusing to score."
            )

        sample = (np.concatenate(self.rows) if self.rows
                  else np.zeros((1, mean.shape[0]), dtype=np.float32))
        distances = whitened_length(sample, mean, scale, precision)
        median = float(np.median(distances))
        expected = float(np.sqrt(mean.shape[0]))
        tolerance = float(config.length_tolerance)
        if not (expected / tolerance <= median <= expected * tolerance):
            raise ReferenceError(
                f"whitened lengths sit at a median of {median:.2f} where a "
                f"{mean.shape[0]}-channel residual should sit near "
                f"sqrt(C) = {expected:.2f}. The whitening is not whitening. This "
                f"assertion exists because the first run of this stage produced "
                f"thresholds of 25.7 to 36.3 against sqrt(12) and scored anyway."
            )
        rate = float(config.admission_rate)
        threshold = float(np.quantile(distances, 1.0 - rate))
        local_multiplier = admitted = None
        if config.local_reference:
            # The multiplier for a trailing local reference, fitted on the same
            # nominal sample: the (1 - rate) quantile of d divided by the local
            # quantile of d. Dimensionless, and the same convention the global
            # threshold uses -- what changes is what it is measured against.
            local = _trailing_quantile(distances, config)
            usable = local > 0
            if usable.sum() < 100:
                raise ReferenceError(
                    "too few nominal windows carry a positive local reference to "
                    "fit a local multiplier"
                )
            ratios = distances[usable] / local[usable]
            local_multiplier = float(np.quantile(ratios, 1.0 - rate))
            admitted = float(np.mean(distances[usable]
                                     >= local_multiplier * local[usable]))
            # docs/MODELS.md 13.3, collapse condition 1. Measured on the nominal
            # pool before any event is scored, and it can fail while the adoption
            # number still looks healthy -- which is exactly what it is for.
            if admitted > 2.0 * rate:
                raise ReferenceError(
                    f"the local reference admits {admitted:.5f} of nominal data "
                    f"against a target of {rate:.5f} -- more than twice. That is "
                    f"the noise-floor collapse signature pre-registered in "
                    f"docs/MODELS.md 13.3, condition 1. Refusing to score."
                )
        else:
            admitted = float(np.mean(distances >= threshold))
        return Whitening(mean=mean, scale=scale, precision=precision,
                         threshold=threshold, nominal_steps=self.n,
                         condition_number=condition, median_length=median,
                         local_multiplier=local_multiplier, nominal_admission=admitted)


def _trailing_quantile(distance: np.ndarray, config: Config) -> np.ndarray:
    """``Q_p`` of the ``error_window`` whitened lengths ending at each segment.

    One value per segment, held across the segment's steps, so the reference is
    the same trailing window telemanom uses -- span ``error_window``, stride
    ``stride`` -- and the two rules are compared on identical history.
    """
    steps = distance.shape[0]
    span, stride = config.error_window, max(1, config.stride)
    out = np.zeros(steps, dtype=np.float64)
    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        window = distance[max(0, seg_lo - span):seg_hi]
        out[seg_lo:seg_hi] = np.quantile(window, config.local_rank) if window.size else 0.0
    return out


def whitened_length(residual: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                    precision: np.ndarray) -> np.ndarray:
    """``sqrt(z' P z)`` per timestep, ``z = (r - mu)/sigma``. One number, not C.

    Standardised first, so ``P`` is an inverse correlation and the length cannot
    be dominated by whichever channel happens to have the smallest variance.
    """
    z = (np.asarray(residual, dtype=np.float64) - mean) / scale
    return np.sqrt(np.maximum(np.einsum("ij,ij->i", z @ precision, z), 0.0))


def contributions(residual: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                  precision: np.ndarray) -> np.ndarray:
    """Which channel carries most of the whitened length, per timestep.

    **This names a channel, not a pair.** Objective.md 4.2 promises
    ``BattTemp / ChargeCurrent decoupled``, which needs the lagged pair map of
    section 7 -- and that does not exist (Objective.md 4.4). Reported so the
    attribution is at least about the whitened direction rather than about which
    raw error happened to be biggest.
    """
    z = (np.asarray(residual, dtype=np.float64) - mean) / scale
    return np.abs(z @ precision).argmax(axis=1).astype(np.int8)


def ratios(residual: np.ndarray, config: Config, whitening: Whitening,
           emission: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    distance = whitened_length(residual, whitening.mean, whitening.scale,
                               whitening.precision)
    who = contributions(residual, whitening.mean, whitening.scale, whitening.precision)

    if config.local_reference:
        if whitening.local_multiplier is None:
            raise ReferenceError(
                "local_reference is set but no local multiplier was fitted; the "
                "calibration and the scoring disagree about which rule this is"
            )
        local = _trailing_quantile(distance, config)
        eps_at = whitening.local_multiplier * local
        eps_at[eps_at <= 0] = float(whitening.threshold)
    else:
        eps_at = np.full(steps, float(whitening.threshold), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(eps_at > 0, distance / eps_at, 0.0).astype(np.float32)
    alarm = np.zeros(steps, dtype=bool)
    stride = max(1, config.stride)

    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - config.error_window)
        window = distance[reference_lo:seg_hi]
        offset = seg_lo - reference_lo
        eps = float(eps_at[seg_lo])
        sequences = telemanom.sequences_at(window, eps, config)
        if not sequences:
            continue
        emitted = False
        for keep, (lo, hi) in zip(telemanom.prune(window, sequences, eps, config.pruning_p),
                                  sequences):
            lo, hi = max(lo, offset), min(hi, window.shape[0])
            if keep and hi > lo:
                alarm[reference_lo + lo:reference_lo + hi] = True
                if np.any(window[offset:] >= eps):
                    emitted = True
        if emission is not None and emitted:
            # See `telemanom.channel_ratios`: the alarm reaches backwards, the
            # detector does not. This is when it could speak.
            emission[seg_hi - 1] = True

    out = raw / (1.0 + raw)
    out[alarm] = np.maximum(raw[alarm], 1.0)
    opening = min(telemanom.EWMA_SETTLE * config.smoothing_window, steps)
    out[:opening] = np.minimum(out[:opening], np.float32(1.0) - np.finfo(np.float32).eps)
    return out, who
