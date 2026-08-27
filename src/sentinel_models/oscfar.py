"""A local order statistic under a nominal floor. `docs/DECISIONS.md` D20.

Two thresholding rules have been measured on identical weights and each fails in
the opposite direction. telemanom's nonparametric dynamic threshold responds to
the **local** error scale, and when a better forecaster made that scale collapse
three- to eightfold the threshold followed it onto the noise floor: fifty alarm
ranges became 1,505 while recall and lead time held (D17, D18). A global quantile
responds to **absolute magnitude**, so a rising error must grow before it crosses
and growing takes time: median lead **-122**, every catch after the event began
(D13).

Those are the same fact seen twice, and the sentence that makes a global rule
unusable *as a threshold* is what makes it sound *as a floor*. The local term
still triggers at onset whenever the local scale is healthy; the floor only
decides what happens when it is not.

    eps(t) = max( alpha * Q_p(R(t)) ,   local -- onset sensitivity
                  floor            )    floor -- bounds the collapse

**`mu` and `sigma` appear nowhere, and that is a measurement rather than a
preference.** Between the two weight generations the within-window standard
deviation fell three- to eightfold while the window maximum did not, at excess
kurtosis reaching 6,754. Every member of the ``mu + k*sigma`` family inherits
that whatever ``k`` is and however it was chosen. An order statistic is unchanged
by how heavy the tail is, which is what removes it.

**(!) The objection to this design, which is in the pre-registration rather than
in a caveat afterwards.** An order statistic is immune to tail *weight*, not to
scale *collapse*: ``Q_p`` of a uniformly tiny window is tiny. So the honest claim
is narrower than *order statistics fix it* --

    what fixes it is calibrating the multiplier against nominal residuals
    instead of transcribing a constant

-- the order statistic is what makes that calibration robust, and the floor
carries the rest. If the floor does all the work and the local term never binds,
this has collapsed back to `lstm-quantile` and is reported as such rather than
defended. :func:`binding_rate` is the number that decides it.

**Nothing here is fitted against a label.** Both multipliers come from the
fitting window, which is normal-only by construction (`splits.train_mask`
removes annotated anomalies), and both use the operating point this project has
used since its first baseline -- `sentinel_eval.detector`'s label-free 99.9th
percentile. The same idea twice: once on the raw error, once on the locally
normalised one. A spacecraft that has never failed can compute both. It cannot
compute recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import telemanom
from .reference import ReferenceError

#: Rank of the local order statistic. Bounded by the **measured** contamination
#: rate rather than transcribed: the median reference window carries 53
#: exceedances in 2,170 samples, 2.4% (docs/THRESHOLD.md), so a rank at 0.75 sits
#: far under any plausible breakdown point. OS-CFAR's own convention is the same
#: neighbourhood, for the same robustness reason (docs/RESEARCH.md).
RANK_P = 0.75

#: The share of nominal timesteps the rule is allowed to alarm on. **Not a
#: property of the detector.** It is an operational input -- how many alarms a
#: mission's operators can absorb -- and a two-person CubeSat team answers
#: differently from an ESOC control room (`docs/RESEARCH.md`: EEMUA 191,
#: ISA-18.2, and fraud detection's precision@k).
#:
#: **This default exists so the dataclass has one. It is not a recommendation.**
#: An earlier version of this design fixed 0.1% and called it label-free, which
#: it is -- and label-free is not the same as derived. Relabelling an arbitrary
#: threshold as an arbitrary budget does not remove the arbitrary constant. The
#: rule is therefore **reported as a curve across admission rates**, never at a
#: point, and no operating point is recommended from scored results.
ADMISSION_RATE = 0.001

#: How the two multipliers are fitted.
#:
#: ``independent`` calibrates each term to the target rate on its own and takes
#: the maximum. **It is wrong, and it is kept because it is what ran**: the
#: maximum of two thresholds each admitting r admits far less than r, so the
#: rule was far quieter than its budget asked for. Preserved so the corrected
#: form is measured against it rather than replacing it.
#:
#: ``joint`` fits both together so that the combined rule -- the maximum --
#: admits the target rate. This is the design as intended.
#:
#: ``local_only`` drops the floor entirely. The local order statistic has never
#: operated: the floor dominated every window of the first run, so there is no
#: measurement of it at all. Run bare, it is either a better design than the one
#: proposed or a demonstration that the floor was load-bearing.
INDEPENDENT, JOINT, LOCAL_ONLY = "independent", "joint", "local_only"
CALIBRATIONS = (INDEPENDENT, JOINT, LOCAL_ONLY)

#: Nominal segments the multipliers are estimated from. Scaled to the admission
#: rate rather than fixed: a 0.01% quantile estimated from 140,000 samples rests
#: on fourteen of them. :func:`_segment_budget` keeps ~200 above the cut.
CALIBRATION_SEGMENTS = 2_000
CALIBRATION_TARGET_EXCEEDANCES = 200


@dataclass(frozen=True)
class Config:
    """The rule's settings. Everything shared with telemanom keeps its value."""

    error_window: int = telemanom.ERROR_WINDOW_BATCH * telemanom.ERROR_WINDOW_COUNT
    stride: int = telemanom.ERROR_WINDOW_BATCH
    smoothing_window: int = int(telemanom.ERROR_WINDOW_BATCH
                                * telemanom.ERROR_WINDOW_COUNT
                                * telemanom.SMOOTHING_PERC)
    error_buffer: int = telemanom.ERROR_BUFFER
    pruning_p: float = telemanom.PRUNING_P
    rank_p: float = RANK_P
    admission_rate: float = ADMISSION_RATE
    calibration: str = INDEPENDENT
    calibration_segments: int = CALIBRATION_SEGMENTS

    #: See `telemanom.Config.guard_segment`. Measured as an arm, not assumed.
    guard_segment: bool = False

    def as_dict(self) -> dict:
        return {"error_window": self.error_window,
                "error_window_stride": self.stride,
                "smoothing_window": self.smoothing_window,
                "error_buffer": self.error_buffer,
                "pruning_p": self.pruning_p,
                "rank_p": self.rank_p,
                "admission_rate": self.admission_rate,
                "calibration": self.calibration,
                "calibration_segments": self.calibration_segments,
                "guard_segment": self.guard_segment}


@dataclass(frozen=True)
class Calibration:
    """What the fitting window's nominal residuals fixed, per channel.

    ``floor`` of zero means the floor is disabled -- the ``local_only`` arm.
    ``admitted`` is what the fitted rule actually let through on nominal data,
    which is the check that the calibration did what it was asked: under
    ``independent`` it comes out well below the target, and that gap is the
    arithmetic error the first run made.
    """

    alpha: np.ndarray          #: multiplier on the local order statistic, per channel
    floor: np.ndarray          #: absolute lower bound on eps, per channel; 0 disables it
    nominal_steps: int
    mode: str = INDEPENDENT
    target_rate: float = ADMISSION_RATE
    admitted: float = float("nan")   #: measured nominal admission of the fitted rule

    def as_dict(self) -> dict:
        return {"mode": self.mode, "target_rate": self.target_rate,
                "admitted_on_nominal": (None if self.admitted != self.admitted
                                        else round(float(self.admitted), 8)),
                "alpha": [round(float(a), 6) for a in self.alpha],
                "floor": [float(f) for f in self.floor],
                "nominal_steps": self.nominal_steps}


def _segment_budget(config: Config, n_segments: int) -> int:
    """Enough sampled segments that the fitted quantile rests on real samples.

    A 0.01% quantile of 140,000 points is decided by fourteen of them. Scaling
    the sample to the rate keeps roughly `CALIBRATION_TARGET_EXCEEDANCES` above
    the cut whatever the budget, which is what stops the low-rate end of the
    curve being noise dressed as a measurement.
    """
    rate = max(float(config.admission_rate), 1e-9)
    need = CALIBRATION_TARGET_EXCEEDANCES / (rate * max(1, config.stride))
    return int(min(n_segments, max(config.calibration_segments, need)))


def _sampled(column: np.ndarray, config: Config, picked, steps: int):
    """``(values, scales)`` for the sampled segments of one channel.

    ``scales`` is that segment's local order statistic, repeated per step, so
    the two arrays align and a candidate rule can be evaluated by elementwise
    comparison rather than by a second pass over the window.
    """
    values, scales = [], []
    stride = max(1, config.stride)
    for seg in picked:
        seg_lo = int(seg) * stride
        seg_hi = min(seg_lo + stride, steps)
        lo = max(0, seg_lo - config.error_window)
        window = (column[lo:seg_lo] if (config.guard_segment and seg_lo > lo)
                  else column[lo:seg_hi])
        if window.size == 0 or seg_hi <= seg_lo:
            continue
        scale = float(np.quantile(window, config.rank_p))
        if scale <= 0.0:
            continue
        segment = column[seg_lo:seg_hi]
        values.append(segment)
        scales.append(np.full(segment.shape[0], scale, dtype=np.float64))
    if not values:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(values).astype(np.float64), np.concatenate(scales)


def _admitted(values, scales, alpha: float, floor: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= np.maximum(alpha * scales, floor)))


def _fit_jointly(values, scales, config: Config) -> tuple[float, float]:
    """Fit both terms so that **the maximum** admits the target rate.

    The first run fitted each term to the target on its own and took the
    maximum, which admits far less than the target -- an arithmetic error, not a
    property of order statistics. Here a single inner level ``u`` sets both, and
    ``u`` is bisected until the combined rule admits what was asked. Both terms
    still come from nominal residuals only; what changed is that they are fitted
    against the rule that is actually applied.
    """
    target = float(config.admission_rate)
    lo, hi = target, 0.5                      # u >= target, since max() only tightens
    best = (float(np.quantile(values / np.maximum(scales, 1e-30), 1.0 - target)),
            float(np.quantile(values, 1.0 - target)))
    for _ in range(40):
        u = 0.5 * (lo + hi)
        alpha = float(np.quantile(values / np.maximum(scales, 1e-30), 1.0 - u))
        floor = float(np.quantile(values, 1.0 - u))
        got = _admitted(values, scales, alpha, floor)
        best = (alpha, floor)
        if abs(got - target) <= 0.02 * target:
            break
        # A looser inner level admits more; the relation is monotone in u.
        if got < target:
            lo = u
        else:
            hi = u
    return best


def calibrate(nominal: np.ndarray, config: Config, *, rng=None) -> Calibration:
    """Fit the multipliers on nominal residuals. No label is consulted.

    ``nominal`` is the fitting window's smoothed error, ``(steps, channels)``.
    That window is normal-only by construction (`splits.train_mask` removes
    annotated anomalies), so this is the pre-launch calibration an adopting
    mission performs on its own flatsat data. What a mission supplies is the
    **admission rate** -- what its operators can absorb -- and that is an
    operational input rather than a constant this project gets to choose, which
    is why the rule is reported as a curve across rates.
    """
    nominal = np.asarray(nominal, dtype=np.float32)
    if nominal.ndim != 2:
        raise ReferenceError(f"calibrate expects (steps, channels), got {nominal.shape}")
    if config.calibration not in CALIBRATIONS:
        raise ReferenceError(
            f"unknown calibration {config.calibration!r}; expected one of {CALIBRATIONS}"
        )
    steps, channels = nominal.shape
    rate = float(config.admission_rate)
    rng = np.random.default_rng(0) if rng is None else rng
    stride = max(1, config.stride)

    n_segments = max(1, (steps + stride - 1) // stride)
    take = _segment_budget(config, n_segments)
    first = config.error_window // stride
    pool = np.arange(first, n_segments) if n_segments > first else np.arange(n_segments)
    picked = np.sort(rng.choice(pool, size=min(take, pool.shape[0]), replace=False))

    alpha = np.empty(channels, dtype=np.float64)
    floor = np.zeros(channels, dtype=np.float64)
    admitted = np.zeros(channels, dtype=np.float64)
    for c in range(channels):
        values, scales = _sampled(nominal[:, c], config, picked, steps)
        if values.size == 0:
            # No usable local scale anywhere. 1.0 makes the local term equal its
            # own reference quantile; the floor, where there is one, then decides.
            alpha[c] = 1.0
            floor[c] = float(np.quantile(nominal[:, c], 1.0 - rate)) \
                if config.calibration != LOCAL_ONLY else 0.0
            continue
        ratios = values / np.maximum(scales, 1e-30)
        if config.calibration == LOCAL_ONLY:
            alpha[c], floor[c] = float(np.quantile(ratios, 1.0 - rate)), 0.0
        elif config.calibration == INDEPENDENT:
            alpha[c] = float(np.quantile(ratios, 1.0 - rate))
            floor[c] = float(np.quantile(values, 1.0 - rate))
        else:
            alpha[c], floor[c] = _fit_jointly(values, scales, config)
        admitted[c] = _admitted(values, scales, alpha[c], floor[c])
    return Calibration(alpha=alpha, floor=floor, nominal_steps=int(steps),
                       mode=config.calibration, target_rate=rate,
                       admitted=float(np.mean(admitted)))


def channel_ratios(e_s: np.ndarray, config: Config, alpha: float, floor: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Score one channel. Returns ``(ratios, local_bound)``.

    ``local_bound`` is True for each segment whose threshold came from the local
    term rather than the floor -- :func:`binding_rate` reads it, and it is the
    number the falsification condition in docs/MODELS.md section 10.3 turns on.

    The encoding is telemanom's, deliberately: ``e_s / eps``, surviving sequences
    forced to at least 1.0 and everything else squashed below it by ``r/(1+r)``,
    so the harness's fixed 1.0 cut reproduces this rule exactly and a
    threshold-free ranking metric still reads an order. Two rules that differ in
    their threshold and agree in everything else is what makes the pair a
    comparison.
    """
    e_s = np.asarray(e_s, dtype=np.float32)
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)

    raw = np.zeros(steps, dtype=np.float32)
    alarm = np.zeros(steps, dtype=bool)
    n_segments = (steps + stride - 1) // stride
    local_bound = np.zeros(max(1, n_segments), dtype=bool)

    for index, seg_lo in enumerate(range(0, steps, stride)):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - span)
        window = e_s[reference_lo:seg_hi]
        offset = seg_lo - reference_lo
        scale_over = (window[:offset] if (config.guard_segment and offset > 0) else window)
        if scale_over.size == 0:
            scale_over = window

        local = float(alpha) * float(np.quantile(scale_over, config.rank_p))
        eps = max(local, float(floor))
        # With no floor the local term decides every segment by construction, so
        # the rate is 1.0 -- which is the honest reading, not a degenerate one:
        # `local_only` exists precisely to run the local term unaided.
        local_bound[index] = local > float(floor)
        if eps <= 0:
            continue

        raw[seg_lo:seg_hi] = e_s[seg_lo:seg_hi] / eps
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
    return out, local_bound


def top_ratios(smoothed: np.ndarray, config: Config, calibration: Calibration,
               depth: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """`telemanom.top_ratios` over this rule, plus the binding rate.

    Running insertion over columns rather than a materialised ``(steps,
    channels)`` matrix, for the reason in `docs/DECISIONS.md` D10: that matrix is
    530 MB on the largest window here.
    """
    steps, channels = smoothed.shape
    if calibration.alpha.shape[0] != channels:
        raise ReferenceError(
            f"calibration covers {calibration.alpha.shape[0]} channels, "
            f"scoring {channels}. A calibration belongs to the fit that produced it"
        )
    top = np.full((depth, steps), -np.inf, dtype=np.float32)
    who = np.zeros(steps, dtype=np.int8)
    bound = 0
    total = 0

    for c in range(channels):
        incoming, local_bound = channel_ratios(smoothed[:, c], config,
                                               calibration.alpha[c], calibration.floor[c])
        bound += int(local_bound.sum())
        total += int(local_bound.shape[0])
        who[incoming > top[0]] = c
        for level in range(depth):
            displaced = np.minimum(incoming, top[level])
            np.maximum(top[level], incoming, out=top[level])
            incoming = displaced
    return top, who, (bound / total if total else 0.0)


def binding_rate(local_bound: np.ndarray) -> float:
    """Share of segments whose threshold came from the local term.

    **The falsification condition.** Below 0.10 the floor is doing all the work,
    the design has collapsed back to a global quantile, and docs/MODELS.md
    section 10.3 says it is reported as that rather than defended.
    """
    local_bound = np.asarray(local_bound, dtype=bool)
    return float(local_bound.mean()) if local_bound.size else 0.0
