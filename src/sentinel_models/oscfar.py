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

#: The label-free operating point, unchanged from
#: `sentinel_eval.detector.DEFAULT_THRESHOLD_QUANTILE`. Repeated as a constant
#: here rather than imported, because `sentinel_models` may import
#: `sentinel_eval.detector` and nothing else from the harness
#: (tests/test_layering.py) and a threshold rule should not widen that.
CALIBRATION_QUANTILE = 0.999

#: How many nominal segments the local multiplier is estimated from. Running the
#: full local rule over a 10.8M-step fitting window to fit one scalar is the cost
#: the NDT short-circuit exists to avoid; 2,000 segments is 140,000 samples per
#: channel, which is far more than a 99.9th percentile needs.
CALIBRATION_SEGMENTS = 2_000


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
    calibration_quantile: float = CALIBRATION_QUANTILE
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
                "calibration_quantile": self.calibration_quantile,
                "calibration_segments": self.calibration_segments,
                "guard_segment": self.guard_segment}


@dataclass(frozen=True)
class Calibration:
    """What the fitting window's nominal residuals fixed, per channel.

    Both are quantiles of nominal data at the same level, which is the whole of
    why neither is a free parameter: ``floor`` is the operating point the trivial
    baselines already use, and ``alpha`` is that same operating point applied to
    the locally normalised error instead of the raw one.
    """

    alpha: np.ndarray          #: multiplier on the local order statistic, per channel
    floor: np.ndarray          #: absolute lower bound on eps, per channel
    nominal_steps: int

    def as_dict(self) -> dict:
        return {"alpha": [round(float(a), 6) for a in self.alpha],
                "floor": [float(f) for f in self.floor],
                "nominal_steps": self.nominal_steps}


def _reference_quantiles(e_s: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """``Q_p`` of each segment's trailing reference window, and the segment starts.

    One channel. The window is the same one telemanom uses -- span
    ``error_window`` ending at the segment -- so the two rules are compared on
    identical history and differ in what they compute from it.
    """
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)
    starts = np.arange(0, steps, stride)
    out = np.empty(starts.shape[0], dtype=np.float64)
    for i, seg_lo in enumerate(starts):
        seg_hi = min(seg_lo + stride, steps)
        lo = max(0, seg_lo - span)
        window = e_s[lo:seg_lo] if (config.guard_segment and seg_lo > lo) else e_s[lo:seg_hi]
        out[i] = np.quantile(window, config.rank_p) if window.size else 0.0
    return out, starts


def calibrate(nominal: np.ndarray, config: Config, *, rng=None) -> Calibration:
    """Fit both multipliers on nominal residuals. No label is consulted.

    ``nominal`` is the fitting window's smoothed error, ``(steps, channels)``.
    That window is normal-only by construction, so this is the pre-launch
    calibration an adopting mission performs on its own flatsat data.
    """
    nominal = np.asarray(nominal, dtype=np.float32)
    if nominal.ndim != 2:
        raise ReferenceError(f"calibrate expects (steps, channels), got {nominal.shape}")
    steps, channels = nominal.shape
    q = float(config.calibration_quantile)
    rng = np.random.default_rng(0) if rng is None else rng

    floor = np.quantile(nominal, q, axis=0).astype(np.float64)

    stride = max(1, config.stride)
    n_segments = max(1, (steps + stride - 1) // stride)
    take = min(config.calibration_segments, n_segments)
    # Sampled rather than swept, and the sample is seeded so a calibration is
    # reproducible from cold. Segments that precede a full reference window are
    # excluded: their quantile is taken over a short window and is not the
    # quantity the scored path will see.
    first = config.error_window // stride
    pool = np.arange(first, n_segments) if n_segments > first else np.arange(n_segments)
    picked = np.sort(rng.choice(pool, size=min(take, pool.shape[0]), replace=False))

    alpha = np.empty(channels, dtype=np.float64)
    for c in range(channels):
        column = nominal[:, c]
        ratios = []
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
            ratios.append(column[seg_lo:seg_hi] / scale)
        # A channel with no usable scale anywhere cannot be normalised locally.
        # 1.0 makes the local term equal its own reference quantile, which the
        # floor then dominates -- silence rather than a guess.
        alpha[c] = float(np.quantile(np.concatenate(ratios), q)) if ratios else 1.0
    return Calibration(alpha=alpha, floor=floor, nominal_steps=int(steps))


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
