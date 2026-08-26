"""telemanom's detection stack: smoothed errors, a dynamic threshold, pruning.

The LSTM is the famous half; this is the half that does the work. Hundman et al.
(KDD 2018) contribute a **nonparametric dynamic threshold** -- no assumption that
prediction errors are Gaussian, no fixed cut -- and a **pruning** step that
throws away detections which are not clearly separated from the rest. Both exist
to protect precision, which is exactly what the gate metric weights twice
(docs/HARNESS.md section 1).

Nothing here knows what produced the errors. Work items 5 and 6 hand it a GRU's
or a TCN's residuals and it behaves identically, which is the only way the
architecture gate compares architectures rather than detection pipelines.

**How a dynamic threshold survives contact with a fixed one.** The harness's
contract is a score per timestep and a single scalar operating point
(`sentinel_eval.detector`). A threshold that moves cannot be expressed as that
scalar -- so it is folded into the score instead: what a detector emits is
``e_s / eps``, the smoothed error divided by *its own window's* dynamic
threshold, and the harness alarming at a fixed 1.0 then reproduces the
nonparametric decision exactly. Pruned sequences are mapped to ``r/(1+r)``, which
is below 1 for every input and order-preserving, so a threshold-free ranking
metric like VUS-PR degrades gracefully instead of being destroyed by a hard zero.

**The window overlap is not an implementation detail, and we nearly lost it.**
telemanom slides its 2,100-error window forward 70 at a time, so every error is
judged in thirty overlapping windows. That looked like an obvious thing to
economise on -- ~52,600 windows per channel per fold on ESA-ADB -- and the first
version here stepped by a whole window instead.

Measured, that was silently fatal. A threshold derived from ``mu + z*sigma`` of
the window it is judging cannot see an anomaly that fills its own window: the
anomaly raises both moments until it sits below its own threshold. With
non-overlapping windows a 100-step injection was detected and a 500-, 2,100- and
8,828-step injection were all missed. `m1-g8.9.10`'s headline-cell events have a
median footprint of 1,951 timesteps and a 75th percentile of 8,828
(docs/HARNESS.md section 7) -- the deviation would have blinded the detector to
almost the whole primary recall set, and the result would have looked like a
weak model rather than a broken thresholder.

**The window is therefore telemanom's, and it trails.** Span 2,100, stride 70 --
but the reference window sits *before* the segment it judges, rather than around
it. telemanom's own windows extend forward, so a timestep there is scored partly
from errors that had not happened yet. `sentinel_models.baselines` states this
repository's position in as many words: *a detector that peeks at future samples
is not something that can fly, and the harness should not measure one that does.*
`rstd`, the number we have to beat, is strictly trailing, and handing the LSTM
2,100 steps of hindsight would not be a comparison.

So each threshold is chosen from the 2,100 errors preceding a 70-step segment and
applied to that segment alone. The mechanism that matters survives intact -- the
reference window at an onset is still mostly nominal, which is exactly why long
events remain visible -- and what it costs is honest: a fixed 70-cycle batching
latency, which delays detection rather than improving it, and which is how the
flight component would have to run anyway. It costs about 112 microseconds per
window, and the sweep skips the ``z`` values that cannot produce an exceedance,
which is free -- they have none by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import ReferenceError

#: telemanom `config.yaml`, transcribed.
SMOOTHING_PERC = 0.05
ERROR_WINDOW_BATCH = 70          # batch_size
ERROR_WINDOW_COUNT = 30          # window_size
ERROR_BUFFER = 100
PRUNING_P = 0.13

#: The z values swept for each window: telemanom's `np.arange(2.5, 12, 0.5)`.
Z_VALUES = np.arange(2.5, 12.0, 0.5)

#: Spans of smoothing to discard at the opening of a series, while the
#: bias-corrected average is still an average of very few samples.
EWMA_SETTLE = 3

#: Used when no z produces a defensible threshold -- telemanom's `sd_lim`, which
#: is where its epsilon starts and stays if nothing improves on it.
Z_LIMIT = 12.0


@dataclass(frozen=True)
class Config:
    """The detection stack's settings. All telemanom's published values."""

    error_window: int = ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT      # h = 2100
    stride: int = ERROR_WINDOW_BATCH                                 # 70; see above
    smoothing_window: int = int(ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT * SMOOTHING_PERC)
    error_buffer: int = ERROR_BUFFER
    pruning_p: float = PRUNING_P

    def as_dict(self) -> dict:
        return {"error_window": self.error_window,
                "error_window_stride": self.stride,
                "smoothing_window": self.smoothing_window,
                "error_buffer": self.error_buffer,
                "pruning_p": self.pruning_p,
                "z_values": [round(float(z), 1) for z in Z_VALUES]}


# -- smoothing --------------------------------------------------------------
@dataclass
class EwmaState:
    """Carries the exponential average across blocks, so blocking is exact."""

    numerator: np.ndarray | None = None
    steps: int = 0


def ewma(values: np.ndarray, span: int, state: EwmaState | None = None) -> np.ndarray:
    """``pandas.DataFrame.ewm(span=...).mean()``, blocked and vectorised.

    telemanom smooths its raw errors this way before thresholding anything, and
    the smoothing is doing real work: a single odd sample is not a fault, and a
    detector that reacts to one is the detector Objective.md 11 rule 2 warns
    about.

    Bias-corrected (pandas' ``adjust=True``): the denominator is the closed form
    ``(1 - r^(t+1)) / a``, so early steps are averaged over what actually exists
    rather than being dragged toward a zero that was never observed. The
    numerator recursion is evaluated in blocks by the scaling identity rather
    than by a Python loop over eleven million steps.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ReferenceError(f"ewma expects (steps, channels), got {values.shape}")
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha

    state = EwmaState() if state is None else state
    if state.numerator is None:
        state.numerator = np.zeros(values.shape[1], dtype=np.float64)

    steps = values.shape[0]
    out = np.empty_like(values)
    # Block short enough that decay ** -block stays far inside float64 range.
    block = max(1, min(steps, 512))
    for lo in range(0, steps, block):
        chunk = values[lo:lo + block]
        n = chunk.shape[0]
        powers = decay ** np.arange(n, dtype=np.float64)
        running = np.cumsum(chunk / powers[:, None], axis=0) * powers[:, None]
        out[lo:lo + n] = running + (decay ** np.arange(1, n + 1))[:, None] * state.numerator
        state.numerator = out[lo + n - 1].copy()

    positions = np.arange(state.steps, state.steps + steps, dtype=np.float64)
    denominator = (1.0 - decay ** (positions + 1.0)) / alpha
    state.steps += steps
    return (out / denominator[:, None]).astype(np.float32)


# -- the nonparametric dynamic threshold ------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _buffered(mask: np.ndarray, buffer: int) -> list[tuple[int, int]]:
    """Widen each exceedance by ``buffer - 1`` steps either side, then merge.

    telemanom adds ``arange(1, error_buffer)`` to every anomalous index and takes
    consecutive groups of the union. Dilating the runs is the same set, computed
    without materialising a hundred indices per exceeded timestep.
    """
    n = mask.shape[0]
    pad = max(0, buffer - 1)
    merged: list[tuple[int, int]] = []
    for lo, hi in _runs(mask):
        lo, hi = max(0, lo - pad), min(n, hi + pad)
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [(lo, hi) for lo, hi in merged if hi - lo > 1]


def dynamic_threshold(e_s: np.ndarray, config: Config
                      ) -> tuple[float, list[tuple[int, int]]]:
    """Choose ``eps`` for one window of smoothed errors, and the sequences above it.

    telemanom's objective: pick the ``eps = mu + z*sigma`` that most reduces the
    mean and standard deviation of what remains, per anomalous point and per
    anomalous *sequence* -- with the sequence count squared, which is what stops
    the sweep from buying a marginal statistical improvement with a shower of
    fragmented detections.

        argmax_z  ( dmu/mu + dsigma/sigma ) / ( |E_seq|^2 + |e_a| )

    If no candidate qualifies, ``eps`` stays at ``mu + 12*sigma`` and nothing is
    reported -- silence, not a guess.
    """
    mu = float(np.mean(e_s))
    sigma = float(np.std(e_s))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0.0:
        return mu + Z_LIMIT * (sigma if np.isfinite(sigma) else 0.0), []

    best_score = -np.inf
    best_eps = mu + Z_LIMIT * sigma
    best_sequences: list[tuple[int, int]] = []

    # A candidate above the window's maximum exceeds nothing, by construction.
    # Skipping those z values is identical arithmetic and three to four times
    # faster, which is what makes telemanom's thirtyfold overlap affordable.
    reach = (float(np.max(e_s)) - mu) / sigma
    for z in Z_VALUES[Z_VALUES <= reach]:
        eps = mu + z * sigma
        above = e_s >= eps
        if not above.any():
            continue
        remainder = e_s[~above]
        if remainder.size == 0:
            continue
        sequences = _buffered(above, config.error_buffer)
        if not sequences:
            continue

        d_mu = (mu - float(np.mean(remainder))) / mu if mu else 0.0
        d_sigma = (sigma - float(np.std(remainder))) / sigma
        covered = sum(hi - lo for lo, hi in sequences)
        score = (d_mu + d_sigma) / (len(sequences) ** 2 + covered)

        # >= rather than >: ties go to the larger z, which is the quieter choice.
        if score >= best_score:
            best_score, best_eps, best_sequences = score, float(eps), sequences
    return best_eps, best_sequences


def prune(e_s: np.ndarray, sequences: list[tuple[int, int]], eps: float,
          p: float = PRUNING_P) -> list[bool]:
    """telemanom's false-positive step. True where a sequence survives.

    Sort the sequence peaks descending, append the largest error that stayed
    *below* the threshold, and walk down the normalised step decreases. A drop of
    more than ``p`` says the sequences above it are genuinely separated from
    everything beneath; anything after the last such drop is reclassified as
    nominal. The effect is that a detector reports the events that stand out, not
    every event that cleared a line.
    """
    if not sequences:
        return []
    peaks = np.array([float(np.max(e_s[lo:hi])) for lo, hi in sequences])
    below = e_s[e_s < eps]
    normal_max = float(np.max(below)) if below.size else 0.0

    order = np.argsort(peaks)[::-1]
    ladder = np.append(peaks[order], normal_max)

    drop: list[int] = []
    for i in range(len(ladder) - 1):
        step = ladder[i] - ladder[i + 1]
        relative = step / ladder[i] if ladder[i] else 0.0
        if relative < p:
            if i < len(order):
                drop.append(int(order[i]))
        else:
            drop = []
    kept = [True] * len(sequences)
    for i in drop:
        kept[i] = False
    return kept


# -- putting a window's verdict back on the timeline -------------------------
def window_ratios(e_s: np.ndarray, config: Config) -> np.ndarray:
    """Score one window of one channel: ``e_s / eps``, pruned sequences suppressed.

    A value at or above 1.0 is what telemanom would have reported as anomalous.
    Everything else keeps its raw ratio, so the ranking a threshold-free metric
    reads is the honest one.
    """
    eps, sequences = dynamic_threshold(e_s, config)
    ratios = e_s / eps if eps > 0 else np.zeros_like(e_s)

    if not sequences:
        return np.minimum(ratios, np.float32(1.0) - np.finfo(np.float32).eps)

    kept = prune(e_s, sequences, eps, config.pruning_p)
    surviving = np.zeros(e_s.shape[0], dtype=bool)
    for keep, (lo, hi) in zip(kept, sequences):
        if keep:
            surviving[lo:hi] = True

    # Anything not in a surviving sequence must land below the operating point.
    # r/(1+r) is below 1 for every r >= 0 and increasing in r, so a suppressed
    # sequence keeps its order against other suppressed values and can never
    # raise an alarm.
    suppressed = ~surviving
    ratios = ratios.copy()
    ratios[suppressed] = ratios[suppressed] / (1.0 + ratios[suppressed])
    # A surviving sequence is an alarm by construction, including the buffered
    # shoulders telemanom adds around the exceedance itself.
    ratios[surviving] = np.maximum(ratios[surviving], 1.0)
    return ratios


def channel_ratios(e_s: np.ndarray, config: Config) -> np.ndarray:
    """Score one channel's whole smoothed-error series with trailing windows.

    Every ``stride`` steps, a threshold is chosen from the ``error_window`` errors
    that came before and applied to the ``stride`` steps that follow. Returns
    ``e_s / eps`` per timestep, forced to at least 1.0 wherever a surviving
    sequence covers the step and pushed below 1.0 everywhere else.

    Trailing rather than centred, for the reason in the module docstring: the
    baseline this has to beat is trailing, and a detector scored on hindsight is
    not one that can fly.
    """
    e_s = np.asarray(e_s, dtype=np.float32)
    steps = e_s.shape[0]
    span, stride = config.error_window, max(1, config.stride)

    raw = np.zeros(steps, dtype=np.float32)
    alarm = np.zeros(steps, dtype=bool)

    for seg_lo in range(0, steps, stride):
        seg_hi = min(seg_lo + stride, steps)
        reference_lo = max(0, seg_lo - span)
        window = e_s[reference_lo:seg_hi]
        offset = seg_lo - reference_lo          # where the judged segment starts

        eps, sequences = dynamic_threshold(window, config)
        if eps > 0:
            raw[seg_lo:seg_hi] = e_s[seg_lo:seg_hi] / eps
        if not sequences:
            continue
        for keep, (lo, hi) in zip(prune(window, sequences, eps, config.pruning_p),
                                  sequences):
            # Only the segment is being judged; the reference window is history
            # and was scored when it was the segment.
            lo, hi = max(lo, offset), min(hi, window.shape[0])
            if keep and hi > lo:
                alarm[reference_lo + lo:reference_lo + hi] = True

    # Anything not inside a surviving sequence must land below the operating
    # point. r/(1+r) is below 1 for every r >= 0 and increasing in r, so a
    # suppressed value keeps its order against other suppressed values and can
    # never raise an alarm -- which is what lets VUS-PR still read a ranking.
    out = raw / (1.0 + raw)
    out[alarm] = np.maximum(raw[alarm], 1.0)

    # The exponential average has not settled over its opening: it starts equal
    # to the raw error, whose variance is far larger than the smoothed series it
    # becomes, so the first samples of any scored window read as an excursion.
    # The bias correction is within 1% of its asymptote after about 2.3 spans;
    # EWMA_SETTLE rounds that up. `ForecastDetector.warmup_steps` already asks
    # the harness to discard a much longer prefix, so this changes nothing in a
    # run -- it is here so the module cannot mislead a caller that has no harness.
    opening = min(EWMA_SETTLE * config.smoothing_window, steps)
    out[:opening] = np.minimum(out[:opening], np.float32(1.0) - np.finfo(np.float32).eps)
    return out


def top_ratios(smoothed: np.ndarray, config: Config, depth: int = 3
               ) -> tuple[np.ndarray, np.ndarray]:
    """The ``depth`` largest per-channel ratios at each timestep, and who was highest.

    Returns ``(top, who)`` with ``top`` shaped ``(depth, steps)`` sorted
    descending down the first axis. ``top[k-1] >= 1`` exactly when at least ``k``
    channels are simultaneously over their own thresholds, which is what makes a
    single pass answer every value of ``k`` -- see :func:`combine`.

    Computed channel by channel and inserted into a running top-``depth`` rather
    than by materialising the full ``(steps, channels)`` ratio matrix, which over
    an eleven-million-step window would be half a gigabyte.
    """
    steps, channels = smoothed.shape
    top = np.full((depth, steps), -np.inf, dtype=np.float32)
    who = np.zeros(steps, dtype=np.int8)

    for c in range(channels):
        incoming = channel_ratios(smoothed[:, c], config)
        who[incoming > top[0]] = c
        for level in range(depth):
            displaced = np.minimum(incoming, top[level])
            np.maximum(top[level], incoming, out=top[level])
            incoming = displaced
    return top, who


def top_columns(matrix: np.ndarray, depth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The ``depth`` largest raw values at each timestep, and which column held the top.

    :func:`top_ratios` does this over per-channel *ratios*, which is what the
    dynamic threshold produces. Quantile-mode detectors threshold the smoothed
    errors directly and need the same reduction over the raw columns, so the
    insertion is factored out rather than duplicated.

    ``np.partition`` would be the obvious one-liner and is refused: it copies its
    input, which over an eleven-million-step window of twelve channels is 530 MB
    allocated beside an 843 MB resident bundle. The running insertion holds
    ``(depth, steps)`` instead -- 132 MB at depth 3 -- and answers every value of
    ``k`` from one pass.
    """
    steps, columns = matrix.shape
    top = np.full((depth, steps), -np.inf, dtype=np.float32)
    who = np.zeros(steps, dtype=np.int8)

    for c in range(columns):
        incoming = np.asarray(matrix[:, c], dtype=np.float32)
        who[incoming > top[0]] = c
        for level in range(depth):
            displaced = np.minimum(incoming, top[level])
            np.maximum(top[level], incoming, out=top[level])
            incoming = displaced
    return top, who


def combine(smoothed: np.ndarray, config: Config, k: int = 1
            ) -> tuple[np.ndarray, np.ndarray]:
    """Reduce every channel to one series, requiring ``k`` of them to agree.

    ``k = 1`` is the maximum: **any one** channel over its threshold raises an
    alarm. That is what this did originally, and on twelve channels it is twelve
    independent chances to be wrong -- measured, 182 alarm ranges against 62 on
    the six-channel subset, which is most of why the primary set scored worse
    than its own subset.

    **The argument for k > 1 is the project's own thesis.** Objective.md 2.4
    defines the target class as one where every channel is individually legal
    while the combination is wrong; a single channel deviating alone is, by that
    definition, not the signal we claim to look for. Maximum-over-channels is a
    per-channel detector wearing a multivariate coat.

    Requiring ``k`` channels over threshold *simultaneously* is exactly the
    ``k``-th largest ratio crossing 1.0, so the whole sweep comes from one
    computation and no timestep is scored twice.

    The attribution stays the strongest channel even when ``k > 1``. It is what
    an operator would look at first, and Objective.md 7's explanation layer names
    a break rather than counting one.
    """
    depth = max(1, int(k))
    top, who = top_ratios(smoothed, config, depth=depth)
    return top[depth - 1], who
