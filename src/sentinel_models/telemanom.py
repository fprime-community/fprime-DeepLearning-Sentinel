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

So each threshold is chosen from a window that ends at the segment it judges,
and applied to that segment alone. **Precisely: the window is
``e_s[seg_lo - 2100 : seg_hi]``, which is 2,170 samples and includes the 70 being
judged.** It has to -- the sequences reported for a segment are found by
:func:`_buffered` over the same array -- but the consequence is worth stating
rather than leaving to be discovered, because ``mu``, ``sigma`` and the
reachability bound are all computed with the judged segment inside them. A
segment therefore contributes about 3.2% of its own threshold.

That is a **guard-cell violation** in the CFAR sense (docs/RESEARCH.md): target
energy in the reference cells raises the threshold and can mask the target, and
the standard answer is to exclude the cells adjacent to the one under test. Under
a sustained event the contribution is not 3.2% of a nominal window, it is a
segment of the event raising the bar the event has to clear. Measured, not
assumed -- see docs/MODELS.md section 10; unmeasured at the time this was
written, and the docstring said "the 2,100 errors preceding", which is what the
code was believed to do rather than what it does.

The mechanism that matters survives intact -- the reference window at an onset is
still mostly nominal, which is exactly why long events remain visible -- and what
it costs is honest: a fixed 70-cycle batching latency, which delays detection
rather than improving it, and which is how the flight component would have to run
anyway. It costs about 112 microseconds per
window, and the sweep skips the ``z`` values that cannot produce an exceedance,
which is free -- they have none by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import ReferenceError

#: telemanom `config.yaml`, transcribed. Read at first hand from
#: `third_party/telemanom/config.yaml`: `batch_size: 70` (:8),
#: `window_size: 30` (:11), `smoothing_perc: 0.05` (:19),
#: `error_buffer: 100` (:22), `p: 0.13` (:53). **`lstm_batch_size: 64` at :30
#: is the trainer's and is NOT this batch size** -- two fields named alike.
SMOOTHING_PERC = 0.05
ERROR_WINDOW_BATCH = 70          # batch_size
ERROR_WINDOW_COUNT = 30          # window_size
ERROR_BUFFER = 100
PRUNING_P = 0.13

#: telemanom's published sweep, `third_party/telemanom/telemanom/errors.py:285`:
#: `for z in np.arange(2.5, self.sd_lim, 0.5)`, with `sd_lim = 12.0` set at
#: `:241`. **A default, not a constant of the method.** z counts standard deviations of the smoothed error,
#: so it is immune to rescaling -- and not immune to a change in the *shape* of
#: the error distribution. Correcting a training defect improved the forecast
#: about fortyfold and this same range then produced 3,548 alarm ranges where it
#: had produced 182: with a poor forecast the residual is dominated by model bias
#: and 2.5 sigma is a real excursion, with a good one it is dominated by noise and
#: 2.5 sigma sits on the floor. See docs/DECISIONS.md, the amendment on
#: dimensionless constants. The floor is therefore fitted per model, not
#: transcribed, which is why it lives on :class:`Config`.
Z_FLOOR = 2.5
Z_CEILING = 12.0
Z_STEP = 0.5

#: Spans of smoothing to discard at the opening of a series, while the
#: bias-corrected average is still an average of very few samples.
EWMA_SETTLE = 3

#: Used when no z produces a defensible threshold -- telemanom's `sd_lim`, which
#: is where its epsilon starts and stays if nothing improves on it. Tied to the
#: sweep's ceiling so raising the range raises the silence fallback with it.
#:
#: (!) UNUSED, and marked rather than deleted (2026-09-11). Nothing in `src/`,
#: `scripts/`, `tests/` or `flight/` references it: :func:`dynamic_threshold`
#: reads ``config.z_ceiling`` directly. Kept because a reader comparing this
#: module against `errors.py:241`'s `sd_lim` will look for exactly this symbol.
Z_LIMIT = Z_CEILING


@dataclass(frozen=True)
class Config:
    """The detection stack's settings. All telemanom's published values."""

    error_window: int = ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT      # h = 2100
    stride: int = ERROR_WINDOW_BATCH                                 # 70; see above
    smoothing_window: int = int(ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT * SMOOTHING_PERC)
    error_buffer: int = ERROR_BUFFER
    pruning_p: float = PRUNING_P
    z_floor: float = Z_FLOOR
    z_ceiling: float = Z_CEILING
    z_step: float = Z_STEP

    #: How the ``n_predictions`` forecasts covering a timestep are collapsed.
    #: ``"mean"`` is what every figure in this repository was measured under;
    #: ``"first"`` is published telemanom, which takes the single one-step-ahead
    #: prediction (`docs/MODELS.md` 28.1, T-a). Lives here rather than on
    #: :class:`~sentinel_models.lstm.Hyper` because it changes the forecast a
    #: fitted model produces and not the model itself, so the two arms share
    #: weights -- the same reason ``error_window`` lives here (26.14).
    aggregate: str = "mean"

    #: Exclude the judged segment from the window the threshold is derived from.
    #: **Default False, which is the behaviour every published number was
    #: measured under.** True is the CFAR guard-cell arrangement: the scale
    #: estimate is taken from reference cells only and then applied to the cells
    #: under test, so a sustained event cannot raise the bar it has to clear.
    #: Measured as an arm of docs/MODELS.md section 10 rather than assumed.
    guard_segment: bool = False

    @property
    def z_values(self) -> np.ndarray:
        """The sweep, from this configuration rather than a module constant."""
        return np.arange(self.z_floor, self.z_ceiling, self.z_step)

    def as_dict(self) -> dict:
        return {"error_window": self.error_window,
                "error_window_stride": self.stride,
                "smoothing_window": self.smoothing_window,
                "error_buffer": self.error_buffer,
                "pruning_p": self.pruning_p,
                "z_floor": self.z_floor, "z_ceiling": self.z_ceiling,
                "z_step": self.z_step, "guard_segment": self.guard_segment,
                # Emitted only when it is not the default, so every recorded
                # config fingerprint is unmoved. D14's rule, applied here.
                **({} if self.aggregate == "mean" else {"aggregate": self.aggregate})}


# -- smoothing --------------------------------------------------------------
@dataclass
class EwmaState:
    """Carries the exponential average across blocks, so blocking is exact."""

    numerator: np.ndarray | None = None
    steps: int = 0


def ewma(values: np.ndarray, span: int, state: EwmaState | None = None) -> np.ndarray:
    """``pandas.DataFrame.ewm(span=...).mean()``, blocked and vectorised.

    `third_party/telemanom/telemanom/errors.py:58`:
    ``pd.DataFrame(self.e).ewm(span=smoothing_window).mean()``, where the span is
    ``int(batch_size * window_size * smoothing_perc)`` at `:51-52` -- 70 * 30 *
    0.05 = 105.

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

    (!) TELEMANOM HAS TWO ASYMMETRIC INDEX-SET FORMS AND THIS TRANSCRIBES THE
    FIRST. `third_party/telemanom/telemanom/errors.py:347-352` adds
    ``np.arange(1, error_buffer + 1)`` on **both** sides of every anomalous index
    and takes consecutive groups of the union -- that is this function. The one at
    `:359` keeps only indices inside the newest ``batch_size`` and is the batch
    clip 26.29.1 records; it is **not** a buffering step and is not implemented
    here. Reading the two as one is how a reproduction goes wrong quietly.

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


def sequences_at(e_s: np.ndarray, eps: float, config: Config) -> list[tuple[int, int]]:
    """The buffered, merged runs of ``e_s`` at or above a threshold chosen elsewhere.

    :func:`dynamic_threshold` derives ``eps`` and finds its sequences in one pass
    over one array, which is right when the two are the same array. Under
    ``guard_segment`` they are not: the threshold comes from reference cells that
    exclude the segment, and is then applied to the segment. Factored out rather
    than duplicated so both paths dilate and merge identically.
    """
    above = np.asarray(e_s) >= eps
    if not above.any():
        return []
    return _buffered(above, config.error_buffer)


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
        return mu + config.z_ceiling * (sigma if np.isfinite(sigma) else 0.0), []

    best_score = -np.inf
    best_eps = mu + config.z_ceiling * sigma
    best_sequences: list[tuple[int, int]] = []

    # A candidate above the window's maximum exceeds nothing, by construction.
    # Skipping those z values is identical arithmetic and three to four times
    # faster, which is what makes telemanom's thirtyfold overlap affordable.
    reach = (float(np.max(e_s)) - mu) / sigma
    candidates = config.z_values
    for z in candidates[candidates <= reach]:
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

    `third_party/telemanom/telemanom/errors.py:386-418`, `Channel.prune_anoms`:
    the peaks at `:403`, sorted descending at `:404`, the largest sub-threshold
    error appended at `:405`, and the ladder walked at `:408-414`. Read line by
    line at `docs/MODELS.md` 37.2 rather than recalled -- `docs/NARRATIVE.md` 11
    records that five recalled readings of this file were later found wrong and
    one cost four pre-registered rungs.

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
    """(!) UNUSED. Marked rather than deleted, 2026-09-11.

    Nothing in `src/`, `scripts/`, `tests/` or `flight/` calls this. It is the
    single-window form of :func:`channel_ratios`, kept because a reader comparing
    this module against `errors.py` will look for exactly that shape -- the
    published code scores one window at a time and the chunked driver is ours.
    It is code rather than a documented figure, so the never-delete rule does not
    bind; it is marked because a reader finding it should know nothing depends on
    it.

    Score one window of one channel: ``e_s / eps``, pruned sequences suppressed.

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


def channel_ratios(e_s: np.ndarray, config: Config,
                   emission: np.ndarray | None = None) -> np.ndarray:
    """Score one channel's whole smoothed-error series with trailing windows.

    Every ``stride`` steps, a threshold is chosen from
    ``e_s[seg_lo - error_window : seg_hi]`` and applied to the ``stride`` steps of
    that segment. **The window spans ``error_window + stride`` samples and
    includes the segment being judged**, which is a guard-cell violation in the
    CFAR sense -- see the module docstring. Returns ``e_s / eps`` per timestep,
    forced to at least 1.0 wherever a surviving sequence covers the step and
    pushed below 1.0 everywhere else.

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

        emitted = False
        if config.guard_segment and offset > 0:
            # Guard cells: the scale estimate sees reference cells only, and the
            # threshold it yields is then applied to the cells under test. The
            # segment can no longer contribute to the bar it has to clear.
            eps, _ = dynamic_threshold(window[:offset], config)
            sequences = sequences_at(window, eps, config)
        else:
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
                # Where the detector could actually SAY something. The alarm above
                # extends `error_buffer - 1` steps backwards from a crossing that
                # has already happened, and a flight component cannot emit
                # retroactively; it emits when the batch containing the crossing is
                # processed, at the end of the segment. Recorded separately so lead
                # time can be measured from a moment that exists.
                if np.any(window[offset:] >= eps):
                    emitted = True
        if emission is not None and emitted:
            emission[seg_hi - 1] = True

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


def top_ratios(smoothed: np.ndarray, config: Config, depth: int = 3,
               emission: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
        incoming = channel_ratios(smoothed[:, c], config, emission=emission)
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
