"""The detector the harness actually holds: forecaster plus detection stack.

`sentinel_eval` knows one thing about a model -- ``fit`` and ``score``
(`sentinel_eval.detector`). This module is where telemanom's forecaster
(`lstm.py`, scored through `reference.py`) meets telemanom's thresholding
(`telemanom.py`) and comes out the other side as that contract. Items 5 and 6
replace the first half and keep the second.

Three things the harness's shape forces, none of them optional:

**A fit must rebuild everything.** `cli.py` constructs one detector object and
reuses it across three folds *and* both paired channel sets -- twelve channels,
then six. Carrying anything over would either crash on the width change or,
far worse, score fold 2 with fold 1's model.

**A score must survive an eleven-million-step window.** The harness scores the
whole training window as well, to derive an operating point, so this runs over
33.1M timesteps per channel set. It is done in blocks whose width is a whole
number of telemanom error windows, so blocking changes no arithmetic, and each
block's forecast is computed as a batch of chunks warmed by a 250-step prefix --
which is precisely the history telemanom's own windowed inference sees.

**A score must be one-dimensional.** The contract permits ``(T, C)``, and
`reduce_scores` would then cast it to float64: over an 11M-step window that is a
gigabyte, allocated twice. So the reduction across channels happens here, and the
attribution -- which channel diverged, the thing Objective.md 7's explanation
layer needs -- is kept beside it as one byte per step rather than eight.

Weights are cached in memory and keyed by content, so scoring `lstm-telemanom`
and `lstm-quantile` in one invocation trains once rather than twice. Nothing
touches disk: Rule 1, and `tests/test_no_local_persistence.py`.
"""
from __future__ import annotations

import hashlib

import numpy as np

from sentinel_eval.detector import Detector

from . import reference, telemanom
from .lstm import Hyper, train
from .reference import ReferenceError, Weights
from .windows import aggregate_predictions

#: How the score is turned into an alarm. ``ndt`` folds telemanom's moving
#: threshold into the score so the harness's fixed cut reproduces it; ``quantile``
#: hands the harness the raw smoothed error and lets it pick the 99.9th
#: percentile of training scores, exactly as the trivial baselines are treated.
NDT = "ndt"
QUANTILE = "quantile"

#: Sub-chunks run as one batch, each covering this many steps. The product is the
#: block size, and the chunk length is a whole number of error windows so that
#: forecasting and thresholding tile the timeline identically.
DEFAULT_CHUNKS = 64
DEFAULT_CHUNK_STEPS = telemanom.ERROR_WINDOW_BATCH * telemanom.ERROR_WINDOW_COUNT

#: Smoothed errors are cached only while they fit in this much memory, so a
#: 10.8M-step training window does not sit in RAM beside the resident bundle.
ERROR_CACHE_BYTES = 256 * 1024 * 1024

_WEIGHTS: dict[tuple, tuple[Weights, dict]] = {}
_ERRORS: dict[tuple, np.ndarray] = {}


def clear_caches() -> None:
    """Drop everything held between detectors. Called by tests, never by a run."""
    _WEIGHTS.clear()
    _ERRORS.clear()


def _digest(*parts) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            hasher.update(str(part.shape).encode())
            hasher.update(np.ascontiguousarray(part).tobytes())
        else:
            hasher.update(repr(part).encode())
    return hasher.hexdigest()


def _sample_digest(values: np.ndarray, stride: int = 997) -> str:
    """Cheap content fingerprint. Strided, because hashing 530 MB per call is not."""
    return _digest(values.shape, values.dtype.str, values[::stride], values[:1], values[-1:])


def _weights_digest(weights: Weights) -> str:
    arrays = [a for layer in weights.layers
              for a in (layer.w_ih, layer.w_hh, layer.b_ih, layer.b_hh)]
    return _digest(*arrays, weights.head_w, weights.head_b)


class ForecastDetector(Detector):
    """A forecaster plus telemanom's detection stack, as one scoreable object."""

    name = "lstm-telemanom"

    def __init__(self, *, hyper: Hyper | None = None,
                 config: telemanom.Config | None = None, mode: str = NDT,
                 chunks: int = DEFAULT_CHUNKS, chunk_steps: int = DEFAULT_CHUNK_STEPS,
                 reuse_weights: bool = True) -> None:
        if mode not in (NDT, QUANTILE):
            raise ReferenceError(f"unknown mode {mode!r}; expected {NDT!r} or {QUANTILE!r}")
        self.hyper = hyper or Hyper()
        self.config = config or telemanom.Config()
        self.mode = mode
        self.chunks = int(chunks)
        self.chunk_steps = int(chunk_steps)
        self.reuse_weights = reuse_weights
        super().__init__(mode=mode, **self.hyper.as_dict(), **self.config.as_dict())

        self._weights: Weights | None = None
        self._fill: np.ndarray | None = None
        self.report: dict | None = None
        self.last_attribution: np.ndarray | None = None

    # -- contract ----------------------------------------------------------
    @property
    def warmup_steps(self) -> int:
        """Enough history for both the recurrence and the first error window.

        A fold scored from its first timestep would otherwise open with a cold
        LSTM state *and* a threshold derived from errors that do not exist yet,
        and a fold boundary would look exactly like an anomaly.
        """
        return self.hyper.window + self.config.error_window

    def fit(self, values, usable, context) -> None:
        self._weights = self._fill = self.report = None
        values = np.asarray(values)
        usable = np.asarray(usable, dtype=bool)

        key = (self.hyper.as_dict_key(), context.channels, context.fold,
               context.window, values.shape, _sample_digest(values),
               _digest(usable[::997]))
        cached = _WEIGHTS.get(key) if self.reuse_weights else None
        if cached is None:
            weights, report = train(values, usable, self.hyper, fold=context.fold)
            cached = (weights, report.as_dict())
            if self.reuse_weights:
                _WEIGHTS.clear()          # one entry: the fold being worked on
                _WEIGHTS[key] = cached

        self._weights, self.report = cached
        rows = values[usable] if usable.any() else values
        with np.errstate(invalid="ignore"):
            self._fill = np.nanmean(rows, axis=0).astype(np.float32)
        self._fill = np.nan_to_num(self._fill, nan=0.0)

    def score(self, values, valid, context) -> np.ndarray:
        if self._weights is None:
            raise ReferenceError("score() before fit(); the harness always fits first")
        values = np.asarray(values)
        smoothed = self._smoothed_errors(values, context)

        if self.mode == QUANTILE:
            combined = smoothed.max(axis=1)
            self.last_attribution = smoothed.argmax(axis=1).astype(np.int8)
        else:
            combined, self.last_attribution = telemanom.combine(smoothed, self.config)

        observed = np.isfinite(np.asarray(values, dtype=np.float32)).all(axis=1)
        if valid is not None:
            observed &= np.asarray(valid, dtype=bool).all(axis=1)
        combined = combined.astype(np.float64)
        combined[~observed] = -np.inf      # nothing measured is never an alarm
        return combined

    def threshold_from(self, train_scores):
        """NDT already chose its operating point; the harness's cut is 1.0.

        telemanom's threshold moves with the data, which the contract's single
        scalar cannot express -- so it lives inside the score instead, and this
        returns the fixed point at which that score means "anomalous". In
        ``quantile`` mode the base class's label-free 99.9th percentile applies
        unchanged, which is the whole purpose of that variant.
        """
        if self.mode == NDT:
            return 1.0
        return super().threshold_from(train_scores)

    # -- forecasting -------------------------------------------------------
    def _smoothed_errors(self, values: np.ndarray, context) -> np.ndarray:
        key = (_weights_digest(self._weights), context.window, values.shape,
               _sample_digest(values), self.config.smoothing_window)
        cached = _ERRORS.get(key)
        if cached is not None:
            return cached

        filled = self._filled(values)
        steps, channels = filled.shape
        smoothed = np.empty((steps, channels), dtype=np.float32)
        ewma_state = telemanom.EwmaState()
        block = self.chunks * self.chunk_steps

        for lo in range(0, steps, block):
            hi = min(lo + block, steps)
            forecast = self._forecast(filled, lo, hi)
            errors = np.abs(filled[lo:hi] - forecast)
            smoothed[lo:hi] = telemanom.ewma(errors, self.config.smoothing_window,
                                             ewma_state)

        if smoothed.nbytes <= ERROR_CACHE_BYTES:
            _ERRORS.clear()
            _ERRORS[key] = smoothed
        return smoothed

    def _forecast(self, filled: np.ndarray, lo: int, hi: int) -> np.ndarray:
        """One-step-ahead forecast for ``[lo, hi)``, as a batch of warmed chunks.

        Every chunk is preceded by ``window`` real timesteps and starts from a
        zero state, so it sees exactly the history a telemanom window sees. The
        forecast for timestep ``t`` is built only from inputs strictly before
        ``t`` -- :func:`~sentinel_models.windows.aggregate_predictions` averages
        predictions made at ``t-1`` and earlier -- so nothing here can peek.
        """
        window = self.hyper.window
        starts = list(range(lo, hi, self.chunk_steps))
        lengths = [min(self.chunk_steps, hi - s) for s in starts]
        span = window + max(lengths)

        batch = np.empty((len(starts), span, filled.shape[1]), dtype=np.float32)
        for i, start in enumerate(starts):
            front = max(0, window - start)          # history that does not exist yet
            take = filled[start - window + front:start + lengths[i]]
            back = span - front - take.shape[0]     # a short final chunk
            pieces = [np.repeat(take[:1], front, axis=0)] if front else []
            pieces.append(take)
            if back:
                pieces.append(np.repeat(take[-1:], back, axis=0))
            batch[i] = np.concatenate(pieces) if len(pieces) > 1 else take
            # Whatever the padding, absolute step `start` lands at index `window`.

        predictions, _ = reference.forward(self._weights, batch)
        out = np.empty((hi - lo, filled.shape[1]), dtype=np.float32)
        for i, start in enumerate(starts):
            aggregated = aggregate_predictions(predictions[i])
            out[start - lo:start - lo + lengths[i]] = aggregated[window:window + lengths[i]]
        return out

    def _filled(self, values: np.ndarray) -> np.ndarray:
        """Replace unobserved values so a gap cannot poison the recurrent state.

        Held forward from the last observation, and from the training-window mean
        where nothing precedes it. The steps concerned are excluded from the
        metrics by the harness anyway (they are unscorable), and their scores are
        forced to ``-inf`` in :meth:`score`; the fill exists only so that the
        arithmetic downstream stays finite.
        """
        values = np.asarray(values, dtype=np.float32)
        if np.isfinite(values).all():
            return np.ascontiguousarray(values)

        filled = values.copy()
        missing = ~np.isfinite(filled)
        filled[missing] = np.nan
        for c in range(filled.shape[1]):
            column = filled[:, c]
            gaps = np.isnan(column)
            if not gaps.any():
                continue
            index = np.where(~gaps, np.arange(column.shape[0]), 0)
            np.maximum.accumulate(index, out=index)
            column[:] = column[index]
            column[np.isnan(column)] = self._fill[c]
        return filled


class TelemanomQuantile(ForecastDetector):
    """Same forecaster and cached weights; the harness picks the threshold.

    The diagnostic that separates *the forecaster is weak* from *the thresholder
    is wrong*. If this scores far better than `lstm-telemanom`, the nonparametric
    dynamic threshold is not earning its place on ESA-ADB and we report that
    rather than quietly keeping the winner.
    """

    name = "lstm-quantile"

    def __init__(self, **kwargs) -> None:
        super().__init__(mode=QUANTILE, **kwargs)


class TelemanomSmoke(ForecastDetector):
    """A deliberately tiny configuration for the fixture and for development.

    **Never a result.** It exists so that the whole pipeline -- sample, train,
    forecast, smooth, threshold, prune, score -- runs end to end in seconds
    against generated data at zero R2 operations.
    """

    name = "lstm-smoke"

    def __init__(self, **kwargs) -> None:
        super().__init__(
            hyper=Hyper(window=60, hidden=(24, 24), n_predictions=5, batch_size=32,
                        max_epochs=8, patience=3, sequence_budget_divisor=4,
                        max_validation_sequences=256),
            config=telemanom.Config(error_window=420, stride=14, smoothing_window=21,
                                    error_buffer=20),
            chunk_steps=420, chunks=16, **kwargs)
