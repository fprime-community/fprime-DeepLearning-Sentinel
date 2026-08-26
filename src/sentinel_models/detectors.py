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
import json
from pathlib import Path

import numpy as np

from sentinel_eval.detector import Detector

from . import reference, telemanom
from .lstm import Hyper, train
from .reference import LayerWeights, ReferenceError, Weights
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

#: Per-channel ratios reduced to their top few, cached across a whole run rather
#: than one window at a time. A decision-layer sweep re-scores the same folds
#: once per grid point while changing nothing upstream of the reduction, so
#: without this the forecaster and the dynamic threshold would be recomputed for
#: every point in the grid -- hours, to answer a question about post-processing.
#: Budgeted rather than counted, because windows differ by a factor of three.
TOPS_CACHE_BYTES = 640 * 1024 * 1024

#: How many channels may be required to agree. Fixed so one pass answers the
#: whole sweep; k above this would need a deeper reduction, not a bigger cache.
MAX_AGREEMENT = 3

#: Where trained weights persist between processes. Rule 1 as stated in
#: docs/HARNESS.md: telemetry never touches this disk, outputs under `runs/` may.
#: A fit costs minutes and a run repeats six of them, so an experiment that only
#: changes the decision layer should not pay for the forecaster again.
#:
#: The stale-checkpoint risk is closed by construction rather than by care: the
#: filename *is* the content digest -- hyperparameters, channel set, fold window
#: and a strided sample of the data -- so weights fitted on anything else cannot
#: be found. `--no-cache` refuses them entirely, and no result enters
#: docs/RESULTS.md until it has been reproduced that way.
WEIGHT_STORE = Path(__file__).resolve().parents[2] / "runs" / "_weights"

#: Set False by `--no-cache`. Reproduce from cold before publishing.
_CACHING = True


def set_caching(enabled: bool) -> None:
    """Turn the weight cache off for a run that must be reproducible from cold."""
    global _CACHING
    _CACHING = bool(enabled)


def caching() -> bool:
    return _CACHING


#: How many fits to keep. `harness.evaluate` loops detectors outermost, so by the
#: time a second detector reaches fold 0 the first has already worked through
#: fold 2 -- a single-entry cache would have been evicted and every fold retrained.
#: Weights are 358 KiB at the production configuration, so holding a run's worth
#: is free and halves the only expensive part of a paired run.
WEIGHT_CACHE_ENTRIES = 8

_WEIGHTS: dict[tuple, tuple[Weights, dict]] = {}
_ERRORS: dict[tuple, np.ndarray] = {}
_TOPS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def clear_caches() -> None:
    """Drop everything held between detectors. Called by tests, never by a run."""
    _WEIGHTS.clear()
    _ERRORS.clear()
    _TOPS.clear()


def _store_path(key: str) -> Path:
    return WEIGHT_STORE / f"{key}.npz"


def _save_weights(key: str, weights: Weights, report: dict) -> None:
    arrays: dict = {"n_channels": np.int64(weights.n_channels),
                    "window": np.int64(weights.window),
                    "n_predictions": np.int64(weights.n_predictions),
                    "n_layers": np.int64(len(weights.layers)),
                    "head_w": weights.head_w, "head_b": weights.head_b,
                    "report": np.array(json.dumps(report))}
    for i, layer in enumerate(weights.layers):
        for name in ("w_ih", "w_hh", "b_ih", "b_hh"):
            arrays[f"l{i}_{name}"] = getattr(layer, name)
    WEIGHT_STORE.mkdir(parents=True, exist_ok=True)
    np.savez(_store_path(key), **arrays)


def _load_weights(key: str) -> tuple[Weights, dict] | None:
    path = _store_path(key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as blob:
            layers = tuple(
                LayerWeights(*(blob[f"l{i}_{name}"]
                               for name in ("w_ih", "w_hh", "b_ih", "b_hh")))
                for i in range(int(blob["n_layers"]))
            )
            weights = Weights(layers=layers, head_w=blob["head_w"],
                              head_b=blob["head_b"],
                              n_channels=int(blob["n_channels"]),
                              window=int(blob["window"]),
                              n_predictions=int(blob["n_predictions"]))
            return weights, json.loads(str(blob["report"]))
    except Exception:
        # A truncated or unreadable file is a cache miss, never a wrong answer:
        # the filename is a content digest, so refitting reproduces it exactly.
        return None


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
                 agreement: int = 1,
                 chunks: int = DEFAULT_CHUNKS, chunk_steps: int = DEFAULT_CHUNK_STEPS,
                 reuse_weights: bool = True) -> None:
        if mode not in (NDT, QUANTILE):
            raise ReferenceError(f"unknown mode {mode!r}; expected {NDT!r} or {QUANTILE!r}")
        if not 1 <= int(agreement) <= MAX_AGREEMENT:
            raise ReferenceError(
                f"agreement must be between 1 and {MAX_AGREEMENT}, got {agreement}"
            )
        self.agreement = int(agreement)
        self.hyper = hyper or Hyper()
        self.config = config or telemanom.Config()
        self.mode = mode
        self.chunks = int(chunks)
        self.chunk_steps = int(chunk_steps)
        self.reuse_weights = reuse_weights
        super().__init__(mode=mode, agreement=self.agreement,
                         **self.hyper.as_dict(), **self.config.as_dict())

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
        """Learn normality from the window the harness stripped of anomalies.

        ``usable`` is intersected with what is actually observed. That should be
        redundant -- `sentinel_eval.bundle.load` folds unobserved steps into
        `truth.unscorable`, and `splits.train_mask` removes them -- but
        `Bundle.subset` rebuilds its truth from the labels alone and never reads
        `self.valid`, so a subset's mask marks gap timesteps usable. The trivial
        baselines never noticed: they are NaN-tolerant by construction. A
        forecaster is not, and a NaN entering the recurrence makes every
        subsequent state NaN.

        Intersecting here is right on its own terms regardless of that: *usable*
        can only mean what the model is able to learn from, and a 1-D mask cannot
        express which of C channels went missing. See docs/MODELS.md section 5.
        """
        self._weights = self._fill = self.report = None
        values = np.asarray(values)
        usable = np.asarray(usable, dtype=bool) & np.isfinite(values).all(axis=1)

        key = (self.hyper.as_dict_key(), context.channels, context.fold,
               context.window, values.shape, _sample_digest(values),
               _digest(usable[::997]))
        reuse = self.reuse_weights and caching()
        digest = _digest(key)
        cached = (_WEIGHTS.get(key) or _load_weights(digest)) if reuse else None
        if cached is None:
            weights, report = train(values, usable, self.hyper, fold=context.fold)
            cached = (weights, report.as_dict())
            if reuse:
                _save_weights(digest, *cached)
        if reuse:
            while len(_WEIGHTS) >= WEIGHT_CACHE_ENTRIES:
                _WEIGHTS.pop(next(iter(_WEIGHTS)))          # oldest first
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

        if self.mode == QUANTILE:
            # Through the cache, exactly as NDT mode is. Reducing straight from
            # `_smoothed_errors` here would recompute the forecast and the
            # smoothing for every point of a decision-layer grid, which is the
            # cost the cache exists to avoid -- and quantile mode is the branch
            # the next grid sweeps.
            top, self.last_attribution = self._tops(values, context)
            combined = top[self.agreement - 1]
        else:
            # The cache is consulted before the errors are computed, not after.
            # Checking afterwards would recompute the forecast and the smoothing
            # for every point of a decision-layer grid -- the expensive two thirds
            # of the work, to answer a question about the cheap third.
            top, self.last_attribution = self._tops(values, context)
            combined = top[self.agreement - 1]

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

    def _tops(self, values, context) -> tuple[np.ndarray, np.ndarray]:
        """The top-``MAX_AGREEMENT`` per-channel ratios, cached across the run.

        Keyed on everything upstream of the reduction and on nothing downstream,
        so every value of ``agreement`` reads the same computation. That is what
        turns a decision-layer grid from hours into one scoring pass.
        """
        key = (_weights_digest(self._weights), context.window, values.shape,
               _sample_digest(values), _digest(sorted(self.config.as_dict().items())),
               self.mode)          # ratios and raw errors are not interchangeable
        cached = _TOPS.get(key)
        if cached is not None:
            return cached

        smoothed = self._smoothed_errors(values, context)
        result = (telemanom.top_columns(smoothed, depth=MAX_AGREEMENT)
                  if self.mode == QUANTILE
                  else telemanom.top_ratios(smoothed, self.config, depth=MAX_AGREEMENT))
        del smoothed
        cost = result[0].nbytes + result[1].nbytes
        held = sum(a.nbytes + b.nbytes for a, b in _TOPS.values())
        while _TOPS and held + cost > TOPS_CACHE_BYTES:
            evicted = _TOPS.pop(next(iter(_TOPS)))          # oldest first
            held -= evicted[0].nbytes + evicted[1].nbytes
        if cost <= TOPS_CACHE_BYTES:
            _TOPS[key] = result
        return result

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
