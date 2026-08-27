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

from . import oscfar, reference, telemanom
from .lstm import Hyper, train
from .reference import LayerWeights, ReferenceError, Weights
from .windows import DEFAULT_DECAY_STEPS, aggregate_predictions, command_features

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
                    "n_exogenous": np.int64(weights.n_exogenous),
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
            n_channels = int(blob["n_channels"])
            # Inferred when absent, because the field postdates the first files
            # written: the input width is in the weights themselves, so this is
            # exact rather than a guess. A file saved before the exogenous count
            # existed describes a model with none, and one saved after says so.
            exogenous = (int(blob["n_exogenous"]) if "n_exogenous" in blob.files
                         else layers[0].n_in - n_channels)
            weights = Weights(layers=layers, head_w=blob["head_w"],
                              head_b=blob["head_b"],
                              n_channels=n_channels,
                              window=int(blob["window"]),
                              n_predictions=int(blob["n_predictions"]),
                              n_exogenous=exogenous)
            return weights, json.loads(str(blob["report"]))
    except Exception:
        # A truncated or unreadable file is a cache miss, never a wrong answer:
        # the filename is a content digest, so refitting reproduces it exactly.
        #
        # That safety is also how a real defect stayed quiet: `n_exogenous` was
        # not being written, so every commanded model failed to reconstruct here
        # and was silently refitted. Correct, and an hour of wasted fitting. A
        # cache that fails safe still has to be checked.
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

    #: Whether this detector wants telecommands supplied. The composition root
    #: reads it *before* loading, so the bundle knows whether to spend the two
    #: Class B on fetching them. False here: `lstm-telemanom` is the control arm
    #: of the ablation and must see exactly what it has always seen.
    wants_commands = False

    def __init__(self, *, hyper: Hyper | None = None,
                 config: telemanom.Config | None = None, mode: str = NDT,
                 agreement: int = 1,
                 chunks: int = DEFAULT_CHUNKS, chunk_steps: int = DEFAULT_CHUNK_STEPS,
                 decay_steps: int = DEFAULT_DECAY_STEPS,
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
        self.decay_steps = int(decay_steps)
        super().__init__(mode=mode, agreement=self.agreement,
                         **self.hyper.as_dict(), **self.config.as_dict())

        self._weights: Weights | None = None
        self._fill: np.ndarray | None = None
        self._impulses: np.ndarray | None = None
        self._fit_window: tuple[int, int] | None = None
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

        impulses = context.commands if self.wants_commands else None
        if self.wants_commands and impulses is None:
            raise ReferenceError(
                f"{self.name} needs telecommands and the bundle carries none. The "
                f"composition root supplies them when a detector asks; a script "
                f"calling the harness directly must pass telecommands= to "
                f"bundle.load"
            )

        key = (self.hyper.as_dict_key(), context.channels, context.fold,
               context.window, values.shape, _sample_digest(values),
               _digest(usable[::997]),
               None if impulses is None else _sample_digest(impulses))
        reuse = self.reuse_weights and caching()
        digest = _digest(key)
        cached = (_WEIGHTS.get(key) or _load_weights(digest)) if reuse else None
        if cached is None:
            weights, report = train(values, usable, self.hyper, fold=context.fold,
                                    impulses=impulses)
            cached = (weights, report.as_dict())
            if reuse:
                _save_weights(digest, *cached)
        if reuse:
            while len(_WEIGHTS) >= WEIGHT_CACHE_ENTRIES:
                _WEIGHTS.pop(next(iter(_WEIGHTS)))          # oldest first
            _WEIGHTS[key] = cached

        self._weights, self.report = cached
        self._impulses = impulses
        self._fit_window = context.window
        rows = values[usable] if usable.any() else values
        with np.errstate(invalid="ignore"):
            self._fill = np.nanmean(rows, axis=0).astype(np.float32)
        self._fill = np.nan_to_num(self._fill, nan=0.0)

    def score(self, values, valid, context) -> np.ndarray:
        if self._weights is None:
            raise ReferenceError("score() before fit(); the harness always fits first")
        values = np.asarray(values)
        if self.wants_commands:
            # This window's commands, not the fit window's. The harness slices
            # them exactly as it slices the values, warm-up included.
            self._impulses = context.commands

        if self.mode == QUANTILE:
            # Through the cache, exactly as NDT mode is. Reducing straight from
            # `_smoothed_errors` here would recompute the forecast and the
            # smoothing for every point of a decision-layer grid, which is the
            # cost the cache exists to avoid -- and quantile mode is the branch
            # the next grid sweeps.
            top, self.last_attribution = self._tops(values, context)
            combined = top[self.agreement - 1]
        elif context.window == self._fit_window:
            # The harness scores the *fitting* window too, to derive an operating
            # point -- and in NDT mode `threshold_from` returns a fixed 1.0 and
            # never looks at these scores. Running the dynamic threshold over
            # 22.1M steps to produce a number nothing reads is two thirds of a
            # run's scoring cost.
            #
            # What comes back is the smoothed error itself, not a placeholder, so
            # a caller that did consult it would get a real quantity rather than
            # nonsense -- and `test_the_ndt_operating_point_ignores_its_argument`
            # pins the invariant that makes skipping safe.
            smoothed = self._smoothed_errors(values, context)
            combined = smoothed.max(axis=1)
            self.last_attribution = smoothed.argmax(axis=1).astype(np.int8)
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

        if self._weights.n_exogenous:
            # The same two features per command the model was fitted on, derived
            # for exactly these chunks. Each chunk's window begins at
            # `start - window`, which is where its first row came from above.
            features = command_features(
                self._impulses, np.asarray([s - window for s in starts]),
                span, self.decay_steps)
            batch = np.concatenate([batch, features], axis=2)

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


class TelemanomGuarded(ForecastDetector):
    """telemanom's rule with CFAR guard cells. The variant, not the baseline.

    The reference window excludes the segment it judges, so a sustained event
    cannot raise the bar it has to clear. `docs/MODELS.md` deviation 6 records
    that our window is `e_s[seg_lo - 2100 : seg_hi]` -- 2,170 samples with the
    judged segment inside -- which is the guard-cell violation
    `docs/RESEARCH.md` describes from radar practice, now confirmed in our own
    code. Measured as an arm rather than assumed either way.
    """

    name = "lstm-telemanom-guarded"

    def __init__(self, **kwargs) -> None:
        config = kwargs.pop("config", None) or telemanom.Config()
        super().__init__(config=type(config)(**{**vars(config), "guard_segment": True}),
                         **kwargs)


class OSCFARDetector(ForecastDetector):
    """A local order statistic under a nominal floor. `sentinel_models.oscfar`.

    Same forecaster, same cached weights, same folds, same bundle as
    `lstm-telemanom`. **The decision rule is the only thing that differs**, which
    is what made the telemanom/quantile pair informative and is the only way this
    pair can be.

    Pre-registered in `docs/MODELS.md` section 10 before its first run, including
    the condition that falsifies it: if the local term binds in fewer than 10% of
    segments, the floor is doing all the work and this is a rediscovery of
    `lstm-quantile` rather than a new rule.
    """

    name = "lstm-oscfar"

    def __init__(self, *, config: oscfar.Config | None = None, **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__(config=config or oscfar.Config(), mode=NDT, **kwargs)
        self.calibration: oscfar.Calibration | None = None
        self.binding = None

    def fit(self, values, usable, context) -> None:
        # A calibration belongs to the fit that produced it; carrying one across
        # folds would be the stale-checkpoint failure in a different costume.
        self.calibration, self.binding = None, None
        super().fit(values, usable, context)
        # The report dict comes out of the shared weight cache, so it is copied
        # before anything is written into it. `harness._score_fold` captures this
        # object by reference straight after the fit, which is how the binding
        # rate reaches the artifact without the harness learning about this rule.
        self.report = dict(self.report or {})

    def score(self, values, valid, context) -> np.ndarray:
        if context.window == self._fit_window and self.calibration is None:
            # The harness scores the fitting window before the test window, to
            # derive an operating point. That window is normal-only
            # (`splits.train_mask`), so it is exactly the nominal pool both
            # multipliers are fitted on -- and its forecast is being computed
            # here anyway, so the calibration is free.
            nominal = self._smoothed_errors(np.asarray(values), context)
            self.calibration = oscfar.calibrate(nominal, self.config)
        return super().score(values, valid, context)

    def _tops(self, values, context) -> tuple[np.ndarray, np.ndarray]:
        if self.calibration is None:
            raise ReferenceError(
                f"{self.name} scored a window before it was calibrated. The "
                f"harness scores the fitting window first and that is where the "
                f"nominal pool comes from; a caller driving the detector directly "
                f"must do the same."
            )
        key = (_weights_digest(self._weights), context.window, values.shape,
               _sample_digest(values), _digest(sorted(self.config.as_dict().items())),
               _digest(self.calibration.alpha, self.calibration.floor), self.mode)
        cached = _TOPS.get(key)
        if cached is not None:
            return cached

        smoothed = self._smoothed_errors(values, context)
        top, who, binding = oscfar.top_ratios(smoothed, self.config, self.calibration,
                                              depth=MAX_AGREEMENT)
        self.binding = binding
        # docs/MODELS.md section 10.3 names this as the number that falsifies the
        # design, and the first run inferred it from the outcome instead of
        # producing it. A number not read from an artifact is not a number.
        if isinstance(self.report, dict):
            self.report["binding_rate"] = round(float(binding), 6)
            self.report["calibration"] = self.calibration.as_dict()
        del smoothed
        result = (top, who)
        cost = top.nbytes + who.nbytes
        held = sum(a.nbytes + b.nbytes for a, b in _TOPS.values())
        while _TOPS and held + cost > TOPS_CACHE_BYTES:
            evicted = _TOPS.pop(next(iter(_TOPS)))
            held -= evicted[0].nbytes + evicted[1].nbytes
        if cost <= TOPS_CACHE_BYTES:
            _TOPS[key] = result
        return result


class OSCFARGuarded(OSCFARDetector):
    """The proposed rule with CFAR guard cells. The fourth arm."""

    name = "lstm-oscfar-guarded"

    def __init__(self, *, config: oscfar.Config | None = None, **kwargs) -> None:
        config = config or oscfar.Config()
        super().__init__(config=type(config)(**{**vars(config), "guard_segment": True}),
                         **kwargs)


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


class TelemanomCommanded(ForecastDetector):
    """The same detector, with telecommands as exogenous inputs. The other arm.

    telemanom feeds its LSTM telemetry **and** encoded command information
    (`docs/DECISIONS.md` D6); `lstm-telemanom` is what we built without that, and
    this is the reproduction completed. The two differ in **that alone** --
    identical architecture, hyperparameters, folds, seeds and detection stack --
    which is what makes the pair an ablation rather than a comparison.

    No ESA-ADB paper has published this ablation. The delta is the result,
    whichever way it lands, and the magnitude is genuinely unknown: ESA's own
    baselines got *worse* precision when telecommands were added, though they
    cannot exploit them and this can. `docs/RESEARCH.md` marks that as the
    thinnest and most consequential evidence in the project.
    """

    name = "lstm-commanded"
    wants_commands = True


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
