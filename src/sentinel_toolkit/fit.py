"""One command's worth of work: healthy telemetry in, a `model.bin` out.

**Nothing here trains, forecasts or serialises anything of its own.** The
forecaster is `sentinel_models.detectors.GRUForecastDetector`, which is the
configuration D61 selected and D62 froze; the smoothed residual is its own
`_smoothed_errors`; the statistic is `sentinel_toolkit.statistic`, moved intact
from the arm D65 measured; the bytes are `sentinel_export.write_model`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel_eval.detector import Context
from sentinel_models import lstm, telemanom
from sentinel_models.detectors import GRUForecastDetector

from . import statistic, validate
from .calibrate import DEFAULT_QUANTILE, Calibration, calibrate
from .errors import CalibrationError
from .limits import DEFAULT_EWMA_SPAN, FLIGHT_ERROR_WINDOW
from .spec import to_spec

#: The flown forecaster: the published configuration with the cell swapped,
#: which is what a 32-configuration label-free grid selected on held-out nominal
#: validation error alone (D61). Overridable, because a mission with less data
#: than SMAP/MSL needs a smaller one and would otherwise get nothing.
FLOWN = dict(window=250, hidden=(80, 80), n_predictions=10, max_epochs=35)

#: Half the healthy data trains the forecaster; the rest is scored by a model
#: that never saw it, and is then split again so the cut and the rate that
#: reports on it never come from the same timesteps.
DEFAULT_TRAIN_FRACTION = 0.5


@dataclass(frozen=True)
class FitResult:
    weights: object
    calibration: Calibration
    model_bytes: bytes
    spec: dict
    train_steps: int
    scored_steps: int
    warmup_steps: int
    channel_names: tuple[str, ...]
    hyper: object


def _hyper(window: int, hidden: tuple[int, ...], n_predictions: int,
           max_epochs: int, seed: int) -> lstm.Hyper:
    """(!) `cell` is named explicitly and is never allowed to default.

    `lstm.Hyper.cell` defaults to `"lstm"` and the adopted forecaster is the
    GRU. Defaulting it trains the wrong architecture, and the failure arrives
    at the writer -- after the training cost, not before it. `docs/MODELS.md`
    40.6 registers it as a stop condition.
    """
    return lstm.Hyper(cell="gru", window=window, hidden=tuple(hidden),
                      n_predictions=n_predictions, max_epochs=max_epochs,
                      seed=seed)


def fit_model(values: np.ndarray, *, channel_names=None, window: int = FLOWN["window"],
              hidden=FLOWN["hidden"], n_predictions: int = FLOWN["n_predictions"],
              max_epochs: int = FLOWN["max_epochs"], seed: int = 0,
              quantile: float = DEFAULT_QUANTILE,
              train_fraction: float = DEFAULT_TRAIN_FRACTION,
              ewma_span: int = DEFAULT_EWMA_SPAN, tier: int = 3,
              mission: str = "mission", log=lambda *a: None) -> FitResult:
    """Train, calibrate label-free, and serialise. Zero cloud operations."""
    values = np.asarray(values, dtype=np.float32)
    validate.check_shape(values)
    validate.check_finite(values)
    validate.check_history(values, window, n_predictions)

    steps, channels = values.shape
    names = tuple(channel_names or [f"channel_{i}" for i in range(channels)])
    if len(names) != channels:
        raise CalibrationError(f"{len(names)} channel names for {channels} channels")

    split = int(steps * train_fraction)
    train, scored = values[:split], values[split:]
    validate.check_history(train, window, n_predictions)

    # The statistic's trailing window is not a model-file field: the flight
    # component takes it from `Config::ERROR_WINDOW`. So the warm-up the scored
    # region has to absorb is the model's own window plus that constant.
    warmup = window + FLIGHT_ERROR_WINDOW
    usable = len(scored) - warmup
    half = usable // 2
    validate.check_calibration_window(max(half, 0), max(usable - half, 0))

    hyper = _hyper(window, hidden, n_predictions, max_epochs, seed)
    # (!) `reuse_weights=False`. The detector banks every fit into
    # `runs/_weights/` by default, which is right for the research harness --
    # refitting is expensive and the store is keyed by fit identity -- and wrong
    # here. A mission's fit is not this project's evidence, and a toolkit that
    # grows the research store as a side effect makes the store's own count
    # meaningless as a gate. Measured: two fixture runs took it 1,313 -> 1,315
    # before this line existed.
    detector = GRUForecastDetector(
        hyper=hyper, reuse_weights=False,
        config=telemanom.Config(error_window=FLIGHT_ERROR_WINDOW))
    context = Context(mission=mission, channels=names, groups=(1,) * channels,
                      period_seconds=1, fold=0, window=(0, len(train)))
    log(f"  training on {len(train):,} timesteps x {channels} channels "
        f"(window {window}, hidden {tuple(hidden)}, horizon {n_predictions})")
    detector.fit(train, np.ones(len(train), dtype=bool), context)

    score_context = Context(mission=mission, channels=names, groups=(1,) * channels,
                            period_seconds=1, fold=0, window=(split, steps))
    smoothed = np.asarray(detector._smoothed_errors(scored, score_context))
    fused = statistic.reduce_across_channels(
        statistic.fused_per_channel(smoothed, scored, FLIGHT_ERROR_WINDOW))

    settled = fused[warmup:]
    log(f"  scored {len(scored):,} timesteps, {len(settled):,} after warm-up "
        f"({warmup:,} = window {window} + trailing span {FLIGHT_ERROR_WINDOW:,})")
    calibration = calibrate(settled[:half], settled[half:],
                            span=FLIGHT_ERROR_WINDOW, quantile=quantile)

    spec = to_spec(detector._weights, calibration.cut, warmup,
                   calibration.provenance(), list(names),
                   param_version=2, tier=tier, baseline_only=False,
                   ewma_span=ewma_span)
    from sentinel_export import write_model
    return FitResult(weights=detector._weights, calibration=calibration,
                     model_bytes=write_model(spec), spec=spec,
                     train_steps=len(train), scored_steps=len(settled),
                     warmup_steps=warmup, channel_names=names, hyper=hyper)
