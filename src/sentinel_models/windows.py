"""Cutting long telemetry into trainable sequences, and putting forecasts back.

Shared by work items 4, 5 and 6. Nothing here knows what an LSTM is; a GRU or a
TCN needs exactly the same two services, and giving them a common implementation
is what keeps the architecture gate a comparison of architectures rather than a
comparison of data pipelines.

**Sequences never cross a discontinuity.** The harness hands `fit` a boolean
``usable`` mask -- inside the training window, not annotated anomalous, not a
communication gap, not an invalid segment, and observed on every channel
(`sentinel_eval.splits.train_mask` composed with the staleness guard in
`sentinel_eval.grid`). A sequence is drawn only from a maximal contiguous run of
that mask, so no training example straddles a gap and none contains a labelled
fault. Objective.md 6.1: a detector taught that a fault is normal is the one
failure this project cannot tolerate.

**Sampling is uniform over start positions, not over runs.** Mission1's usable
runs vary by orders of magnitude in length, so drawing a run first and an offset
second would over-sample the short ones and hand the model a distorted picture
of normality.

**Validation is the chronological tail.** telemanom holds out a random 20%.
Here it is the last 20% of the usable steps, because docs/HARNESS.md section 3
is unconditional -- a model is never fitted on data that follows what it is
scored on -- and early stopping is a fitting decision.
"""
from __future__ import annotations

import numpy as np

from .reference import ReferenceError


#: How long a "recently commanded" trace takes to fall to 1/e. Long enough to
#: outlast a response transient, short enough that overlapping commands stay
#: distinguishable. A model hyperparameter, which is why it lives here.
DEFAULT_DECAY_STEPS = 60

#: Time constants of history used to warm the decay before a window starts. After
#: five the carry is below 1% and the trace is indistinguishable from one computed
#: over the whole mission.
DECAY_WARMUP = 5


def decay(impulses: np.ndarray, steps: int = DEFAULT_DECAY_STEPS) -> np.ndarray:
    """A "recently commanded" trace: each impulse decaying over ``steps``.

    ``y[t] = max(x[t], y[t-1] * r)`` with ``r`` set so the trace falls to 1/e over
    ``steps``. **Maximum rather than sum**, so overlapping commands do not
    accumulate into a level the model has never seen: the feature answers *how
    recently*, and the impulse column beside it already answers *how many*.

    Derived rather than stored (`docs/DECISIONS.md` D11) -- over a whole mission a
    stored float32 trace costs four times the `uint8` impulses it comes from.
    """
    values = np.asarray(impulses, dtype=np.float32)
    rate = np.float32(np.exp(-1.0 / max(1, steps)))
    out = np.empty_like(values)

    # Carries a whole batch at once when given one. The recurrence runs along the
    # time axis and every sequence in a batch is independent of the others, so
    # advancing them together is the same arithmetic -- asserted bit-identical by
    # test, not assumed. It matters: derived per batch inside training, the
    # per-sequence version was 21x slower and became the bottleneck of the
    # commanded arm on a GPU that had made everything else instant.
    carry = np.zeros(values.shape[:-2] + values.shape[-1:], dtype=np.float32)
    for t in range(values.shape[-2]):
        carry = np.maximum(values[..., t, :], carry * rate)
        out[..., t, :] = carry
    return out


def command_features(impulses: np.ndarray, starts: np.ndarray, window: int,
                     decay_steps: int = DEFAULT_DECAY_STEPS) -> np.ndarray:
    """``(batch, window, 2K)`` -- the impulse, and how recently it fired.

    Each window is decayed over a warm-up prefix that is then discarded, so a
    sequence beginning long after a command still sees the tail of it. Computing
    the trace over the whole mission and slicing it would be exact and would cost
    968 MB for eleven commands over 14.7M timesteps; this is bounded by the batch.
    """
    warm = DECAY_WARMUP * decay_steps
    span = warm + window
    offsets = np.arange(span)[None, :]
    rows = np.clip(starts[:, None] - warm + offsets, 0, impulses.shape[0] - 1)

    block = np.asarray(impulses[rows], dtype=np.float32)      # (batch, span, K)
    traced = decay(block, decay_steps)                       # batched, same result
    return np.concatenate([block[:, warm:], traced[:, warm:]], axis=2)


def usable_runs(usable: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """Maximal contiguous ``True`` runs of at least ``minimum`` steps, as [lo, hi)."""
    flat = np.asarray(usable, dtype=bool)
    edges = np.diff(np.concatenate(([0], flat.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]


def split_runs(runs: list[tuple[int, int]], validation_fraction: float
               ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split runs chronologically so the last ``fraction`` of usable steps validates.

    A run straddling the boundary is cut, not assigned wholesale: Mission1's
    longest usable runs span years, and handing one to either side would move the
    boundary by an arbitrary amount.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ReferenceError(f"validation fraction {validation_fraction} is not in (0, 1)")
    total = sum(hi - lo for lo, hi in runs)
    target = total * (1.0 - validation_fraction)

    train, validate, seen = [], [], 0
    for lo, hi in runs:
        length = hi - lo
        if seen >= target:
            validate.append((lo, hi))
        elif seen + length <= target:
            train.append((lo, hi))
        else:
            cut = lo + int(target - seen)
            if cut > lo:
                train.append((lo, cut))
            if hi > cut:
                validate.append((cut, hi))
        seen += length
    return train, validate


class SequenceSampler:
    """Draws ``(inputs, targets)`` batches from a set of runs.

    ``inputs`` is ``(batch, window, channels)`` and ``targets`` is
    ``(batch, n_predictions, channels)`` -- the window, and the block of future
    values immediately after it. That is telemanom's formulation exactly: hide
    the next values and ask the model to guess them (Objective.md 6).
    """

    def __init__(self, values: np.ndarray, runs: list[tuple[int, int]], *,
                 window: int, n_predictions: int,
                 impulses: np.ndarray | None = None,
                 decay_steps: int = DEFAULT_DECAY_STEPS) -> None:
        self.values = values
        self.impulses = impulses
        self.decay_steps = int(decay_steps)
        self.window = int(window)
        self.n_predictions = int(n_predictions)
        span = self.window + self.n_predictions

        self._starts: list[np.ndarray] = []
        counts = []
        for lo, hi in runs:
            available = hi - lo - span + 1
            if available > 0:
                self._starts.append(np.array([lo, available], dtype=np.int64))
                counts.append(available)
        self._offsets = np.cumsum([0] + counts) if counts else np.zeros(1, dtype=np.int64)
        self.n_positions = int(self._offsets[-1])

    def __len__(self) -> int:
        return self.n_positions

    def _resolve(self, picks: np.ndarray) -> np.ndarray:
        """Map positions in [0, n_positions) to absolute start indices."""
        which = np.searchsorted(self._offsets, picks, side="right") - 1
        bases = np.array([run[0] for run in self._starts], dtype=np.int64)
        return bases[which] + (picks - self._offsets[which])

    def draw(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.n_positions == 0:
            raise ReferenceError(
                f"no usable run is long enough for a {self.window}+{self.n_predictions} "
                f"step sequence; the training window is too fragmented to fit on"
            )
        starts = self._resolve(rng.integers(0, self.n_positions, size=n))
        return self.gather(starts)

    def sample_positions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """A fixed set of start indices -- used to pin the validation set."""
        n = min(n, self.n_positions)
        return np.sort(self._resolve(
            rng.choice(self.n_positions, size=n, replace=False)
        ))

    def gather(self, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = starts[:, None] + np.arange(self.window + self.n_predictions)[None, :]
        block = self.values[rows]
        inputs = np.ascontiguousarray(block[:, :self.window])
        targets = np.ascontiguousarray(block[:, self.window:])

        if self.impulses is not None:
            # Commands join the *inputs* and never the targets. The model is asked
            # what the telemetry will do given what was commanded; asking it to
            # forecast the commands would be asking it to predict the ground
            # segment, and a wrong answer there would inflate the loss for a
            # quantity no anomaly can be detected in.
            extra = command_features(self.impulses, starts, self.window,
                                     self.decay_steps)
            inputs = np.ascontiguousarray(np.concatenate([inputs, extra], axis=2))
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():  # noqa: E501
            raise ReferenceError(
                "a sampled sequence contains a non-finite value; the usable mask and "
                "the values disagree, which means the grid's staleness guard was bypassed"
            )
        return inputs, targets


def aggregate_predictions(predictions: np.ndarray) -> np.ndarray:
    """Collapse ``(steps, n_predictions, channels)`` to one forecast per timestep.

    The forecast made at step ``s`` covers ``s+1 .. s+n_predictions``, so every
    timestep is forecast up to ``n_predictions`` times, from that many distances
    away. telemanom averages them, and the averaging is doing real work: an error
    that survives being predicted from ten different starting points is a
    property of the data rather than of one unlucky forward pass.

    The first ``n_predictions`` steps have fewer contributions and are averaged
    over what exists rather than padded, because inventing a forecast is worse
    than admitting a shorter average. The caller discards them anyway -- they sit
    inside the warm-up prefix.
    """
    steps, n_predictions, channels = predictions.shape
    total = np.zeros((steps, channels), dtype=np.float64)
    count = np.zeros((steps, 1), dtype=np.float64)
    for j in range(n_predictions):
        first = 1 + j                       # earliest timestep this offset covers
        if first >= steps:
            break
        total[first:] += predictions[: steps - first, j]
        count[first:] += 1.0
    return (total / np.maximum(count, 1.0)).astype(np.float32)
