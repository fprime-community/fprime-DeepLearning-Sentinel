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
                 window: int, n_predictions: int) -> None:
        self.values = values
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
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
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
