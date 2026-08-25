"""Chronological splits, with the denominators and the contamination on show.

Two policies, both strictly forward in time -- a model is never fitted on data
that follows what it is scored on:

``chronological``      one boundary. Train ``[0, b)``, test ``[b, n)``.
``forward_chaining``   a seed window, then K test folds, each fitted on
                       everything before it. The union of the test folds equals
                       the single split's test side, so the denominators match,
                       while each fold trains on more history than the last --
                       which also yields a data-sufficiency curve for free.

**Where the boundary goes is not a detail.** Measured on `m1-ss5`, moving from
50/50 to 25/75 costs 3.5 training years and takes the test side from 29 anomalies
to 42, from 21 headline-cell events to 31, and from 8 point anomalies to **all
11**. Meanwhile `mission1/subsystem_3` at a 75% boundary leaves *zero* test
anomalies -- a run that would report a confident, perfectly meaningless number.
Hence :class:`~sentinel_eval.errors.SplitTooThin`: below a floor of scorable
events the harness refuses rather than scores.

**Training is normal-only** (Objective.md 6.1). A spacecraft has no failure
examples, so the model learns normality and treats everything else as suspicious.
:func:`train_mask` removes annotated anomalies from the fitting window, and how
much was removed is reported rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import SplitTooThin
from .labels import ANOMALY, HEADLINE_CELL, RARE_EVENT, Truth

#: Fewer scorable anomalies than this and a comparison is noise, not evidence.
MIN_TEST_ANOMALY_EVENTS = 5


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[int, int]
    test: tuple[int, int]

    @property
    def train_steps(self) -> int:
        return self.train[1] - self.train[0]

    @property
    def test_steps(self) -> int:
        return self.test[1] - self.test[0]


@dataclass(frozen=True)
class Split:
    policy: str
    params: dict
    folds: tuple[Fold, ...]

    def __len__(self) -> int:
        return len(self.folds)

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.policy}({parts}) -> {len(self.folds)} fold(s)"


def chronological(n: int, *, fraction: float = 0.25) -> Split:
    """One boundary at ``fraction`` of the timeline. Train first, test after."""
    boundary = int(round(n * fraction))
    if boundary < 1 or boundary >= n:
        raise SplitTooThin(f"boundary {boundary} leaves no data on one side of {n} steps")
    return Split("chronological", {"fraction": fraction},
                 (Fold(0, (0, boundary), (boundary, n)),))


def forward_chaining(n: int, *, seed_fraction: float = 0.25, folds: int = 3) -> Split:
    """A seed window, then ``folds`` test blocks, each fitted on all prior data."""
    if folds < 1:
        raise SplitTooThin("forward chaining needs at least one fold")
    edges = [int(round(n * (seed_fraction + (1 - seed_fraction) * i / folds)))
             for i in range(folds + 1)]
    out = []
    for i in range(folds):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            raise SplitTooThin(
                f"fold {i} is empty: {n} steps will not divide into {folds} folds "
                f"after a {seed_fraction:.0%} seed"
            )
        out.append(Fold(i, (0, lo), (lo, hi)))
    return Split("forward_chaining", {"seed_fraction": seed_fraction, "folds": folds},
                 tuple(out))


# --------------------------------------------------------------------------
def train_mask(fold: Fold, truth: Truth) -> np.ndarray:
    """Timesteps usable for fitting: nominal, present, and inside the train window.

    Annotated anomalies are removed, because a detector taught that a fault is
    normal is the one failure this project cannot tolerate. Gaps and invalid
    segments go too -- there is nothing there to learn from.
    """
    lo, hi = fold.train
    usable = np.zeros(truth.anomaly.shape[0], dtype=bool)
    usable[lo:hi] = True
    return usable & ~truth.anomaly & ~truth.unscorable


@dataclass(frozen=True)
class Coverage:
    """What a fold actually holds. Every figure a count, never a rate."""

    fold: int
    train_steps: int
    train_usable_steps: int
    train_anomaly_steps: int
    train_rare_events: int
    test_steps: int
    test_scorable_steps: int
    test_anomalies: int
    test_contextual: int
    test_headline_cell: int
    test_point: int
    test_rare_events: int

    @property
    def masked_steps(self) -> int:
        return self.train_steps - self.train_usable_steps


def coverage(split: Split, truth: Truth) -> list[Coverage]:
    """Per-fold census. An event belongs to the fold its span starts in."""
    out = []
    for fold in split.folds:
        lo, hi = fold.test
        usable = train_mask(fold, truth)

        def starts_in(event, a=lo, b=hi) -> bool:
            span = truth.spans.get(event.event_id)
            return span is not None and a <= span[0] < b

        anomalies = [e for e in truth.of(ANOMALY) if starts_in(e)]
        rare_test = [e for e in truth.of(RARE_EVENT) if starts_in(e)]
        rare_train = [
            e for e in truth.of(RARE_EVENT)
            if (s := truth.spans.get(e.event_id)) and fold.train[0] <= s[0] < fold.train[1]
        ]
        out.append(Coverage(
            fold=fold.index,
            train_steps=fold.train_steps,
            train_usable_steps=int(usable[fold.train[0]:fold.train[1]].sum()),
            train_anomaly_steps=int(truth.anomaly[fold.train[0]:fold.train[1]].sum()),
            train_rare_events=len(rare_train),
            test_steps=fold.test_steps,
            test_scorable_steps=int(truth.scorable[lo:hi].sum()),
            test_anomalies=len(anomalies),
            test_contextual=sum(1 for e in anomalies if e.contextual),
            test_headline_cell=sum(1 for e in anomalies if e.cell == HEADLINE_CELL),
            test_point=sum(1 for e in anomalies if e.is_point),
            test_rare_events=len(rare_test),
        ))
    return out


def check(split: Split, truth: Truth, *, minimum: int = MIN_TEST_ANOMALY_EVENTS
          ) -> list[Coverage]:
    """Census the split, refusing one too thin to carry a comparison."""
    rows = coverage(split, truth)
    pooled = sum(r.test_anomalies for r in rows)
    if pooled < minimum:
        raise SplitTooThin(
            f"{split.describe()} leaves {pooled} scorable anomal{'y' if pooled == 1 else 'ies'} "
            f"across {len(rows)} fold(s); {minimum} is the floor.\n"
            f"  A recall over {pooled} events has {pooled + 1} possible values, so any "
            f"difference between detectors would be indistinguishable from noise.\n"
            f"  Move the boundary earlier, or choose a channel set with more anomalies."
        )
    return rows


def render(rows: list[Coverage]) -> str:
    """The table printed before a run, so denominators are seen before results."""
    head = (f"    {'fold':>4}  {'train':>9} {'usable':>9} {'masked':>8} {'rare':>5} | "
            f"{'test':>9} {'anom':>5} {'ctx':>4} {'MVGS':>5} {'pt':>3} {'rare':>5}")
    lines = [head, "    " + "-" * (len(head) - 4)]
    for r in rows:
        lines.append(
            f"    {r.fold:>4}  {r.train_steps:>9,} {r.train_usable_steps:>9,} "
            f"{r.masked_steps:>8,} {r.train_rare_events:>5} | {r.test_steps:>9,} "
            f"{r.test_anomalies:>5} {r.test_contextual:>4} {r.test_headline_cell:>5} "
            f"{r.test_point:>3} {r.test_rare_events:>5}"
        )
    if len(rows) > 1:
        lines.append(
            f"    {'all':>4}  {'':>9} {'':>9} {'':>8} {'':>5} | "
            f"{sum(r.test_steps for r in rows):>9,} "
            f"{sum(r.test_anomalies for r in rows):>5} "
            f"{sum(r.test_contextual for r in rows):>4} "
            f"{sum(r.test_headline_cell for r in rows):>5} "
            f"{sum(r.test_point for r in rows):>3} "
            f"{sum(r.test_rare_events for r in rows):>5}"
        )
    return "\n".join(lines)
