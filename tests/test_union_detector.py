"""`lstm-gru-or` is the OR of its members' alarm masks, and nothing cleverer.

docs/MODELS.md 17 measured the union of `lstm-quantile` and `gru-quantile` as
``max(r_L, r_G) >= 1`` on each member's own bar. D28 named that a post-gate
deployment configuration; the closure run (docs/MODELS.md 18) scores it through
the one tested path, `python -m sentinel_eval run`, so it exists as a detector.
This file pins that the detector's mask is that OR, fold by fold, on the
fixture -- computed here independently from the members themselves.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import harness, tasks
from sentinel_eval.detector import Context, reduce_scores
from sentinel_eval.splits import train_mask
from sentinel_models import detectors as D
from sentinel_models.lstm import Hyper
from sentinel_models.telemanom import Config


def _member(cell):
    kind = {"lstm": D.TelemanomQuantile, "gru": D.GRUQuantile}[cell]
    return kind(hyper=Hyper(window=30, hidden=(8, 8), n_predictions=2, batch_size=16,
                            max_epochs=2, patience=2, sequence_budget_divisor=8,
                            max_validation_sequences=32, cell=cell),
                config=Config(error_window=300, stride=60, smoothing_window=15, error_buffer=10),
                chunks=4, chunk_steps=300)


def _context(bundle, fold, window):
    return Context(mission=bundle.mission, channels=bundle.channel_ids, groups=bundle.groups,
                   period_seconds=bundle.provenance["grid_period_seconds"], fold=fold.index,
                   window=window)


def _mask(detector, bundle, fold):
    """The harness's path, step for step: fit, own threshold, warmed scoring, cut."""
    train_lo, train_hi = fold.train
    test_lo, test_hi = fold.test
    usable = train_mask(fold, bundle.truth)[train_lo:train_hi]
    fitting = _context(bundle, fold, fold.train)
    detector.fit(bundle.values[train_lo:train_hi], usable, fitting)
    raw = detector.score(bundle.values[train_lo:train_hi], bundle.valid[train_lo:train_hi], fitting)
    train_scores, _ = reduce_scores(raw, train_hi - train_lo)
    threshold = detector.threshold_from(train_scores[usable] if usable.any() else train_scores)
    warmup = min(detector.warmup_steps, test_lo)
    scored = _context(bundle, fold, (test_lo, test_hi))
    raw = detector.score(bundle.values[test_lo - warmup:test_hi],
                         bundle.valid[test_lo - warmup:test_hi], scored)
    scores, _ = reduce_scores(raw, test_hi - test_lo + warmup)
    return np.asarray(scores[warmup:] >= threshold, dtype=bool)


@pytest.fixture(scope="module")
def masks(loaded, split):
    D.clear_caches()
    out = []
    for fold in split.folds:
        lstm, gru = _mask(_member("lstm"), loaded, fold), _mask(_member("gru"), loaded, fold)
        union = _mask(D.UnionQuantile(members=(_member("lstm"), _member("gru"))), loaded, fold)
        out.append((lstm, gru, union))
    D.clear_caches()
    return out


def test_the_union_mask_is_the_or_of_its_members(masks):
    for lstm, gru, union in masks:
        assert np.array_equal(union, lstm | gru)
        assert union.sum() >= max(lstm.sum(), gru.sum())


def test_the_union_fires_somewhere_each_member_does_not(masks):
    """Otherwise the fixture cannot tell an OR from a copy of one member."""
    assert any((union & ~lstm).any() or (union & ~gru).any() for lstm, gru, union in masks)


def test_the_cut_is_one_whatever_it_is_handed():
    union = D.UnionQuantile(members=(_member("lstm"), _member("gru")))
    for argument in (np.array([0.0]), np.array([1e9, -1e9]), np.linspace(0, 100, 50),
                     np.array([np.nan, np.inf])):
        assert union.threshold_from(argument) == 1.0


def test_it_scores_through_the_harness_with_both_training_blocks(loaded, split):
    D.clear_caches()
    lstm, gru = _member("lstm"), _member("gru")
    union = D.UnionQuantile(members=(_member("lstm"), _member("gru")))
    record = harness.evaluate(loaded, split, [lstm, gru, union], tasks.get("synthetic"), sweep=False)
    D.clear_caches()
    cards = {card.detector: card for card in record.scorecards}
    assert set(cards) == {"lstm-quantile", "gru-quantile", "lstm-gru-or"}
    for i, fold in enumerate(cards["lstm-gru-or"].folds):
        assert set(fold.training) == {"lstm-quantile", "gru-quantile"}
        assert fold.training["gru-quantile"]["epochs_run"] >= 1
        assert fold.threshold == 1.0
        recall = fold.events.recall.k
        assert recall >= max(cards["lstm-quantile"].folds[i].events.recall.k,
                             cards["gru-quantile"].folds[i].events.recall.k)
        rare = fold.false_alarms.rare_events.k
        assert rare <= (cards["lstm-quantile"].folds[i].false_alarms.rare_events.k
                        + cards["gru-quantile"].folds[i].false_alarms.rare_events.k)


def test_the_registry_builds_it_from_the_flown_members():
    from sentinel_models import registry

    built = registry.build("lstm-gru-or")
    assert built.name == "lstm-gru-or" and built.warmup_steps == D.TelemanomQuantile().warmup_steps
    assert [m.name for m in built.members] == ["lstm-quantile", "gru-quantile"]
    assert built.fingerprint() != D.TelemanomQuantile().fingerprint()
    with pytest.raises(D.ReferenceError, match="two members"):
        D.UnionQuantile(members=(D.TelemanomQuantile(),))
