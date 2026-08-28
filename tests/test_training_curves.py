"""Training histories are part of the scorecard now, because they were the signal.

The defect that disabled training for every model in the project announced itself
in exactly one place: eleven epochs with `best_epoch = 0`, on every fit, every
time. Nothing was reading it. Had it been on the scorecard it would have been
visible from the first run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel_eval import harness, splits, tasks
from sentinel_eval.metrics.counts import Count
from sentinel_eval.metrics.falsealarm import FalseAlarmScore
from sentinel_eval.scorecard import FoldResult
from sentinel_models import baselines


def _bare_fold(**extra) -> FoldResult:
    return FoldResult(fold=0, window=(0, 10), threshold=1.0, events=None,
                      false_alarms=FalseAlarmScore(Count(0, 0), Count(0, 0), None, 0),
                      vus_pr=None, vus_detail={}, oracle_f_beta=None,
                      oracle_threshold=None, **extra)


def test_a_detector_that_does_not_train_carries_no_training_block(loaded, split):
    """Additive: the trivial baselines' scorecards are unchanged."""
    record = harness.evaluate(loaded, split, [baselines.MovingAverage(60)],
                              tasks.get("synthetic"), sweep=False)
    payload = json.loads(record.to_json())
    for card in payload["scorecards"]:
        for fold in card["folds"]:
            assert "training" not in fold


def test_it_is_absent_rather_than_null():
    """Byte-identical to a scorecard produced before the field existed."""
    assert "training" not in _bare_fold().as_dict()


@pytest.mark.parametrize("cell", ("lstm", "gru"))
def test_a_training_report_reaches_the_scorecard(loaded, split, cell):
    from sentinel_models import detectors as D
    from sentinel_models.lstm import Hyper
    from sentinel_models.telemanom import Config

    D.clear_caches()
    kind = {"lstm": D.ForecastDetector, "gru": D.GRUForecastDetector}[cell]
    detector = kind(
        hyper=Hyper(window=30, hidden=(8, 8), n_predictions=2, batch_size=16,
                    max_epochs=3, patience=2, sequence_budget_divisor=8,
                    max_validation_sequences=32, cell=cell),
        config=Config(error_window=300, stride=60, smoothing_window=15, error_buffer=10),
        chunks=4, chunk_steps=300)
    record = harness.evaluate(loaded, split, [detector], tasks.get("synthetic"),
                              sweep=False)
    D.clear_caches()

    folds = json.loads(record.to_json())["scorecards"][0]["folds"]
    for fold in folds:
        report = fold["training"]
        assert report["epochs_run"] >= 1
        assert "best_epoch" in report and "validation_history" in report
        assert len(report["validation_history"]) == report["epochs_run"]


def test_the_render_calls_out_a_fit_that_kept_its_first_epoch():
    """The signature that went unread for three work items, made unmissable."""
    from sentinel_eval.scorecard import RunRecord, Scorecard

    card = Scorecard(detector="x", detector_params={}, fingerprint="abc")
    card.folds = [_bare_fold(training={"epochs_run": 11, "best_epoch": 0,
                                       "best_validation_mse": 1.7e-4,
                                       "sequences_per_epoch": 20_064})]
    rendered = "\n".join(RunRecord(task={}, bundle={}, coverage=[],
                                   scorecards=[card])._render_scores(card))
    assert "KEPT THE FIRST EPOCH" in rendered
    assert "11 epochs" in rendered


def test_a_healthy_fit_is_not_flagged():
    from sentinel_eval.scorecard import RunRecord, Scorecard

    card = Scorecard(detector="x", detector_params={}, fingerprint="abc")
    card.folds = [_bare_fold(training={"epochs_run": 22, "best_epoch": 17,
                                       "best_validation_mse": 3.1e-5,
                                       "sequences_per_epoch": 20_064})]
    rendered = "\n".join(RunRecord(task={}, bundle={}, coverage=[],
                                   scorecards=[card])._render_scores(card))
    assert "KEPT THE FIRST EPOCH" not in rendered
    assert "best epoch 17" in rendered
