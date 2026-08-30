"""The gate that must pass before a single R2 operation is spent.

The synthetic bucket is built with genuine cross-channel structure -- one latent
driver and per-channel lags (`sentinel_eval.synthetic`) -- specifically so that a
detector blind to it underperforms rather than merely scoring lower. A
forecaster that models the whole channel set should therefore beat a per-channel
moving average and a per-channel rolling standard deviation on it.

If it does not, the fault is in the model or the wiring, and finding that out
costs nothing here and fifteen Class B operations on the real bucket. So this
runs first, and it runs offline.

Deliberately a floor and not a target: the assertion is that the GRU wins, not
that it wins by a particular margin, because a margin measured on generated data
is not evidence about ESA-ADB and would only invite tuning against the fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from sentinel_eval import bundle as bundle_mod
from sentinel_eval import harness, splits, synthetic, tasks
from sentinel_eval.catalog import Catalog
from sentinel_eval.labels import LabelSet
from sentinel_eval.read import read_annotation
from sentinel_models import baselines, detectors

STEPS = 40_000


@pytest.fixture(scope="module")
def scored():
    """One fixture load, three detectors, three folds. Zero R2 operations."""
    bucket = synthetic.build(seed=0, n=STEPS)
    catalog = Catalog.load(bucket)
    labels = LabelSet.from_table(read_annotation(bucket, catalog, "labels"))
    task = tasks.get("synthetic")
    loaded = bundle_mod.load(bucket, catalog, labels, mission="missionX",
                             channel_ids=task.selection.resolve(catalog))
    split = splits.forward_chaining(len(loaded.grid), seed_fraction=0.25, folds=3)

    detectors.clear_caches()
    record = harness.evaluate(
        loaded, split,
        [detectors.GRUSmoke(), baselines.RollingStd(120), baselines.MovingAverage(120)],
        task, sweep=False)
    detectors.clear_caches()
    return {card.detector: card.pooled for card in record.scorecards}


def _gate(pooled) -> float:
    """F0.5, with undefined read as zero -- a detector that never fires scores nothing."""
    value = pooled.get("event_f0.5")
    return 0.0 if value is None else float(value)


def test_the_forecaster_beats_both_trivial_baselines(scored):
    lstm = _gate(scored["gru-smoke"])
    for name in ("rstd", "mavg"):
        assert lstm > _gate(scored[name]), (
            f"gru-smoke F0.5 {lstm:.3f} did not beat {name} "
            f"{_gate(scored[name]):.3f} on data built to reward cross-channel "
            f"modelling. Diagnose here, before spending operations on R2."
        )


def test_it_recalls_the_cross_channel_class(scored):
    """The headline cell is what the project's claim rests on."""
    recall = scored["gru-smoke"]["headline_cell_recall"]
    assert recall.n > 0 and recall.k > 0, f"headline-cell recall {recall.brief()}"


def test_it_fires_at_all_and_does_not_fire_constantly(scored):
    """Both failure modes in one assertion: silence, and carpet-bombing."""
    precision = scored["gru-smoke"]["event_precision"]
    assert precision.n > 0, "the detector never raised an alarm"
    assert precision.rate > 0.05, f"precision {precision.brief()} is carpet-bombing"


def test_the_denominators_are_the_same_for_every_detector(scored):
    """The referee treated them identically; otherwise the comparison is void."""
    counts = {name: pooled["event_recall"].n for name, pooled in scored.items()}
    assert len(set(counts.values())) == 1, counts
