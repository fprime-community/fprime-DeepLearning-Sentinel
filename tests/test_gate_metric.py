"""The gate number is F0.5. Bare recall is not renderable as a headline.

Recall alone is satisfiable by carpet-bombing: the trivial baseline reached 29/31
on `m1-ss5` by firing 5,535 alarms for 42 events. A detector that fires
constantly detects everything and is worth nothing, so recall is a necessary
component and never a headline.
"""
from __future__ import annotations

from sentinel_eval import harness, tasks
from sentinel_eval.metrics.counts import Count
from sentinel_models import baselines


def _render(loaded, split, detector):
    record = harness.evaluate(loaded, split, [detector], tasks.get("synthetic"),
                              sweep=False)
    return record.render(), record


def test_f_beta_is_printed_before_any_recall(loaded, split):
    text, _ = _render(loaded, split, baselines.MovingAverage(60))
    body = text[text.index("DETECTOR"):]
    assert "F0.5" in body and "recall" in body
    assert body.index("F0.5") < body.index("recall")


def test_recall_never_appears_without_its_precision(loaded, split):
    text, _ = _render(loaded, split, baselines.MovingAverage(60))
    for line in text.splitlines():
        if "from recall" in line:
            continue
        if "recall" in line and "components" not in line and "F0.5" not in line:
            # every remaining recall line sits under the components caption,
            # which names the precision it must be read against
            assert "recall components" in text


def test_the_components_block_names_the_precision_it_is_read_against(loaded, split):
    text, record = _render(loaded, split, baselines.MovingAverage(60))
    precision = record.scorecards[0].pooled["event_precision"]
    assert f"read against precision {precision.brief()}" in text
    assert "never alone" in text


def test_the_adoption_number_is_always_present(loaded, split):
    """Rare-event false alarms appear on every scorecard, without exception."""
    for detector in (baselines.MovingAverage(60), baselines.AlwaysQuiet(),
                     baselines.RandomScore(1)):
        text, _ = _render(loaded, split, detector)
        assert "ADOPTION  rare-event false alarms" in text


def test_a_compact_count_still_carries_its_denominator():
    assert Count(6, 7).brief() == "6/7 (0.857)"
    assert Count(0, 0).brief() == "-/0"
