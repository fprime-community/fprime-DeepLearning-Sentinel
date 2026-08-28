"""The players. Detectors scored by the `sentinel_eval` harness.

The dependency runs one way only: models import
:class:`sentinel_eval.detector.Detector`, and the harness never imports a model.
That is the "referee before the players" principle expressed in the package
graph, and `tests/test_layering.py` enforces it -- so work items 4, 5 and 6 add
LSTM, GRU and TCN here without touching a line of the harness.

    baselines    the floor, the ceiling, and the one that must lose
    windows      cutting telemetry into sequences; shared by items 4-6
    lstm         the 2x80 forecaster, trained with PyTorch -- telemanom's LSTM
                 cell, or the GRU (work item 5), selected by `Hyper.cell`
    reference    the plain-NumPy forward pass -- the Phase 2 C++ blueprint,
                 both cells
    telemanom    smoothed errors, the dynamic threshold, pruning
    detectors    the two halves wired together as one `Detector`
    registry     name -> detector, resolved only at the composition root

**Training and scoring use different code paths on purpose.** Gradients come
from autograd, because a hand-written backward pass that is subtly wrong yields
a plausible bad number. Scoring runs through `reference`, because Phase 2's C++
needs something to verify itself against. `tests/test_reference_equivalence.py`
asserts they agree, and that assertion is the contract holding it together.
"""
from __future__ import annotations

__all__ = ["registry", "baselines", "windows", "lstm", "reference", "telemanom",
           "detectors"]
