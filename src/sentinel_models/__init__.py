"""The players. Detectors scored by the `sentinel_eval` harness.

The dependency runs one way only: models import
:class:`sentinel_eval.detector.Detector`, and the harness never imports a model.
That is the "referee before the players" principle expressed in the package
graph, and `tests/test_layering.py` enforces it -- so work items 4, 5 and 6 add
LSTM, GRU and TCN here without touching a line of the harness.
"""
from __future__ import annotations

__all__ = ["registry", "baselines"]
