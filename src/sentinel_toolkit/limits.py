"""The flight component's compile-time maxima, in one place.

These mirror `flight/include/sentinel/Config.hpp` and are the values
`src/sentinel_export/reader.py:67-71` compares against to return `TOO_LARGE`.

**(!) They lived only in a test before this.** `tests/test_model_file.py` held
the single Python copy, so the ground side had no home for them and a third
copy was the obvious next step. `tests/test_toolkit.py` asserts this dict
against `Config.hpp` itself, so the copy is checked rather than remembered.
"""
from __future__ import annotations

#: Mirrors `flight/include/sentinel/Config.hpp`. Checked by test, not trusted.
FLIGHT_LIMITS = {
    "MAX_CHANNELS": 16,
    "MAX_LAYERS": 2,
    "MAX_HIDDEN": 80,
    "MAX_PREDICTIONS": 10,
    "MAX_INPUTS": 16,
}

#: (!) The trailing span the fused statistic is standardised over is **not a
#: field in the model file**. `flight/src/DerivativeStream.cpp:12` configures its
#: window with `Config::ERROR_WINDOW` directly, and `Detector::configure` passes
#: it no value from the file. So a mission does not choose it and neither does
#: this toolkit: calibrating at any other span would derive a cut for a statistic
#: the flight component does not compute. Section 40.3a.
FLIGHT_ERROR_WINDOW = 2100

#: `ewma_span`, by contrast, **is** a file field and is read --
#: `flight/src/Detector.cpp:23` configures the EWMA from `m_model.ewmaSpan`, and
#: `ModelFile.cpp:251` refuses a zero. The toolkit may choose it, and ships
#: telemanom's published 105.
DEFAULT_EWMA_SPAN = 105
