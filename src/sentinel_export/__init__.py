"""The ground-segment model-file writer, and the reader that mirrors the flight one.

`model.bin` is the contract between the Python training toolkit and the C++ flight
loader (Objective.md 14.2). It is frozen at version 1 by `docs/DECISIONS.md` D30 and
specified byte for byte in `docs/MODEL_FILE.md`, **which is normative**: where this
package and that document disagree, the document is right and this package has a
defect.

This package has its own home for a reason that has not changed:

* `sentinel_eval` is the referee and must not know a weight format exists;
* `sentinel_models` will keep changing while the file format must be frozen.

**It depends on nothing else in this repository** -- standard library and numpy only.
It takes a plain weights-and-metadata dictionary and writes the binary, so the format
is defined by the written specification rather than by a Python class. That matters
because the C++ loader is written against that specification and cannot import
anything from here. No Python, no ML framework and no interpreter goes to space;
flight review boards approve a C++ program reading a data file, which is how F's own
parameter database already works.

`reader.read_model` returns a `Status` rather than raising, and its values are shared
with the C++ `LoadStatus`, so one test can assert both sides refuse the same bytes for
the same reason. `writer.replace_params` performs the in-orbit recalibration the
separately-CRC'd parameter block exists for: the weights are neither read nor
rewritten.
"""
from __future__ import annotations

from .format import Status
from .reader import read_model
from .writer import ModelFileError, replace_params, write_model

__all__ = ["ModelFileError", "Status", "read_model", "replace_params", "write_model"]
