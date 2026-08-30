"""The ground-segment model-file writer. Phase 2 -- placeholder, nothing built.

Objective.md 14.2 requires the `model.bin` format frozen **before Phase 2 starts**,
because it is the contract between the Python training toolkit and the C++ flight
loader. It gets its own package for a reason that will not change:

* `sentinel_eval` is the referee and must not know a weight format exists;
* `sentinel_models` will keep changing while the file format must be frozen.

**This package will depend on nothing else in this repository.** It takes a plain
weights-and-metadata dictionary and writes the binary, so the format is defined
by a written specification -- `docs/MODEL_FILE.md`, which this package will own --
rather than by a Python class. That matters because the C++ loader is written
against that specification and cannot import anything from here. No Python, no ML
framework and no interpreter goes to space; flight review boards approve a C++
program reading a data file, which is how F's own parameter database already
works.

Nothing here is implemented in Phase 1, deliberately: the format cannot be frozen
until the architecture gate has chosen what goes in it.
"""
from __future__ import annotations

__all__: list[str] = []
