"""(!) The C++ half of the allocation claim, checked by symbol rather than by reading.

`Objective.md` 11's CPP-1 forbids heap allocation after init, and the flight core is
written to it. Until 2026-09-22 that was enforced on the C++ side by **a human reading
the code** -- which is exactly the contrast `master:docs/DESIGN.md` draws when it argues
for OxCaml: *"in C++ that is checked by a human reading code, and OxCaml's
`[@zero_alloc strict]` makes it a compile error, transitively across callees."*

That sentence is fair, and it was also **asymmetric evidence**: the OxCaml side had a
compiler refusing builds and a ladder of guards, and the C++ side had a claim. This file
makes the comparison symmetric, using the technique the project already trusts --
`tests/test_detector_binary_has_no_ocaml_runtime.py` and
`scripts/fprime_ref_patch.sh:132-148` both prove things by symbol rather than by string.

**What this can and cannot say.** A linked binary that references no allocator symbol
cannot allocate through one: that is a real, checkable property of the artifact. It is
NOT the same as `[@zero_alloc strict]`, and the difference is the interesting part of
the comparison rather than something to paper over:

* `strict` is checked **at compile time, per function, transitively across callees**,
  and it fails the build. This scan runs **after linking, over a whole binary**, and it
  fails a test somebody remembered to write.
* `strict` says *this function performs no allocation on any path*. This says *the
  linked image contains no reference to these allocator symbols* -- which a stack
  allocation, a placement-new into a static buffer, or a custom pool would all satisfy.

Both directions are checked. The positive control is a translation unit that really does
call `new`: if the scan cannot see an allocator there, a clean result below means
nothing. That is the lesson 72.5 recorded when the OCaml-symbol guard was found to have
been vacuous rather than passing.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLIGHT_BUILD = ROOT / "flight" / "build"

#: (!) MANGLED AND UNMANGLED BOTH, AND THE LEADING UNDERSCORE MATTERS. macOS `nm`
#: prefixes every symbol with one, so an anchor that assumed the bare name would miss
#: every hit -- the same trap `test_detector_binary_has_no_ocaml_runtime.py:35-39`
#: records falling into. `_Znwm`/`_Znam` are `operator new` and `operator new[]`;
#: `_Zdl`/`_Zda` are the deletes. The C entry points are listed by name because a
#: custom allocator reaching libc would show up as those and not as the C++ ones.
ALLOCATOR = re.compile(
    r"(?<![A-Za-z0-9])_?(?:"
    r"_Zn[wa][jmy]"                 # operator new / new[]
    r"|_Zd[la]Pv"                   # operator delete / delete[]
    r"|malloc|calloc|realloc|free|posix_memalign|aligned_alloc|reallocf"
    r"|strdup|strndup|asprintf"     # allocate internally and are easy to reach for
    r")(?![A-Za-z0-9_])")

#: Every binary `flight/Makefile` builds. These link `sentinel_core` and nothing else
#: of this project's, so an allocator here is an allocator in the core or in the
#: harness around it -- and the harness is deliberately allocation-free too.
BINARIES = ("GoldenVectors", "RefusalTests", "DeterminismTest", "RoundTripDump",
            "Footprint", "BaselineVectors", "TrailingVectors", "ThresholdVectors")


def _symbols(binary: pathlib.Path) -> str:
    return subprocess.run(["nm", str(binary)], capture_output=True, text=True).stdout


def _require_nm() -> None:
    if shutil.which("nm") is None:
        pytest.skip("nm is not installed; this check needs a symbol table")


def _built() -> list[pathlib.Path]:
    return [FLIGHT_BUILD / name for name in BINARIES if (FLIGHT_BUILD / name).exists()]


def test_the_scan_can_see_an_allocator_when_one_is_there(tmp_path) -> None:
    """The positive control, and the reason the clean result below is worth anything.

    72.5's finding, applied before it can happen again: a guard that has only ever
    passed is not known to work, and this one would pass over an empty set if the
    pattern were wrong.
    """
    _require_nm()
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler on PATH")
    source = tmp_path / "allocates.cpp"
    source.write_text(
        "#include <cstddef>\n"
        "// Deliberately allocates. `volatile` and the return stop the optimiser\n"
        "// removing the call, which would make this control silently vacuous.\n"
        "int* leak(std::size_t n) { return new int[n]; }\n"
        "int main() { volatile int* p = leak(4); return p ? 0 : 1; }\n",
        encoding="utf-8")
    binary = tmp_path / "allocates"
    built = subprocess.run([compiler, "-std=c++14", "-O2", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, f"the control did not build:\n{built.stderr}"
    found = ALLOCATOR.findall(_symbols(binary))
    assert found, (
        "the scan found no allocator symbol in a binary that calls `new`. The pattern "
        "is blind, and every clean result in this file is worth nothing until it is "
        "fixed.")


def test_no_flight_binary_references_an_allocator() -> None:
    """CPP-1 on the C++ side, enforced rather than read."""
    _require_nm()
    built = _built()
    if not built:
        pytest.skip("flight/build is empty; run `make -C flight test` to build it")
    offenders = {}
    for binary in built:
        hits = sorted(set(ALLOCATOR.findall(_symbols(binary))))
        if hits:
            offenders[binary.name] = hits
    assert not offenders, (
        f"allocator symbols in the flight core's binaries: {offenders}. "
        "`Objective.md` 11 CPP-1 forbids heap allocation after init, and the core is "
        "written to it -- a reference here means something in `flight/src/` or its "
        "harness reaches an allocator, which is the claim `master:docs/DESIGN.md` "
        "makes when it contrasts C++ with OxCaml's compile-time check.")


def test_the_scan_actually_covered_the_binaries_it_claims() -> None:
    """(!) A scan over nothing passes.

    `flight/CMakeLists.txt` built six of the eight binaries `flight/Makefile` builds
    until 2026-09-22, so "every flight binary" was a smaller set than a reader would
    assume. This asserts the set is not silently empty and names what is missing.
    """
    _require_nm()
    built = _built()
    if not built:
        pytest.skip("flight/build is empty; run `make -C flight test` to build it")
    missing = [n for n in BINARIES if not (FLIGHT_BUILD / n).exists()]
    assert len(built) >= 6, (
        f"only {len(built)} of {len(BINARIES)} flight binaries are built ({missing} "
        "absent), so this scan covers less than it appears to. Build them with "
        "`make -C flight test`.")
