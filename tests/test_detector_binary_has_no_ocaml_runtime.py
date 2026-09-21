"""(!) The detector's process must never contain an OCaml runtime, and PUBLIC linkage
means one wiring mistake would put it there silently.

`fprime/SentinelRef/Retrainer/CMakeLists.txt:46` is

    target_link_libraries(${FPRIME_CURRENT_MODULE} PUBLIC "${OX_OBJ}")

and `${OX_OBJ}` is `cycle_complete.o`, built with `-output-complete-obj`, which carries a
whole OCaml runtime. **PUBLIC**, so anything that links the `Retrainer` module transitively
links the runtime. `SentinelRef` is clean today only because its topology never names
`Retrainer` and `register_fprime_deployment` depends on `_Top` alone.

`docs/DECISIONS.md` D70 consequence 2 makes the separate process a **requirement**, and
`docs/MODELS.md`'s 47.15b ground 1 refuses reading (b) precisely because it "puts the garbage
collector in the detector's process, on the process that owns the 1 Hz rate group".
`docs/MODELS.md` 61.5a narrows stop 32 to permit a separate retrainer deployment -- and a
narrowed stop that is only remembered is worse than one left too wide, so this is the guard
that proves the narrowing was not abused.

Checked by **symbol**, not by string, which is the technique
`scripts/fprime_ref_patch.sh:132-148` already uses to prove adoption rather than assert it.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FPRIME = ROOT / "fprime"

#: `caml_` covers the runtime C symbols and `camlXxx` the compiled modules. The optional
#: leading underscore matters: macOS `nm` prefixes every symbol with one, and a `\b`
#: anchor does NOT match between `_` and `caml` because `_` is a word character -- which
#: made the first version of this file blind, and the positive control below caught it.
OCAML_SYMBOL = re.compile(r"(?<![A-Za-z0-9])_?caml[A-Za-z_0-9]*")

DETECTOR_BINARIES = (
    FPRIME / "build-artifacts" / "Darwin" / "SentinelRef" / "bin" / "SentinelRef",
    FPRIME / "build-fprime-automatic-native" / "bin" / "Darwin" / "SentinelRef",
)

#: The unit-test executable DOES link ${OX_OBJ}. It is the positive control: if this file
#: cannot see the runtime here, it cannot see it anywhere, and a clean detector binary
#: would mean nothing.
POSITIVE_CONTROL = (FPRIME / "build-fprime-automatic-native-ut" / "bin" / "Darwin"
                    / "SentinelRef_Retrainer_ut_exe")


def _ocaml_symbols(binary: pathlib.Path) -> int:
    out = subprocess.run(["nm", "-C", str(binary)], capture_output=True, text=True)
    return len(OCAML_SYMBOL.findall(out.stdout))


def _require_nm():
    if shutil.which("nm") is None:
        pytest.skip("nm is not installed; this check needs a symbol table")


def test_the_check_can_see_an_ocaml_runtime_when_one_is_there() -> None:
    """Both directions. A guard that has only ever passed is not known to work."""
    _require_nm()
    if not POSITIVE_CONTROL.exists():
        pytest.skip(f"{POSITIVE_CONTROL.name} is not built; nothing to control against")
    found = _ocaml_symbols(POSITIVE_CONTROL)
    assert found > 100, (
        f"{POSITIVE_CONTROL.name} links ${{OX_OBJ}} and should be full of OCaml symbols; "
        f"this check found {found}. The check is blind, and the clean result below is "
        "worth nothing until it is fixed.")


def test_no_detector_binary_contains_an_ocaml_runtime() -> None:
    """D70 consequence 2 and 47.15b ground 1, enforced rather than remembered."""
    _require_nm()
    built = [b for b in DETECTOR_BINARIES if b.exists()]
    if not built:
        pytest.skip("no SentinelRef binary is built; run fprime-util build -p ./SentinelRef")
    offenders = {}
    for binary in built:
        found = _ocaml_symbols(binary)
        if found:
            offenders[str(binary.relative_to(ROOT))] = found
    assert not offenders, (
        f"OCaml symbols in the detector's own binary: {offenders}. "
        "fprime/SentinelRef/Retrainer/CMakeLists.txt:46 links ${OX_OBJ} PUBLIC, so this is "
        "what a transitive dependency on the Retrainer module looks like -- the garbage "
        "collector in the process that owns the 1 Hz rate group, which docs/DECISIONS.md "
        "D70 consequence 2 forbids and docs/MODELS.md 47.15b ground 1 refuses by name.")
