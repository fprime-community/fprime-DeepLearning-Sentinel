"""D84 / `docs/MODELS.md` 77, SX11 and SX13: the retrainer is exported OPT-IN, and OFF means OFF.

`fprime/library.cmake` is what a mission consumes. D84 adds the retrainer to it
behind `SENTINEL_WITH_RETRAINER`, default OFF, and makes two promises this file
checks by CONFIGURING the real file rather than reading it:

1. **OFF -- and unset, which must mean OFF -- exports exactly what it did before
   D84**: `sentinel_core` and `Sentinel/Monitor`, nothing else.
2. **ON exports `Sentinel/Retrainer` as well**, and only after a platform check
   that refuses, with ONE message naming the supported platforms, any system,
   processor or cross-compile OxCaml does not support.

No F' is needed. The probe project stubs the three F' names `library.cmake`
touches -- `add_fprime_subdirectory`, and the `Fw_Types` target the core links --
and records every directory the library asks F' to register. `flight/` is added
for real, because the library adds it for real and it builds with no F' at all.

Both directions: two mutants of `library.cmake` (the default flipped to ON, and
the registration moved outside the `if`) must each be caught.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "fprime" / "library.cmake"
PLATFORM = ROOT / "fprime" / "Sentinel" / "Retrainer" / "retrainer_platform.cmake"

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake is not installed")

PROBE = """cmake_minimum_required(VERSION 3.26)
project(SentinelLibraryProbe LANGUAGES C CXX)
add_library(Fw_Types INTERFACE)
set_property(GLOBAL PROPERTY SENTINEL_PROBE_EXPORTS "")
function(add_fprime_subdirectory dir)
    get_filename_component(_d "${dir}" ABSOLUTE)
    set_property(GLOBAL APPEND PROPERTY SENTINEL_PROBE_EXPORTS "${_d}")
endfunction()
include("@LIBRARY@")
get_property(_e GLOBAL PROPERTY SENTINEL_PROBE_EXPORTS)
foreach(_x IN LISTS _e)
    message(STATUS "PROBE_EXPORT ${_x}")
endforeach()
if (TARGET sentinel_core)
    message(STATUS "PROBE_TARGET sentinel_core")
endif()
"""

#: What library.cmake exported before D84, as paths relative to fprime/.
BEFORE_D84 = {"Sentinel/Monitor"}

SUPPORTED_MESSAGE = ("The retrainer is built with OxCaml, which supports x86-64 and arm64 "
                     "Linux and arm64 macOS only")


def _exports(tmp_path: pathlib.Path, library: pathlib.Path, *defines: str):
    """Configure the probe around `library`; return (rc, exported dirs, has core, log)."""
    probe = tmp_path / "probe"
    probe.mkdir(exist_ok=True)
    (probe / "CMakeLists.txt").write_text(PROBE.replace("@LIBRARY@", str(library)),
                                          encoding="utf-8")
    build = tmp_path / "build"
    shutil.rmtree(build, ignore_errors=True)
    result = subprocess.run(["cmake", "-S", str(probe), "-B", str(build), *defines],
                            capture_output=True, text=True, timeout=600)
    log = result.stdout + result.stderr
    fprime_dir = library.parent
    exported = set()
    for line in re.findall(r"PROBE_EXPORT (.+)", log):
        exported.add(os.path.relpath(line.strip(), fprime_dir))
    return result.returncode, exported, "PROBE_TARGET sentinel_core" in log, log


def _mutant_library(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    """A mutated library.cmake at the same depth, beside links to the real trees."""
    root = tmp_path / "mutant"
    (root / "fprime").mkdir(parents=True)
    (root / "flight").symlink_to(ROOT / "flight")
    (root / "fprime" / "Sentinel").symlink_to(ROOT / "fprime" / "Sentinel")
    lib = root / "fprime" / "library.cmake"
    lib.write_text(text, encoding="utf-8")
    return lib


# -- SX11: OFF means OFF --------------------------------------------------------------

def test_unset_exports_exactly_what_it_did_before_d84(tmp_path: pathlib.Path) -> None:
    rc, exported, core, log = _exports(tmp_path, LIBRARY)
    assert rc == 0, log[-3000:]
    assert exported == BEFORE_D84, exported
    assert core, "sentinel_core is no longer brought by the library"


def test_off_exports_exactly_what_it_did_before_d84(tmp_path: pathlib.Path) -> None:
    rc, exported, core, log = _exports(tmp_path, LIBRARY, "-DSENTINEL_WITH_RETRAINER=OFF")
    assert rc == 0, log[-3000:]
    assert exported == BEFORE_D84 and core, exported


def test_on_exports_the_retrainer_as_well(tmp_path: pathlib.Path) -> None:
    """On this host (the only one the project has built on) the platform check passes."""
    rc, exported, core, log = _exports(tmp_path, LIBRARY, "-DSENTINEL_WITH_RETRAINER=ON")
    if rc != 0 and SUPPORTED_MESSAGE in " ".join(log.split()):
        pytest.skip("this host is not one OxCaml supports; the refusal is tested below")
    assert rc == 0, log[-3000:]
    assert exported == BEFORE_D84 | {"Sentinel/Retrainer"}, exported
    assert core


@pytest.mark.parametrize("mutation", [
    ("the default flipped to ON", lambda s: s.replace(
        '"Export the OxCaml retrainer (docs/DECISIONS.md D84). Needs scripts/oxcaml_setup.sh."\n'
        "       OFF)",
        '"Export the OxCaml retrainer (docs/DECISIONS.md D84). Needs scripts/oxcaml_setup.sh."\n'
        "       ON)")),
    ("the registration moved outside the if", lambda s: s.replace(
        '    add_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Sentinel/Retrainer")\nendif()\n',
        'endif()\nadd_fprime_subdirectory("${CMAKE_CURRENT_LIST_DIR}/Sentinel/Retrainer")\n')),
])
def test_the_off_check_catches_a_mutated_library(tmp_path: pathlib.Path, mutation) -> None:
    name, mutate = mutation
    original = LIBRARY.read_text(encoding="utf-8")
    mutated = mutate(original)
    assert mutated != original, f"the mutation '{name}' no longer applies to library.cmake"
    rc, exported, _, log = _exports(tmp_path, _mutant_library(tmp_path, mutated))
    assert exported != BEFORE_D84 or rc != 0, (
        f"library.cmake with {name} still looks like OFF to this guard: {exported}")


# -- SX13: ON where OxCaml does not run stops, with one message --------------------------

def _platform(tmp_path: pathlib.Path, system: str, processor: str, cross: str = "FALSE"):
    script = tmp_path / "platform.cmake"
    script.write_text(
        f'include("{PLATFORM}")\n'
        f'sentinel_retrainer_check_platform("{system}" "{processor}" "{cross}")\n'
        'message(STATUS "PLATFORM_OK")\n', encoding="utf-8")
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True,
                            timeout=120)
    return result.returncode, result.stdout + result.stderr


@pytest.mark.parametrize("system,processor", [
    ("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64"),
])
def test_a_supported_platform_passes(tmp_path: pathlib.Path, system: str, processor: str) -> None:
    rc, log = _platform(tmp_path, system, processor)
    assert rc == 0 and "PLATFORM_OK" in log, log


@pytest.mark.parametrize("system,processor,cross", [
    ("Darwin", "x86_64", "FALSE"),
    ("Linux", "armv7l", "FALSE"),
    ("Linux", "i686", "FALSE"),
    ("Windows", "AMD64", "FALSE"),
    ("Generic", "arm", "FALSE"),
    ("Linux", "aarch64", "TRUE"),
])
def test_an_unsupported_platform_stops_with_one_message(
        tmp_path: pathlib.Path, system: str, processor: str, cross: str) -> None:
    rc, log = _platform(tmp_path, system, processor, cross)
    assert rc != 0 and "PLATFORM_OK" not in log, log
    flat = " ".join(log.split())            # CMake wraps a long message across lines
    assert SUPPORTED_MESSAGE in flat, log
    assert flat.count("CMake Error") == 1, log
    assert f"not supported on {system} ({processor}" in flat, log
    assert "Set SENTINEL_WITH_RETRAINER OFF to build the detector alone" in flat, log
