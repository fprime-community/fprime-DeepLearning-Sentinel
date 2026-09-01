"""The C++ core, run from `pytest`, so the verification trio still covers it.

`make -C flight test` is the primary route; these tests shell out to the same
target so `pytest -q` alone does not silently skip the flight half. They skip
rather than fail when no C++ compiler is present, because the Python side must
stay runnable on a machine that cannot build the core.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sentinel_export import Status, read_model

ROOT = Path(__file__).resolve().parents[1]
FLIGHT = ROOT / "flight"
VECTORS = FLIGHT / "test" / "vectors"

needs_toolchain = pytest.mark.skipif(
    shutil.which("clang++") is None and shutil.which("g++") is None,
    reason="no C++ compiler on this machine")


def run(*args, **kwargs):
    return subprocess.run(args, cwd=str(FLIGHT), capture_output=True, text=True,
                          timeout=600, **kwargs)


@pytest.fixture(scope="module")
def built():
    """Build once for the module. Returns the `make test` output."""
    if shutil.which("clang++") is None and shutil.which("g++") is None:
        pytest.skip("no C++ compiler on this machine")
    result = run("make", "test")
    if result.returncode != 0:
        pytest.fail(f"make -C flight test failed:\n{result.stdout}\n{result.stderr}")
    return result


@needs_toolchain
def test_the_core_builds_with_no_warnings_at_all(built):
    """`-Werror` at -Wall -Wextra -Wpedantic -Wconversion -Wshadow.

    The build must be silent, not merely successful: a warning is an error, so a
    successful `make` is itself the assertion. This checks the flags were really
    the ones used, in case the Makefile drifts.
    """
    fresh = run("make", "clean")
    assert fresh.returncode == 0
    result = run("make", "all")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    for flag in ("-Werror", "-Wconversion", "-Wshadow", "-Wpedantic",
                 "-fno-exceptions", "-fno-rtti", "-ffp-contract=off"):
        assert flag in combined, f"{flag} was not on the command line"
    assert "warning:" not in combined.lower(), combined


@needs_toolchain
def test_every_flight_suite_passes(built):
    for suite in ("footprint", "refusals", "determinism", "golden vectors"):
        assert f"{suite}: all checks passed" in built.stdout, built.stdout
    assert "FAIL" not in built.stdout, built.stdout


@needs_toolchain
def test_the_golden_vectors_are_within_the_pre_registered_tolerance(built):
    """`docs/MODELS.md` 19, prediction F5: <= 1e-5 on state and forecast."""
    worst = [float(m) for m in re.findall(r"max \|diff\| ([0-9.e+-]+)", built.stdout)]
    assert worst, built.stdout
    assert max(worst) <= 1e-5, f"worst stage difference {max(worst):.3e}"


@needs_toolchain
def test_the_crossing_flag_matched_exactly_on_every_step(built):
    assert "crossing flag differs" not in built.stdout
    assert "emitted flag differs" not in built.stdout


@needs_toolchain
def test_the_core_is_deterministic_across_processes(built):
    assert "across processes:" in built.stdout
    matched = re.search(r"across processes: (0x[0-9A-F]{8}) == (0x[0-9A-F]{8})",
                        built.stdout)
    assert matched is not None, built.stdout
    assert matched.group(1) == matched.group(2)


@needs_toolchain
@pytest.mark.parametrize("tier", ("g1", "g2"))
def test_python_to_cpp_to_python_round_trips_byte_identically(tier, built, tmp_path):
    """The C++ re-emits the file from what it parsed; Python compares the bytes.

    `docs/MODELS.md` 19, prediction F6. If a single field were misread, the
    re-emitted file would differ -- and the CRCs are recomputed by the C++ from
    its own arrays, so a wrong weight cannot hide behind a copied checksum.
    """
    source = VECTORS / f"{tier}.bin"
    out = tmp_path / f"{tier}.rt"
    result = run("./build/RoundTripDump", str(source), str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_bytes() == source.read_bytes(), "re-emitted bytes differ"

    status, spec = read_model(out.read_bytes())
    assert status is Status.OK
    assert spec["params"]["threshold"] == read_model(source.read_bytes())[1]["params"]["threshold"]


@needs_toolchain
def test_the_footprint_is_what_the_pre_registration_predicted(built):
    """312,642 bytes of declared members was committed before the code existed."""
    matched = re.search(r"sizeof\(Detector\)\s+(\d+) B", built.stdout)
    assert matched is not None, built.stdout
    measured = int(matched.group(1))
    predicted = 312_642
    assert abs(measured - predicted) / predicted < 0.01, (
        f"measured {measured:,} against {predicted:,} predicted")


def test_the_cmake_flags_match_the_makefile_exactly():
    """Work item 9's build must not be quieter than the one work item 8 verified.

    Runs without a compiler: it compares two text files.
    """
    makefile = (FLIGHT / "Makefile").read_text()
    cmake = (FLIGHT / "CMakeLists.txt").read_text()
    required = ("-fno-exceptions", "-fno-rtti", "-ffp-contract=off", "-Wall",
                "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", "-Werror")
    for flag in required:
        assert flag in makefile, f"{flag} missing from flight/Makefile"
        assert flag in cmake, f"{flag} missing from flight/CMakeLists.txt"
    for forbidden in ("-ffast-math", "-Ofast"):
        assert f"{forbidden}\n" not in makefile.replace("never ", "")
        assert "FATAL_ERROR" in cmake, "CMake must refuse the forbidden flags"


def test_the_status_codes_agree_between_python_and_cpp():
    """One test, both sides. A file refused here is refused there for the same reason."""
    header = (FLIGHT / "include" / "sentinel" / "Status.hpp").read_text()
    for status in Status:
        pattern = rf"\b{status.name}\s*=\s*{status.value}U\b"
        assert re.search(pattern, header), (
            f"{status.name} = {status.value} is not in LoadStatus")
