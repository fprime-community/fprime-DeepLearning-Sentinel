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
    for suite in ("footprint", "refusals", "determinism", "golden vectors", "baseline"):
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


def test_the_types_shim_is_the_only_file_that_knows_about_fprime():
    """D35's claim, asserted rather than trusted.

    Work item 9 puts the core under F' by selecting types in one header. The
    promise D31 consequence 2 was protecting is that nothing else changes, so
    `Types.hpp` must be the only file under `flight/` that includes an F' header
    or tests the build's selector macro. A comment naming F' is fine -- most of
    the core carries one -- so this looks for the `#include` and the macro, not
    for the word.
    """
    shim = FLIGHT / "include" / "sentinel" / "Types.hpp"
    include = re.compile(r'^\s*#\s*include\s*[<"]Fw/', re.MULTILINE)
    selector = re.compile(r"^\s*#\s*(if|ifdef|elif).*SENTINEL_FPRIME_TYPES", re.MULTILINE)

    offenders = []
    for path in sorted((FLIGHT / "src").glob("*.cpp")) + \
            sorted((FLIGHT / "include" / "sentinel").glob("*.hpp")) + \
            sorted((FLIGHT / "test").glob("*.cpp")) + \
            sorted((FLIGHT / "test").glob("*.hpp")):
        if path == shim:
            continue
        text = path.read_text()
        if include.search(text) or selector.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == [], f"only Types.hpp may know about F': {offenders}"

    text = shim.read_text()
    assert selector.search(text), "Types.hpp must select on SENTINEL_FPRIME_TYPES"
    assert include.search(text), "Types.hpp must include an F' header on the F' side"


def test_the_shim_asserts_the_float_properties_rather_than_a_macro():
    """`FW_HAS_F64` does not exist in F' v4.3.0, so the shim must not test it.

    `F64` is unconditional at `Fw/Types/BasicTypes.h:86`; the macro appears once
    in the whole framework, in a documentation table. Asserting the property is
    what the dtype map in `docs/MODEL_FILE.md` 9 actually needs. See
    `docs/MODELS.md` 20.2 correction 13.
    """
    text = (FLIGHT / "include" / "sentinel" / "Types.hpp").read_text()
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("//"))
    assert "FW_HAS_F64" not in code, "the shim must not branch on a macro F' does not define"
    for needed in ("sizeof(F32) == 4U", "sizeof(F64) == 8U",
                   "std::numeric_limits<F64>::is_iec559"):
        assert needed in code, f"{needed} is not asserted in Types.hpp"


@needs_toolchain
def test_the_core_compiles_against_fprime_types_when_the_checkout_is_present():
    """The other half of `both worlds, one core` -- skipped without a checkout.

    Compiles every core translation unit with `SENTINEL_FPRIME_TYPES` defined,
    at the same flag set the freestanding build uses. `scripts/fprime_setup.sh`
    provides the checkout; without it there is nothing to compile against and the
    test skips rather than failing on a machine that has not run setup.
    """
    fprime = ROOT / "fprime" / "lib" / "fprime"
    if not (fprime / "Fw" / "FPrimeBasicTypes.hpp").exists():
        pytest.skip("no F' checkout; run scripts/fprime_setup.sh")
    builds = sorted((fprime / "TestDeploymentsProject").glob("build-fprime-automatic-*"))
    generated = next((b for b in builds if (b / "F-Prime" / "default").is_dir()), None)
    if generated is None:
        pytest.skip("no generated F' build cache to supply config headers")

    compiler = shutil.which("clang++") or shutil.which("g++")
    flags = ["-std=c++14", "-fno-exceptions", "-fno-rtti", "-ffp-contract=off",
             "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", "-Werror",
             "-O2", "-DSENTINEL_FPRIME_TYPES",
             f"-I{FLIGHT / 'include'}", f"-I{fprime}",
             f"-I{fprime / 'cmake' / 'platform' / 'unix'}",
             f"-I{generated / 'F-Prime' / 'default'}", f"-I{generated / 'F-Prime'}",
             f"-I{generated}"]
    for source in sorted((FLIGHT / "src").glob("*.cpp")):
        result = subprocess.run([compiler, *flags, "-c", str(source), "-o", "/dev/null"],
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, (
            f"{source.name} does not compile against F' types:\n{result.stderr}")
